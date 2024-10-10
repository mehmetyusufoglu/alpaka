/* Copyright 2024 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "helpers.hpp"

#include <alpaka/alpaka.hpp>

//! alpaka version of explicit finite-difference 2D heat equation solver
//!
//! \tparam T_SharedMemSize1D size of the shared memory box
//!
//! Solving equation u_t(x, t) = u_xx(x, t) + u_yy(y, t) using a simple explicit scheme with
//! forward difference in t and second-order central difference in x and y
//!
//! \param uCurrBuf Current buffer with grid values of u for each x, y pair and the current value of t:
//!                 u(x, y, t) | t = t_current
//! \param uNextBuf resulting grid values of u for each x, y pair and the next value of t:
//!              u(x, y, t) | t = t_current + dt
//! \param chunkSize The size of the chunk or tile that the user divides the problem into. This defines the size of the
//!                  workload handled by each thread block.
//! \param pitchCurr The pitch (or stride) in memory corresponding to the TDim grid in the accelerator's memory.
//!              This is used to calculate memory offsets when accessing elements in the current buffer.
//! \param pitchNext The pitch used to calculate memory offsets when accessing elements in the next buffer.
//! \param dx step in x
//! \param dy step in y
//! \param dt step in t
template<size_t T_SharedMemSize1D>
struct StencilKernel
{
    template<typename TAcc, typename TDim, typename TIdx>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        double const* const uCurrBuf,
        double* const uNextBuf,
        alpaka::Vec<TDim, TIdx> const chunkSize,
        alpaka::Vec<TDim, TIdx> const pitchCurr,
        alpaka::Vec<TDim, TIdx> const pitchNext,
        double const dx,
        double const dy,
        double const dt) const -> void
    {
        auto& sdata = alpaka::declareSharedVar<double[T_SharedMemSize1D], __COUNTER__>(acc);

        // Get extents(dimensions)
        auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        auto const numThreadsPerBlock = blockThreadExtent.prod();

        // Get indexes
        auto const gridBlockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        auto const blockThreadIdx2D = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const blockThreadIdx1D = alpaka::mapIdx<1>(blockThreadIdx2D, blockThreadExtent)[0u];
        // Each block is responsible from a chunk of data
        auto const blockStartIdxInData = gridBlockIdx * chunkSize;

        constexpr alpaka::Vec<TDim, TIdx> halo{2, 2};

        for(auto i = blockThreadIdx1D; i < T_SharedMemSize1D; i += numThreadsPerBlock)
        {
            auto idxData2D = alpaka::mapIdx<2>(alpaka::Vec(i), chunkSize + halo);
            idxData2D = idxData2D + blockStartIdxInData;
            auto elem = getElementPtr(uCurrBuf, idxData2D, pitchCurr);
            sdata[i] = *elem;
        }

        alpaka::syncBlockThreads(acc);

        // Each kernel executes one element
        double const rX = dt / (dx * dx);
        double const rY = dt / (dy * dy);


        // go over only core cells
        for(auto i = blockThreadIdx2D[0]; i < chunkSize[0]; i += blockThreadExtent[0])
        {
            for(auto j = blockThreadIdx2D[1]; i < chunkSize[1]; i += blockThreadExtent[1])
            {
                // offset for halo, data index in buffer includes halo/2 border
                auto localDataIdx2D = alpaka::Vec(i, j) + alpaka::Vec<TDim, TIdx>{1, 1};
                auto const globalDataIdx2D = localDataIdx2D + blockStartIdxInData;
                auto elem = getElementPtr(uNextBuf, globalDataIdx2D, pitchNext);

                // convert indexes to 1D for shared memory access
                auto localDataIdx1D = alpaka::mapIdx<1>(localDataIdx2D, chunkSize + halo)[0u];

                *elem = sdata[localDataIdx1D] * (1.0 - 2.0 * rX - 2.0 * rY) + sdata[localDataIdx1D - 1] * rX
                        + sdata[localDataIdx1D + 1] * rX + sdata[localDataIdx1D - chunkSize[1] - halo[1]] * rY
                        + sdata[localDataIdx1D + chunkSize[1] + halo[1]] * rY;
            }
        }
    }
};
