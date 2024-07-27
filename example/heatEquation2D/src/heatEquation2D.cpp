/* Copyright 2020 Benjamin Worpitz, Matthias Werner, Jakob Krude, Sergei Bastrakov, Bernhard Manfred Gruber
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>
#include <pngwriter.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>


struct HeatEquation2DKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        double const* const uCurrBuf,
        double* const uNextBuf,
        uint32_t const extentX,
        uint32_t const extentY,
        double const dx,
        double const dy,
        double const dt) const -> void
    {
        // Each kernel executes one element
        double const rx = dt / (dx * dx);
        double const ry = dt / (dy * dy);
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];

        if(i > 0 && i < extentX - 1u && j > 0 && j < extentY - 1u)
        {
            uNextBuf[i * extentY + j] = uCurrBuf[i * extentY + j] * (1.0 - 2.0 * (rx + ry))
                                        + uCurrBuf[(i - 1) * extentY + j] * rx + uCurrBuf[(i + 1) * extentY + j] * rx
                                        + uCurrBuf[i * extentY + (j - 1)] * ry + uCurrBuf[i * extentY + (j + 1)] * ry;
        }
    }
};

// Exact solution to the test problem
auto exactSolution(double const x, double const y, double const t) -> double
{
    constexpr double pi = 3.14159265358979323846;
    return std::exp(-2 * pi * pi * t) * std::sin(pi * x) * std::sin(pi * y);
}

template<typename TAccTag>
auto example(TAccTag const&) -> int
{
    uint32_t const numNodesX = 256;
    uint32_t const numNodesY = 256;
    uint32_t const numTimeSteps = 1000;
    double const tMax = 0.01;
    // x, y in [0, 1], t in [0, tMax]
    double const dx = 1.0 / static_cast<double>(numNodesX - 1);
    double const dy = 1.0 / static_cast<double>(numNodesY - 1);
    double const dt = 0.5 * std::min(dx, dy) * std::min(dx, dy); // Adjusted dt for stability

           // Check the stability condition
    double const r = dt / (dx * dx) + dt / (dy * dy);
    if(r > 0.5)
    {
        std::cerr << "Stability condition check failed: dt/(dx^2) + dt/(dy^2) = " << r << ", it is required to be <= 0.5\n";
        return EXIT_FAILURE;
    }

           // Set Dim and Idx type
    using Dim = alpaka::DimInt<2u>;
    using Idx = uint32_t;
    alpaka::Vec<Dim, Idx> const extent2D;

           // Define the accelerator
    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

           // Select specific devices
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);
    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

           // Select queue
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    QueueAcc queue{devAcc};

           // Initialize host buffers
    auto uNextBufHost = alpaka::allocBuf<double, Idx>(devHost, extent2D);
    auto uCurrBufHost = alpaka::allocBuf<double, Idx>(devHost, extent2D);

    double* const pCurrHost = alpaka::getPtrNative(uCurrBufHost);
    double* const pNextHost = alpaka::getPtrNative(uNextBufHost);

           // Initialize accelerator buffers
    using BufAcc = alpaka::Buf<Acc, double, Dim, Idx>;
    auto uNextBufAcc = alpaka::allocBuf<double, Idx>(devAcc, extent2D);
    auto uCurrBufAcc = alpaka::allocBuf<double, Idx>(devAcc, extent2D);

    double* pCurrAcc = alpaka::getPtrNative(uCurrBufAcc);
    double* pNextAcc = alpaka::getPtrNative(uNextBufAcc);


           // Apply initial conditions for the test problem
    for(uint32_t i = 0; i < numNodesX; ++i)
    {
        for(uint32_t j = 0; j < numNodesY; ++j)
        {
            pCurrHost[i * numNodesY + j] = exactSolution(i * dx, j * dy, 0.0);
        }
    }

           // Copy host -> device
    alpaka::memcpy(queue, uCurrBufAcc, uCurrBufHost);
    alpaka::memcpy(queue, uNextBufAcc, uCurrBufAcc);
    alpaka::wait(queue);

           // Kernel setup
    HeatEquation2DKernel heatEqKernel;
    auto const & kernelBundle = alpaka::KernelBundle(heatEqKernel, std::data(uCurrBufAcc), std::data(uNextBufAcc), numNodesX, numNodesY, dx, dy, dt);

    auto workDiv = alpaka::getValidWorkDivForKernel<Acc>(
        devAcc, kernelBundle,
        extent2D,
        alpaka::Vec<Dim, Idx>::ones(),
        false,
        alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);

    for(uint32_t step = 0; step < numTimeSteps; ++step)
    {
        // Compute next values
        alpaka::exec<Acc>(queue, workDiv, heatEqKernel, pCurrAcc, pNextAcc, numNodesX, numNodesY, dx, dy, dt);

               // We assume the boundary conditions are constant and so these values
               // do not need to be updated.
       std::swap(pCurrAcc, pNextAcc);
    }

           // Copy device -> host
    alpaka::memcpy(queue, uNextBufHost, uNextBufAcc);
    alpaka::wait(queue);

           // Calculate error
    double maxError = 0.0;
    for(uint32_t i = 0; i < numNodesX; ++i)
    {
        for(uint32_t j = 0; j < numNodesY; ++j)
        {
            auto const error = std::abs(pNextHost[i * numNodesY + j] - exactSolution(i * dx, j * dy, tMax));
            maxError = std::max(maxError, error);
        }
    }

    double const errorThreshold = 1e-5;
    bool resultCorrect = (maxError < errorThreshold);
    if(resultCorrect)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Execution results incorrect: error = " << maxError << " (the grid resolution may be too low)"
                  << std::endl;
        return EXIT_FAILURE;
    }
}

auto main() -> int
{
    // Execute the example once for each enabled accelerator.
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}
