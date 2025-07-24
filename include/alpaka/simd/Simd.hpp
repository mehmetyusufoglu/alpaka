/* Copyright 2025 Your Name
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Traits.hpp"
#include "alpaka/core/Common.hpp"
#include "alpaka/core/Vectorize.hpp"

#include <type_traits>

// Forward declare accelerator types
namespace alpaka
{
    template<typename TDim, typename TIdx>
    class AccCpuSerial;
    template<typename TDim, typename TIdx>
    class AccCpuThreads;
    template<typename TDim, typename TIdx>
    class AccCpuOmp2Blocks;
    template<typename TDim, typename TIdx>
    class AccCpuOmp2Threads;
    template<typename TDim, typename TIdx>
    class AccCpuTbbBlocks;
    template<typename TDim, typename TIdx>
    class AccGpuCudaRt;
    template<typename TDim, typename TIdx>
    class AccGpuHipRt;
} // namespace alpaka

namespace alpaka::simd
{
    //! Primary template for portable SIMD operations
    //! Provides a unified interface for SIMD operations across different accelerators
    template<typename T, typename TAcc>
    class PortableSimd;

} // namespace alpaka::simd

// Include accelerator-specific implementations
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#    include "alpaka/simd/detail/SimdCpu.hpp"
#endif

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#    include "alpaka/simd/detail/SimdGpu.hpp"
#endif

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
#    include "alpaka/simd/detail/SimdGpu.hpp"
#endif

// Include traits
#include "alpaka/simd/SimdTraits.hpp"

namespace alpaka::simd
{
    //! Type alias for easier usage
    template<typename T, typename TAcc>
    using Simd = PortableSimd<T, TAcc>;

    //! Get the SIMD width for a given type and accelerator
    template<typename T, typename TAcc>
    struct SimdWidth
    {
        static constexpr std::size_t value = PortableSimd<T, TAcc>::size();
    };

} // namespace alpaka::simd
