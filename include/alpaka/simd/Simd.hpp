/* Copyright 2025 Your Name
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/Traits.hpp"
#include "alpaka/core/Common.hpp"
#include "alpaka/core/Vectorize.hpp"

#include <cstdint>
#include <type_traits>

namespace alpaka::simd
{
    // Forward declare mask type
    template<typename T, typename TAcc>
    class SimdMask;

    //! Primary template for portable SIMD operations
    //! Provides a unified interface for SIMD operations across different accelerators
    template<typename T, typename TAcc>
    class alignas(32) PortableSimd
    {
    public:
        using value_type = T;
        using mask_type = SimdMask<T, TAcc>;

        // Standard operations are implemented in backend-specific files

        // Horizontal operations
        ALPAKA_FN_ACC T hadd() const; // Horizontal add (sum of all elements)
        ALPAKA_FN_ACC T hmin() const; // Horizontal minimum
        ALPAKA_FN_ACC T hmax() const; // Horizontal maximum
        ALPAKA_FN_ACC T hmul() const; // Horizontal multiply

        // Masked operations
        ALPAKA_FN_ACC void masked_store(T* ptr, mask_type const& mask) const;
        ALPAKA_FN_ACC static PortableSimd masked_load(T const* ptr, mask_type const& mask);

        // Gather/Scatter operations
        template<typename IndexType>
        ALPAKA_FN_ACC static PortableSimd gather(T const* base, IndexType const* indices);
        template<typename IndexType>
        ALPAKA_FN_ACC void scatter(T* base, IndexType const* indices) const;

        // Memory prefetch hint
        ALPAKA_FN_ACC static void prefetch(T const* ptr, int hint = 0);

        // Comparison operations returning masks
        ALPAKA_FN_ACC mask_type operator<(PortableSimd const& other) const;
        ALPAKA_FN_ACC mask_type operator>(PortableSimd const& other) const;
        ALPAKA_FN_ACC mask_type operator<=(PortableSimd const& other) const;
        ALPAKA_FN_ACC mask_type operator>=(PortableSimd const& other) const;

        // Load/Store with alignment hints
        ALPAKA_FN_ACC void load_aligned(T const* ptr);
        ALPAKA_FN_ACC void store_aligned(T* ptr) const;
    };

} // namespace alpaka::simd

// Include accelerator-specific implementations
// Conditional compilation based on compiler and enabled accelerators
#ifdef __NVCC__
  // NVCC doesn't support std::experimental::simd, use scalar fallback for CPU backends
#    include "alpaka/simd/detail/SimdCpuFallback.hpp"
#else
  // Use actual vector SIMD for CPU backends
#    ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#        include "alpaka/simd/detail/SimdCpu.hpp"
#    endif
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
