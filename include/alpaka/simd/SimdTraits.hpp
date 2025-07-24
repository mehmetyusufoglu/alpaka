/* Copyright 2025 Your Name
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"

#include <concepts>
#include <type_traits>

// Forward declarations
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
} // namespace alpaka

namespace alpaka::simd
{
    //! Trait to determine if an accelerator supports SIMD operations
    template<typename TAcc>
    struct IsSimdSupported : std::false_type
    {
    };

    //! Trait specializations for CPU accelerators that support SIMD
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    template<typename TDim, typename TIdx>
    struct IsSimdSupported<AccCpuSerial<TDim, TIdx>> : std::true_type
    {
    };
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED
    template<typename TDim, typename TIdx>
    struct IsSimdSupported<AccCpuThreads<TDim, TIdx>> : std::true_type
    {
    };
#endif

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    template<typename TDim, typename TIdx>
    struct IsSimdSupported<AccCpuOmp2Blocks<TDim, TIdx>> : std::true_type
    {
    };
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
    template<typename TDim, typename TIdx>
    struct IsSimdSupported<AccCpuOmp2Threads<TDim, TIdx>> : std::true_type
    {
    };
#endif

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
    template<typename TDim, typename TIdx>
    struct IsSimdSupported<AccCpuTbbBlocks<TDim, TIdx>> : std::true_type
    {
    };
#endif

    //! Helper variable template
    template<typename TAcc>
    inline constexpr bool is_simd_supported_v = IsSimdSupported<TAcc>::value;

    //! Concept to check if an accelerator supports SIMD
    template<typename TAcc>
    concept SimdCapable = IsSimdSupported<TAcc>::value;

    //! Get the optimal SIMD type for a given element type and accelerator
    template<typename T, typename TAcc>
    using SimdType = PortableSimd<T, TAcc>;

    //! Helper to get SIMD width at compile time
    template<typename T, typename TAcc>
    struct SimdWidthTrait
    {
        static constexpr std::size_t value = PortableSimd<T, TAcc>::size();
    };

    template<typename T, typename TAcc>
    inline constexpr std::size_t simd_width_v = SimdWidthTrait<T, TAcc>::value;

    //! Helper function to determine if a size is SIMD-aligned
    template<typename T, typename TAcc>
    ALPAKA_FN_HOST_ACC constexpr bool isSimdAligned(std::size_t size)
    {
        return size % simd_width_v<T, TAcc> == 0;
    }

    //! Helper to round up to next SIMD-aligned size
    template<typename T, typename TAcc>
    ALPAKA_FN_HOST_ACC constexpr std::size_t roundUpToSimdWidth(std::size_t size)
    {
        constexpr auto width = simd_width_v<T, TAcc>;
        return ((size + width - 1) / width) * width;
    }

    //! Helper to get the number of SIMD operations needed for a given size
    template<typename T, typename TAcc>
    ALPAKA_FN_HOST_ACC constexpr std::size_t getSimdOpCount(std::size_t size)
    {
        constexpr auto width = simd_width_v<T, TAcc>;
        return (size + width - 1) / width;
    }

} // namespace alpaka::simd
