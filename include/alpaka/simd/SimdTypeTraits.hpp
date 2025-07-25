#pragma once

#include "alpaka/simd/Simd.hpp"

#include <type_traits>

namespace alpaka::simd
{

    //! Type trait to check if a type is a SIMD type
    template<typename T>
    struct is_simd_type : std::false_type
    {
    };

    template<typename T, typename TAcc>
    struct is_simd_type<PortableSimd<T, TAcc>> : std::true_type
    {
    };

    template<typename T>
    inline constexpr bool is_simd_type_v = is_simd_type<T>::value;

    //! Type trait for promoting types in mixed SIMD operations
    template<typename T1, typename T2>
    struct simd_promote
    {
        using type = decltype(std::declval<T1>() + std::declval<T2>());
    };

    template<typename T1, typename T2>
    using simd_promote_t = typename simd_promote<T1, T2>::type;

    //! Get the optimal SIMD width for a given type and accelerator
    template<typename T, typename TAcc>
    struct simd_width
    {
#if defined(__AVX512F__)
        static constexpr size_t value = 64 / sizeof(T);
#elif defined(__AVX2__) || defined(__AVX__)
        static constexpr size_t value = 32 / sizeof(T);
#elif defined(__SSE2__)
        static constexpr size_t value = 16 / sizeof(T);
#else
        static constexpr size_t value = 1;
#endif
    };

    template<typename T, typename TAcc>
    inline constexpr size_t simd_width_v = simd_width<T, TAcc>::value;

} // namespace alpaka::simd
