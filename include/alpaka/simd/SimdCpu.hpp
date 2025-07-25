#pragma once

#include "alpaka/simd/Simd.hpp"
#include "alpaka/simd/SimdTypeTraits.hpp"

#include <immintrin.h>

#include <cstdint>

namespace alpaka::simd::detail
{

#if defined(__AVX512F__)
    template<typename T>
    struct simd_native_type;

    template<>
    struct simd_native_type<float>
    {
        using type = __m512;
        static constexpr std::size_t width = 16;
    };

    template<>
    struct simd_native_type<double>
    {
        using type = __m512d;
        static constexpr std::size_t width = 8;
    };

    template<typename T>
    struct simd_native_type_integral
    {
        using type = __m512i;
        static constexpr std::size_t width = 64 / sizeof(T);
    };

    template<>
    struct simd_native_type<std::int32_t> : simd_native_type_integral<std::int32_t>
    {
    };

    template<>
    struct simd_native_type<std::int64_t> : simd_native_type_integral<std::int64_t>
    {
    };

#elif defined(__AVX2__)
    template<typename T>
    struct simd_native_type
    {
        using type = std::conditional_t<
            std::is_same_v<T, float>,
            __m256,
            std::conditional_t<std::is_same_v<T, double>, __m256d, __m256i>>;
        static constexpr std::size_t width = sizeof(__m256i) / sizeof(T);
    };
#else
#    error "No supported SIMD instruction set available (requires at least AVX2)"
#endif

    template<typename T, typename TAcc>
    class SimdCpuImpl
    {
        using native_type = typename simd_native_type<T>::type;

    public:
        static constexpr std::size_t width = simd_native_type<T>::width;

        ALPAKA_FN_HOST_ACC SimdCpuImpl() = default;

        ALPAKA_FN_HOST_ACC explicit SimdCpuImpl(T value)
        {
#if defined(__AVX512F__)
            if constexpr(std::is_same_v<T, float>)
                data_ = _mm512_set1_ps(value);
            else if constexpr(std::is_same_v<T, double>)
                data_ = _mm512_set1_pd(value);
            else if constexpr(std::is_integral_v<T>)
                data_ = _mm512_set1_epi64(value);
#elif defined(__AVX2__)
            if constexpr(std::is_same_v<T, float>)
                data_ = _mm256_set1_ps(value);
            else if constexpr(std::is_same_v<T, double>)
                data_ = _mm256_set1_pd(value);
            else if constexpr(std::is_integral_v<T>)
                data_ = _mm256_set1_epi64x(value);
#endif
        }

        // Load/Store operations
        ALPAKA_FN_ACC void load_aligned(T const* ptr)
        {
#if defined(__AVX512F__)
            if constexpr(std::is_same_v<T, float>)
                data_ = _mm512_load_ps(ptr);
            else if constexpr(std::is_same_v<T, double>)
                data_ = _mm512_load_pd(ptr);
            else if constexpr(std::is_integral_v<T>)
                data_ = _mm512_load_epi64(ptr);
#elif defined(__AVX2__)
            if constexpr(std::is_same_v<T, float>)
                data_ = _mm256_load_ps(ptr);
            else if constexpr(std::is_same_v<T, double>)
                data_ = _mm256_load_pd(ptr);
            else if constexpr(std::is_integral_v<T>)
                data_ = _mm256_load_si256(reinterpret_cast<__m256i const*>(ptr));
#endif
        }

        ALPAKA_FN_ACC void store_aligned(T* ptr) const
        {
#if defined(__AVX512F__)
            if constexpr(std::is_same_v<T, float>)
                _mm512_store_ps(ptr, data_);
            else if constexpr(std::is_same_v<T, double>)
                _mm512_store_pd(ptr, data_);
            else if constexpr(std::is_integral_v<T>)
                _mm512_store_epi64(ptr, data_);
#elif defined(__AVX2__)
            if constexpr(std::is_same_v<T, float>)
                _mm256_store_ps(ptr, data_);
            else if constexpr(std::is_same_v<T, double>)
                _mm256_store_pd(ptr, data_);
            else if constexpr(std::is_integral_v<T>)
                _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), data_);
#endif
        }

        // Arithmetic operations
        ALPAKA_FN_ACC SimdCpuImpl operator+(SimdCpuImpl const& rhs) const
        {
            SimdCpuImpl result;
#if defined(__AVX512F__)
            if constexpr(std::is_same_v<T, float>)
                result.data_ = _mm512_add_ps(data_, rhs.data_);
            else if constexpr(std::is_same_v<T, double>)
                result.data_ = _mm512_add_pd(data_, rhs.data_);
            else if constexpr(std::is_integral_v<T>)
                result.data_ = _mm512_add_epi64(data_, rhs.data_);
#elif defined(__AVX2__)
            if constexpr(std::is_same_v<T, float>)
                result.data_ = _mm256_add_ps(data_, rhs.data_);
            else if constexpr(std::is_same_v<T, double>)
                result.data_ = _mm256_add_pd(data_, rhs.data_);
            else if constexpr(std::is_integral_v<T>)
                result.data_ = _mm256_add_epi64(data_, rhs.data_);
#endif
            return result;
        }

        // Horizontal operations
        ALPAKA_FN_ACC T hadd() const
        {
#if defined(__AVX512F__)
            if constexpr(std::is_same_v<T, float>)
                return _mm512_reduce_add_ps(data_);
            else if constexpr(std::is_same_v<T, double>)
                return _mm512_reduce_add_pd(data_);
            else if constexpr(std::is_integral_v<T>)
                return _mm512_reduce_add_epi64(data_);
#elif defined(__AVX2__)
            if constexpr(std::is_same_v<T, float>)
            {
                __m128 sum = _mm_add_ps(_mm256_extractf128_ps(data_, 0), _mm256_extractf128_ps(data_, 1));
                sum = _mm_hadd_ps(sum, sum);
                sum = _mm_hadd_ps(sum, sum);
                return _mm_cvtss_f32(sum);
            }
            else if constexpr(std::is_same_v<T, double>)
            {
                __m128d sum = _mm_add_pd(_mm256_extractf128_pd(data_, 0), _mm256_extractf128_pd(data_, 1));
                sum = _mm_hadd_pd(sum, sum);
                return _mm_cvtsd_f64(sum);
            }
            else if constexpr(std::is_integral_v<T>)
            {
                __m128i sum = _mm_add_epi64(_mm256_extracti128_si256(data_, 0), _mm256_extracti128_si256(data_, 1));
                return _mm_cvtsi128_si64(sum) + _mm_cvtsi128_si64(_mm_srli_si128(sum, 8));
            }
#endif
            static_assert(false, "No SIMD support available");
        }

    private:
        native_type data_;
    };

} // namespace alpaka::simd::detail
