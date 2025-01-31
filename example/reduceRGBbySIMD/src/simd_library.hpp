#pragma once
#include <alpaka/alpaka.hpp>

#include <type_traits>

namespace trait
{
    // Primary template declaration (intentionally undefined)
    template<typename Acc, typename T, typename TDim, typename TIdx>
    class PortableSimd;
} // namespace trait

// CPU specializations --------------------------------------------------------
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
#    if !BOOST_LANG_CUDA
#        include <experimental/simd> // Only include for CPU backend
namespace stdx = std::experimental;

namespace trait
{
    // CPU specialization
    template<typename T, typename TDim, typename TIdx>
    class PortableSimd<alpaka::AccCpuSerial<TDim, TIdx>, T, TDim, TIdx>
    {
        stdx::simd<T> data;

    public:
        static constexpr size_t size()
        {
            return stdx::simd<T>::size();
        }

        // Constructors
        ALPAKA_FN_ACC PortableSimd() : data(0)
        {
        }

        ALPAKA_FN_ACC explicit PortableSimd(T scalar) : data(scalar)
        {
        }

        ALPAKA_FN_ACC explicit PortableSimd(stdx::simd<T> const& simd) : data(simd)
        {
        }

        // Load/store operations
        ALPAKA_FN_ACC void load(T const* ptr)
        {
            data = stdx::simd<T>(ptr, stdx::element_aligned);
        }

        ALPAKA_FN_ACC void store(T* ptr) const
        {
            data.copy_to(ptr, stdx::element_aligned);
        }

        // Arithmetic operators
        ALPAKA_FN_ACC PortableSimd operator+(PortableSimd const& other) const
        {
            return PortableSimd(data + other.data);
        }

        ALPAKA_FN_ACC PortableSimd operator*(PortableSimd const& other) const
        {
            return PortableSimd(data * other.data);
        }

        ALPAKA_FN_ACC PortableSimd operator/(PortableSimd const& other) const
        {
            return PortableSimd(data / other.data);
        }

        // Bitwise shift operators
        ALPAKA_FN_ACC PortableSimd operator<<(unsigned int shift) const
        {
            return PortableSimd(data << shift);
        }

        ALPAKA_FN_ACC PortableSimd operator>>(unsigned int shift) const
        {
            return PortableSimd(data >> shift);
        }

        // Bitwise AND operator
        ALPAKA_FN_ACC PortableSimd operator&(PortableSimd const& other) const
        {
            return PortableSimd(data & other.data);
        }

        // Bitwise OR operator
        ALPAKA_FN_ACC PortableSimd operator|(PortableSimd const& other) const
        {
            return PortableSimd(data | other.data);
        }

        // Summation
        ALPAKA_FN_ACC T sum() const
        {
            return stdx::reduce(data);
        }
    };
} // namespace trait
#    endif
#endif // CPU specialization

// CUDA specialization --------------------------------------------------------
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#    if BOOST_LANG_CUDA
namespace trait
{
    template<typename T, typename TDim, typename TIdx>
    class PortableSimd<alpaka::AccGpuCudaRt<TDim, TIdx>, T, TDim, TIdx>
    {
        T data;

    public:
        static constexpr size_t size()
        {
            return 1;
        }

        ALPAKA_FN_ACC PortableSimd() : data(0)
        {
        }

        ALPAKA_FN_ACC explicit PortableSimd(T scalar) : data(scalar)
        {
        }

        ALPAKA_FN_ACC void load(T const* ptr)
        {
            data = *ptr;
        }

        ALPAKA_FN_ACC void store(T* ptr) const
        {
            *ptr = data;
        }

        ALPAKA_FN_ACC PortableSimd operator+(PortableSimd const& other) const
        {
            return PortableSimd(data + other.data);
        }

        ALPAKA_FN_ACC PortableSimd operator*(PortableSimd const& other) const
        {
            return PortableSimd(data * other.data);
        }

        ALPAKA_FN_ACC PortableSimd operator/(PortableSimd const& other) const
        {
            return PortableSimd(data / other.data);
        }

        // Bitwise shift operators
        ALPAKA_FN_ACC PortableSimd operator<<(unsigned int shift) const
        {
            return PortableSimd(data << shift);
        }

        ALPAKA_FN_ACC PortableSimd operator>>(unsigned int shift) const
        {
            return PortableSimd(data >> shift);
        }

        // Bitwise AND operator
        ALPAKA_FN_ACC PortableSimd operator&(PortableSimd const& other) const
        {
            return PortableSimd(data & other.data);
        }

        // Bitwise OR operator
        ALPAKA_FN_ACC PortableSimd operator|(PortableSimd const& other) const
        {
            return PortableSimd(data | other.data);
        }

        ALPAKA_FN_ACC T sum() const
        {
            return data;
        }
    };
} // namespace trait
#    endif
#endif // CUDA specialization

template<typename T, typename Acc>
using PortableSimd = typename trait::PortableSimd<Acc, T, typename alpaka::Dim<Acc>, typename alpaka::Idx<Acc>>;
