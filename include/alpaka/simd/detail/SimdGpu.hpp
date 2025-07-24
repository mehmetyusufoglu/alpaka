/* Copyright 2025 Your Name
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccGpuCudaRt.hpp"
#include "alpaka/acc/AccGpuHipRt.hpp"
#include "alpaka/simd/Simd.hpp"

namespace alpaka::simd
{
    //! GPU SIMD implementation - uses scalar operations (SIMD width = 1)
    //! GPUs handle parallelism through many threads rather than SIMD within threads
    template<typename T, typename TDim, typename TIdx>
    class PortableSimd<T, AccGpuCudaRt<TDim, TIdx>>
    {
        T data;

    public:
        //! SIMD width is 1 for GPU scalar operations
        static constexpr std::size_t size()
        {
            return 1;
        }

        //! Default constructor
        ALPAKA_FN_ACC PortableSimd() : data(T{0})
        {
        }

        //! Scalar constructor
        ALPAKA_FN_ACC explicit PortableSimd(T scalar) : data(scalar)
        {
        }

        //! Load single element
        ALPAKA_FN_ACC void load(T const* ptr)
        {
            data = *ptr;
        }

        //! Store single element
        ALPAKA_FN_ACC void store(T* ptr) const
        {
            *ptr = data;
        }

        //! Arithmetic operators (scalar operations)
        ALPAKA_FN_ACC PortableSimd operator+(PortableSimd const& other) const
        {
            return PortableSimd(data + other.data);
        }

        ALPAKA_FN_ACC PortableSimd operator-(PortableSimd const& other) const
        {
            return PortableSimd(data - other.data);
        }

        ALPAKA_FN_ACC PortableSimd operator*(PortableSimd const& other) const
        {
            return PortableSimd(data * other.data);
        }

        ALPAKA_FN_ACC PortableSimd operator/(PortableSimd const& other) const
        {
            return PortableSimd(data / other.data);
        }

        //! Compound assignment operators
        ALPAKA_FN_ACC PortableSimd& operator+=(PortableSimd const& other)
        {
            data += other.data;
            return *this;
        }

        ALPAKA_FN_ACC PortableSimd& operator-=(PortableSimd const& other)
        {
            data -= other.data;
            return *this;
        }

        ALPAKA_FN_ACC PortableSimd& operator*=(PortableSimd const& other)
        {
            data *= other.data;
            return *this;
        }

        ALPAKA_FN_ACC PortableSimd& operator/=(PortableSimd const& other)
        {
            data /= other.data;
            return *this;
        }

        //! Sum returns the single scalar value
        ALPAKA_FN_ACC T sum() const
        {
            return data;
        }

        //! Element access (only element 0 exists)
        ALPAKA_FN_ACC T operator[](std::size_t idx) const
        {
            return data; // idx should always be 0
        }
    };

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    //! HIP specialization - same as CUDA
    template<typename T, typename TDim, typename TIdx>
    class PortableSimd<T, AccGpuHipRt<TDim, TIdx>> : public PortableSimd<T, AccGpuCudaRt<TDim, TIdx>>
    {
        using Base = PortableSimd<T, AccGpuCudaRt<TDim, TIdx>>;

    public:
        using Base::Base;
    };
#endif

} // namespace alpaka::simd
