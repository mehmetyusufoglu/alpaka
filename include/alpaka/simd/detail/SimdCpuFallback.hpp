/* Copyright 2025 Your Name
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"

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
} // namespace alpaka

namespace alpaka::simd
{
    //! Fallback CPU SIMD implementation (scalar operations when std::experimental::simd unavailable)
    //! Used when compiling with NVCC which doesn't support std::experimental::simd
    template<typename T, typename TDim, typename TIdx>
    class PortableSimd<T, AccCpuSerial<TDim, TIdx>>
    {
        T data;

    public:
        //! SIMD width is 1 for fallback scalar operations
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

        //! Load single element (fallback to scalar)
        ALPAKA_FN_ACC void load(T const* ptr)
        {
            data = *ptr;
        }

        //! Store single element (fallback to scalar)
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

    //! Other CPU accelerators inherit from the base implementation
    template<typename T, typename TDim, typename TIdx>
    class PortableSimd<T, AccCpuThreads<TDim, TIdx>> : public PortableSimd<T, AccCpuSerial<TDim, TIdx>>
    {
        using Base = PortableSimd<T, AccCpuSerial<TDim, TIdx>>;

    public:
        using Base::Base;
    };

    template<typename T, typename TDim, typename TIdx>
    class PortableSimd<T, AccCpuOmp2Blocks<TDim, TIdx>> : public PortableSimd<T, AccCpuSerial<TDim, TIdx>>
    {
        using Base = PortableSimd<T, AccCpuSerial<TDim, TIdx>>;

    public:
        using Base::Base;
    };

    template<typename T, typename TDim, typename TIdx>
    class PortableSimd<T, AccCpuOmp2Threads<TDim, TIdx>> : public PortableSimd<T, AccCpuSerial<TDim, TIdx>>
    {
        using Base = PortableSimd<T, AccCpuSerial<TDim, TIdx>>;

    public:
        using Base::Base;
    };

    template<typename T, typename TDim, typename TIdx>
    class PortableSimd<T, AccCpuTbbBlocks<TDim, TIdx>> : public PortableSimd<T, AccCpuSerial<TDim, TIdx>>
    {
        using Base = PortableSimd<T, AccCpuSerial<TDim, TIdx>>;

    public:
        using Base::Base;
    };

} // namespace alpaka::simd
