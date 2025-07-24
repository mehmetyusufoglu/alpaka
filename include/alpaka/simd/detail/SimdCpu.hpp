/* Copyright 2025 Your Name
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/acc/AccCpuSerial.hpp"

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
#    include "alpaka/acc/AccCpuThreads.hpp"
#endif

#if defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
#    include "alpaka/acc/AccCpuOmp2Blocks.hpp"
#endif

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)
#    include "alpaka/acc/AccCpuOmp2Threads.hpp"
#endif

#if defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
#    include "alpaka/acc/AccCpuTbbBlocks.hpp"
#endif
#include "alpaka/simd/Simd.hpp"

#include <experimental/simd>

namespace stdx = std::experimental;

namespace alpaka::simd
{
    //! CPU SIMD implementation using std::experimental::simd
    template<typename T, typename TDim, typename TIdx>
    class PortableSimd<T, AccCpuSerial<TDim, TIdx>>
    {
        stdx::simd<T> data;

        // Private constructor for internal use
        explicit PortableSimd(stdx::simd<T> const& simd_data) : data(simd_data)
        {
        }

    public:
        //! Number of elements that can be processed in parallel
        static constexpr std::size_t size()
        {
            return stdx::simd<T>::size();
        }

        //! Default constructor - initializes to zero
        ALPAKA_FN_ACC PortableSimd() : data(stdx::simd<T>(0))
        {
        }

        //! Scalar constructor - broadcasts scalar to all SIMD lanes
        ALPAKA_FN_ACC explicit PortableSimd(T scalar) : data(scalar)
        {
        }

        //! Load data from aligned memory
        void load(T const* ptr)
        {
            data.copy_from(ptr, stdx::element_aligned);
        }

        //! Store data to aligned memory
        void store(T* ptr) const
        {
            data.copy_to(ptr, stdx::element_aligned);
        }

        //! Addition operator
        ALPAKA_FN_ACC PortableSimd operator+(PortableSimd const& other) const
        {
            return PortableSimd(data + other.data);
        }

        //! Subtraction operator
        ALPAKA_FN_ACC PortableSimd operator-(PortableSimd const& other) const
        {
            return PortableSimd(data - other.data);
        }

        //! Multiplication operator
        ALPAKA_FN_ACC PortableSimd operator*(PortableSimd const& other) const
        {
            return PortableSimd(data * other.data);
        }

        //! Division operator
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

        //! Horizontal sum of all elements
        T sum() const
        {
            return stdx::reduce(data);
        }

        //! Access individual element (for debugging/testing)
        T operator[](std::size_t idx) const
        {
            return data[idx];
        }
    };

    // Template specializations for other CPU accelerators, guarded by backend macros

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
    template<typename T, typename TDim, typename TIdx>
    class PortableSimd<T, AccCpuThreads<TDim, TIdx>> : public PortableSimd<T, AccCpuSerial<TDim, TIdx>>
    {
        using Base = PortableSimd<T, AccCpuSerial<TDim, TIdx>>;

    public:
        using Base::Base;
    };
#endif

#if defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
    template<typename T, typename TDim, typename TIdx>
    class PortableSimd<T, AccCpuOmp2Blocks<TDim, TIdx>> : public PortableSimd<T, AccCpuSerial<TDim, TIdx>>
    {
        using Base = PortableSimd<T, AccCpuSerial<TDim, TIdx>>;

    public:
        using Base::Base;
    };
#endif

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)
    template<typename T, typename TDim, typename TIdx>
    class PortableSimd<T, AccCpuOmp2Threads<TDim, TIdx>> : public PortableSimd<T, AccCpuSerial<TDim, TIdx>>
    {
        using Base = PortableSimd<T, AccCpuSerial<TDim, TIdx>>;

    public:
        using Base::Base;
    };
#endif

#if defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
    template<typename T, typename TDim, typename TIdx>
    class PortableSimd<T, AccCpuTbbBlocks<TDim, TIdx>> : public PortableSimd<T, AccCpuSerial<TDim, TIdx>>
    {
        using Base = PortableSimd<T, AccCpuSerial<TDim, TIdx>>;

    public:
        using Base::Base;
    };
#endif

} // namespace alpaka::simd
