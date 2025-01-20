#ifndef PORTABLE_SIMD_HPP
#define PORTABLE_SIMD_HPP

#include <alpaka/alpaka.hpp>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#include <experimental/simd>
#endif
#include <type_traits>
#include <numeric>

namespace detail {
    // Helper type trait to determine if the Acc type corresponds to CUDA.
    template <typename Acc>
    struct IsCudaAcc {
        static constexpr bool value = std::is_same_v<typename alpaka::trait::AccToTag<Acc>::type, alpaka::TagGpuCudaRt>;
    };
} // namespace detail

template <typename T, typename Acc>
class PortableSimd {
private:
    using SimdType = std::conditional_t<
        detail::IsCudaAcc<Acc>::value,
        T,                              // For CUDA, PortableSimd<T> is just T
#ifndef __CUDACC__
        std::experimental::simd<T>>;    // For CPU, use std::simd
#else
        T>;                             // Fallback for CUDA
#endif

    SimdType data;

public:
    // Default constructor
    ALPAKA_FN_ACC PortableSimd() {
        if constexpr (detail::IsCudaAcc<Acc>::value) {
            data = T{}; // Scalar initialization to 0
        } else {
#ifndef __CUDACC__
            data = std::experimental::simd<T>(); // Zero-initialize SIMD register
#endif
        }
    }

           // Constructor to initialize all elements to a scalar value
    ALPAKA_FN_ACC explicit PortableSimd(T scalar) {
        if constexpr (detail::IsCudaAcc<Acc>::value) {
            data = scalar; // Scalar initialization
        } else {
#ifndef __CUDACC__
            data = std::experimental::simd<T>(scalar); // SIMD register initialization
#endif
        }
    }

           // Get the SIMD width
    ALPAKA_FN_ACC static constexpr size_t size() {
        if constexpr (detail::IsCudaAcc<Acc>::value) {
            return 1; // Scalar for CUDA
        } else {
#ifndef __CUDACC__
            return std::experimental::simd<T>::size();
#else
            return 1; // Fallback for CUDA
#endif
        }
    }

           // Load from memory
    ALPAKA_FN_ACC void load(const T* ptr) {
        if constexpr (detail::IsCudaAcc<Acc>::value) {
            data = *ptr; // Scalar load
        } else {
#ifndef __CUDACC__
            data = std::experimental::simd<T>(ptr, std::experimental::element_aligned);
#endif
        }
    }

           // Store to memory
    ALPAKA_FN_ACC void store(T* ptr) const {
        if constexpr (detail::IsCudaAcc<Acc>::value) {
            *ptr = data; // Scalar store
        } else {
#ifndef __CUDACC__
            data.copy_to(ptr, std::experimental::element_aligned);
#endif
        }
    }

           // Perform addition
    ALPAKA_FN_ACC PortableSimd operator+(const PortableSimd& other) const {
        PortableSimd result;
        if constexpr (detail::IsCudaAcc<Acc>::value) {
            result.data = data + other.data; // Scalar addition
        } else {
#ifndef __CUDACC__
            result.data = data + other.data; // SIMD addition
#endif
        }
        return result;
    }

           // Perform multiplication
    ALPAKA_FN_ACC PortableSimd operator*(const PortableSimd& other) const {
        PortableSimd result;
        if constexpr (detail::IsCudaAcc<Acc>::value) {
            result.data = data * other.data; // Scalar multiplication
        } else {
#ifndef __CUDACC__
            result.data = data * other.data; // SIMD multiplication
#endif
        }
        return result;
    }

           // Compute the sum of all elements in the SIMD object
    ALPAKA_FN_ACC T sum() const {
        if constexpr (detail::IsCudaAcc<Acc>::value) {
            return data; // Scalar sum
        } else {
#ifndef __CUDACC__
             T sum{};
            for(int i=0;i<size();i++)
            sum+=data[i];
            return sum;
#else
            return data; // Fallback for CUDA
#endif
        }
    }
};

#endif // PORTABLE_SIMD_HPP
