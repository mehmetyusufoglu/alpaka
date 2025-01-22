#ifndef SIMD_LIBRARY_HPP
#define SIMD_LIBRARY_HPP

#include <cstddef>

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#include <cuda_runtime.h>
#endif

// CPU SIMD implementation
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
#include <experimental/simd>
namespace stdx = std::experimental;

template <typename T>
class PortableSimdCpu {
    stdx::simd<T> data;

public:
    static constexpr size_t size() {
        return stdx::simd<T>::size();
    }

    void load(const T* ptr) {
        data = stdx::simd<T>(ptr, stdx::element_aligned);
    }

    PortableSimdCpu& operator=(const PortableSimdCpu& other) {
        if (this != &other) {
            data = other.data;
        }
        return *this;
    }

    PortableSimdCpu operator+(const PortableSimdCpu& other) const {
        PortableSimdCpu result;
        result.data = data + other.data;
        return result;
    }

    PortableSimdCpu operator*(const PortableSimdCpu& other) const {
        PortableSimdCpu result;
        result.data = data * other.data;
        return result;
    }

    void store(T* ptr) const {
        data.copy_to(ptr, stdx::element_aligned);
    }

    T sum() const {
        T result = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            result += data[i];
        }
        return result;
    }
};
#endif // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

// CUDA SIMD implementation
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
template <typename T>
class PortableSimdCuda {
    T data;

public:
    ALPAKA_FN_ACC static constexpr size_t size() {
        return 1; // Scalar for CUDA
    }

    ALPAKA_FN_ACC void load(const T* ptr) {
        data = *ptr;
    }

    ALPAKA_FN_ACC PortableSimdCuda& operator=(const PortableSimdCuda& other) {
        if (this != &other) {
            data = other.data;
        }
        return *this;
    }

    ALPAKA_FN_ACC PortableSimdCuda operator+(const PortableSimdCuda& other) const {
        PortableSimdCuda result;
        result.data = data + other.data;
        return result;
    }

    ALPAKA_FN_ACC PortableSimdCuda operator*(const PortableSimdCuda& other) const {
        PortableSimdCuda result;
        result.data = data * other.data;
        return result;
    }

    ALPAKA_FN_ACC void store(T* ptr) const {
        *ptr = data;
    }

    ALPAKA_FN_ACC T sum() const {
        return data; // Scalar sum
    }
};
#endif // ALPAKA_ACC_GPU_CUDA_ENABLED

#endif // SIMD_LIBRARY_HPP
