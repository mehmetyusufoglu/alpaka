#ifndef SIMD_LIBRARY_HPP
#define SIMD_LIBRARY_HPP

#include <experimental/simd>
#include <cstddef>

namespace stdx = std::experimental;

template <typename T>
class PortableSimd {
    stdx::simd<T> data;

public:
    static constexpr size_t size() {
        return stdx::simd<T>::size();
    }
    void load(const T* ptr) {
        data = stdx::simd<T>(ptr, stdx::element_aligned);
    }

    // Assignment operator
    PortableSimd& operator=(const PortableSimd& other) {
        if (this != &other) {
            data = other.data;
        }
        return *this;
    }

    // Operator := for loading data
    PortableSimd& operator==(const T* ptr) {
        data = stdx::simd<T>(ptr, stdx::element_aligned);
        return *this;
    }

    // Perform addition
    PortableSimd operator+(const PortableSimd& other) const {
        PortableSimd result;
        result.data = data + other.data;
        return result;
    }

    // Perform multiplication
    PortableSimd operator*(const PortableSimd& other) const {
        PortableSimd result;
        result.data = data * other.data;
        return result;
    }

    // Store the data back to memory
    void store(T* ptr) const {
        data.copy_to(ptr, stdx::element_aligned);
    }

    // Compute the sum of all elements in the SIMD object
    T sum() const {
        T result = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            result += data[i];
        }
        return result;
    }

    // Apply a custom operation using a functor
    template <typename Functor>
    PortableSimd apply(const PortableSimd& other, Functor func) const {
        PortableSimd result;
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = func(data[i], other.data[i]);
        }
        return result;
    }
};

#endif // SIMD_LIBRARY_HPP
