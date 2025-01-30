#ifndef SIMDCPUSEQ_H
#define SIMDCPUSEQ_H
// CPU specializations --------------------------------------------------------
#if  defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)

#include <experimental/simd> // Only include for CPU backend
namespace stdx = std::experimental;

// CPU specialization
template <typename T, typename TDim, typename TIdx>
class PortableSimd<T, alpaka::AccCpuSerial<TDim, TIdx>>
{
    stdx::simd<T> data;

public:
    static constexpr size_t size() { return stdx::simd<T>::size(); }

           // Constructors
    ALPAKA_FN_ACC PortableSimd() : data(0) {}
    ALPAKA_FN_ACC explicit PortableSimd(T scalar) : data(scalar) {}
    ALPAKA_FN_ACC explicit PortableSimd(stdx::simd<T> const& simd) : data(simd) {}

           // Load/store operations
    ALPAKA_FN_ACC void load(const T* ptr) {
        data = stdx::simd<T>(ptr, stdx::element_aligned);
    }

    ALPAKA_FN_ACC void store(T* ptr) const {
        data.copy_to(ptr, stdx::element_aligned);
    }

           // Arithmetic operators
    ALPAKA_FN_ACC PortableSimd operator+(const PortableSimd& other) const {
        return PortableSimd(data + other.data);
    }

    ALPAKA_FN_ACC PortableSimd operator*(const PortableSimd& other) const {
        return PortableSimd(data * other.data);
    }

           // Summation
    ALPAKA_FN_ACC T sum() const {
        return stdx::reduce(data);
    }
};

#endif // CPU specialization

#endif // SIMDCPUSEQ_H
