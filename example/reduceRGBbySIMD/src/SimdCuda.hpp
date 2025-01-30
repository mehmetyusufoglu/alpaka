#ifndef SIMDCUDA_HPP
#define SIMDCUDA_HPP

// CUDA specialization --------------------------------------------------------
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
template <typename T, typename TDim, typename TIdx>
class PortableSimd<T, alpaka::AccGpuCudaRt<TDim, TIdx>>
{
    T data;

public:
    static constexpr size_t size() { return 1; }

    ALPAKA_FN_ACC PortableSimd() : data(0) {}
    ALPAKA_FN_ACC explicit PortableSimd(T scalar) : data(scalar) {}

    ALPAKA_FN_ACC void load(const T* ptr) { data = *ptr; }
    ALPAKA_FN_ACC void store(T* ptr) const { *ptr = data; }

    ALPAKA_FN_ACC PortableSimd operator+(const PortableSimd& other) const {
        return PortableSimd(data + other.data);
    }

    ALPAKA_FN_ACC PortableSimd operator*(const PortableSimd& other) const {
        return PortableSimd(data * other.data);
    }

    ALPAKA_FN_ACC T sum() const { return data; }
};

#endif // CUDA specialization
#endif // SIMDCUDA_HPP
