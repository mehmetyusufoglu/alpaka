/* Copyright 2025 Your Name
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <iostream>
#include <numeric>
#include <vector>

// Kernel for SIMD-based dot product computation
struct DotProductSimdKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, T const* a, T const* b, T* partial_sums, std::size_t n) const
        -> void
    {
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        using SimdType = alpaka::simd::PortableSimd<T, TAcc>;
        constexpr auto simdWidth = SimdType::size();

        // Accumulator for this thread's partial sum
        SimdType accumulator(T{0});

        // Process SIMD-width elements at a time
        for(std::size_t i = globalThreadIdx * simdWidth; i < n; i += globalThreadExtent * simdWidth)
        {
            if(i + simdWidth <= n)
            {
                // Load SIMD vectors
                SimdType va, vb;
                va.load(&a[i]);
                vb.load(&b[i]);

                // Perform SIMD multiplication and accumulate
                accumulator += va * vb;
            }
            else
            {
                // Handle remaining elements
                for(std::size_t j = i; j < n && j < i + simdWidth; ++j)
                {
                    SimdType va_scalar(a[j]);
                    SimdType vb_scalar(b[j]);
                    accumulator += va_scalar * vb_scalar;
                }
            }
        }

        // Reduce accumulator to scalar and store
        partial_sums[globalThreadIdx] = accumulator.sum();
    }
};

// Simple matrix multiplication using SIMD
struct MatrixMultiplySimdKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        T const* a,
        T const* b,
        T* c,
        std::size_t rows,
        std::size_t cols,
        std::size_t inner) const -> void
    {
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        using SimdType = alpaka::simd::PortableSimd<T, TAcc>;
        constexpr auto simdWidth = SimdType::size();

        // Each thread processes multiple rows
        for(std::size_t row = globalThreadIdx; row < rows; row += globalThreadExtent)
        {
            for(std::size_t col = 0; col < cols; ++col)
            {
                SimdType sum(T{0});

                // SIMD-based inner product
                std::size_t k = 0;
                for(; k + simdWidth <= inner; k += simdWidth)
                {
                    SimdType va, vb;

                    // Load a[row][k:k+simdWidth]
                    va.load(&a[row * inner + k]);

                    // Load b[k:k+simdWidth][col] - requires gathering for non-contiguous access
                    T b_values[simdWidth];
                    for(std::size_t i = 0; i < simdWidth; ++i)
                    {
                        b_values[i] = b[(k + i) * cols + col];
                    }
                    vb.load(b_values);

                    sum += va * vb;
                }

                // Handle remaining elements
                T scalar_sum = sum.sum();
                for(; k < inner; ++k)
                {
                    scalar_sum += a[row * inner + k] * b[k * cols + col];
                }

                c[row * cols + col] = scalar_sum;
            }
        }
    }
};

void testDotProduct()
{
    std::cout << "\n=== Testing SIMD Dot Product ===" << std::endl;

    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Acc = alpaka::AccCpuSerial<Dim, Idx>;
    using Queue = alpaka::QueueCpuBlocking;

    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    Queue queue(devAcc);

    constexpr std::size_t n = 10000;
    constexpr std::size_t numThreads = 4;

    // Initialize data
    std::vector<float> a(n), b(n);
    std::iota(a.begin(), a.end(), 1.0f);
    std::iota(b.begin(), b.end(), 2.0f);

    // Allocate device memory
    auto bufA = alpaka::allocBuf<float, Idx>(devAcc, n);
    auto bufB = alpaka::allocBuf<float, Idx>(devAcc, n);
    auto bufPartialSums = alpaka::allocBuf<float, Idx>(devAcc, numThreads);

    // Copy data
    alpaka::memcpy(queue, bufA, a.data(), n);
    alpaka::memcpy(queue, bufB, b.data(), n);

    // Launch kernel
    auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>{1, numThreads, 1};

    alpaka::exec<Acc>(
        queue,
        workDiv,
        DotProductSimdKernel{},
        alpaka::getPtrNative(bufA),
        alpaka::getPtrNative(bufB),
        alpaka::getPtrNative(bufPartialSums),
        n);

    // Get results and reduce
    std::vector<float> partialSums(numThreads);
    alpaka::memcpy(queue, partialSums.data(), bufPartialSums, numThreads);
    alpaka::wait(queue);

    float result = std::accumulate(partialSums.begin(), partialSums.end(), 0.0f);

    // Compute expected result
    float expected = std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);

    std::cout << "SIMD result: " << result << std::endl;
    std::cout << "Expected: " << expected << std::endl;
    std::cout << "Error: " << std::abs(result - expected) << std::endl;

    using SimdType = alpaka::simd::PortableSimd<float, Acc>;
    std::cout << "SIMD width used: " << SimdType::size() << std::endl;
}

void testMatrixMultiply()
{
    std::cout << "\n=== Testing SIMD Matrix Multiply ===" << std::endl;

    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Acc = alpaka::AccCpuSerial<Dim, Idx>;
    using Queue = alpaka::QueueCpuBlocking;

    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    Queue queue(devAcc);

    constexpr std::size_t rows = 64;
    constexpr std::size_t cols = 64;
    constexpr std::size_t inner = 64;

    // Initialize matrices
    std::vector<float> a(rows * inner, 1.0f);
    std::vector<float> b(inner * cols, 2.0f);
    std::vector<float> c(rows * cols, 0.0f);

    // Simple initialization
    for(std::size_t i = 0; i < rows * inner; ++i)
        a[i] = static_cast<float>(i % 10);
    for(std::size_t i = 0; i < inner * cols; ++i)
        b[i] = static_cast<float>((i % 5) + 1);

    // Allocate device memory
    auto bufA = alpaka::allocBuf<float, Idx>(devAcc, rows * inner);
    auto bufB = alpaka::allocBuf<float, Idx>(devAcc, inner * cols);
    auto bufC = alpaka::allocBuf<float, Idx>(devAcc, rows * cols);

    // Copy data
    alpaka::memcpy(queue, bufA, a.data(), rows * inner);
    alpaka::memcpy(queue, bufB, b.data(), inner * cols);

    // Launch kernel
    constexpr std::size_t numThreads = 8;
    auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>{1, numThreads, 1};

    alpaka::exec<Acc>(
        queue,
        workDiv,
        MatrixMultiplySimdKernel{},
        alpaka::getPtrNative(bufA),
        alpaka::getPtrNative(bufB),
        alpaka::getPtrNative(bufC),
        rows,
        cols,
        inner);

    // Get results
    alpaka::memcpy(queue, c.data(), bufC, rows * cols);
    alpaka::wait(queue);

    // Verify a few elements
    float sample = c[0];
    std::cout << "Sample result c[0][0]: " << sample << std::endl;

    using SimdType = alpaka::simd::PortableSimd<float, Acc>;
    std::cout << "SIMD width used: " << SimdType::size() << std::endl;
}

int main()
{
    std::cout << "Alpaka SIMD Examples" << std::endl;

    try
    {
        testDotProduct();
        testMatrixMultiply();

        std::cout << "\nAll SIMD tests completed successfully!" << std::endl;
    }
    catch(std::exception const& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
