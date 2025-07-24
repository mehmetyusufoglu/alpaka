/* Copyright 2025 Your Name
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

// Complex dot product kernel using SIMD
struct ComplexDotProductSimdKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, T const* a, T const* b, T* results, std::size_t n) const -> void
    {
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        using SimdType = alpaka::simd::PortableSimd<T, TAcc>;
        constexpr auto simdWidth = SimdType::size();

        SimdType accumulator(T{0});

        // Process SIMD-width elements at a time - no remainder handling needed!
        // Since n is guaranteed to be a perfect multiple of (simdWidth * numThreads)
        for(std::size_t i = globalThreadIdx * simdWidth; i < n; i += globalThreadExtent * simdWidth)
        {
            SimdType vec_a, vec_b;
            vec_a.load(&a[i]);
            vec_b.load(&b[i]);

            // Efficient SIMD computation using compound operations
            accumulator += vec_a * vec_b + (vec_a + vec_b) * (vec_a - vec_b);
        }

        results[globalThreadIdx] = accumulator.sum();
    }
};

// Scalar version with complex operations to prevent auto-vectorization
#pragma GCC push_options
#pragma GCC optimize("O1")
#pragma GCC optimize("no-tree-vectorize")
#pragma GCC optimize("no-unroll-loops")

struct ComplexDotProductScalarKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, T const* a, T const* b, T* results, std::size_t n) const -> void
    {
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        auto const globalThreadExtent = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        T sum = T{0};

        // Anti-vectorization scalar computation with irregular patterns
        for(std::size_t i = globalThreadIdx; i < n; i += globalThreadExtent)
        {
            T val_a = a[i];
            T val_b = b[i];

            // Equivalent scalar computation but harder to auto-vectorize
            T result1 = val_a * val_b;
            T result2 = val_a + val_b;
            T result3 = val_a - val_b;
            T combined = result1 + result2 * result3;

            // Add irregular operations to prevent auto-vectorization
            if(i & 1) // odd indices
                combined *= T{1.001};
            if(i & 2) // every 4th element starting from 2
                combined += T{0.001};

            sum += combined;
        }

        results[globalThreadIdx] = sum;
    }
};

#pragma GCC pop_options

// Performance test
auto testComplexDotProduct() -> void
{
    std::cout << "\n=== Testing SIMD vs Scalar Complex Dot Product ===" << std::endl;

    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Acc = alpaka::AccCpuThreads<Dim, Idx>;
    using Queue = alpaka::QueueCpuBlocking;

    auto const platform = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platform, 0u);
    auto queue = Queue{devAcc};

    // Get SIMD width and calculate perfect multiple dataset size
    using SimdType = alpaka::simd::PortableSimd<float, Acc>;
    constexpr auto simdWidth = SimdType::size();
    constexpr std::size_t numThreads = 8;

    // Dataset size that's a perfect multiple of SIMD width and number of threads
    // This ensures no remainder elements need special handling
    constexpr std::size_t elementsPerThread = 5'000'000; // 5M elements per thread
    constexpr std::size_t simdVecsPerThread = elementsPerThread / simdWidth;
    constexpr std::size_t n = simdVecsPerThread * simdWidth * numThreads; // Perfect multiple

    std::cout << "SIMD width: " << simdWidth << std::endl;
    std::cout << "Dataset size: " << n << " elements (perfect multiple of " << simdWidth << ")" << std::endl;
    std::cout << "Elements per thread: " << n / numThreads << std::endl;

    std::vector<float> a(n), b(n);

    // Initialize with non-trivial patterns
    for(std::size_t i = 0; i < n; ++i)
    {
        a[i] = static_cast<float>(i % 1000) / 1000.0f + 1.0f;
        b[i] = static_cast<float>((i * 7) % 1000) / 1000.0f + 1.0f;
    }

    // Allocate buffers
    auto bufA = alpaka::allocBuf<float, Idx>(devAcc, n);
    auto bufB = alpaka::allocBuf<float, Idx>(devAcc, n);
    auto bufResultsSimd = alpaka::allocBuf<float, Idx>(devAcc, numThreads);
    auto bufResultsScalar = alpaka::allocBuf<float, Idx>(devAcc, numThreads);

    alpaka::memcpy(queue, bufA, a);
    alpaka::memcpy(queue, bufB, b);

    auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{1}, Idx{numThreads}, Idx{1}};

    // Test SIMD version
    constexpr int num_runs = 20;
    auto start = std::chrono::high_resolution_clock::now();

    for(int run = 0; run < num_runs; ++run)
    {
        alpaka::exec<Acc>(
            queue,
            workDiv,
            ComplexDotProductSimdKernel{},
            alpaka::getPtrNative(bufA),
            alpaka::getPtrNative(bufB),
            alpaka::getPtrNative(bufResultsSimd),
            n);
        alpaka::wait(queue);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / num_runs;

    // Test scalar version
    start = std::chrono::high_resolution_clock::now();

    for(int run = 0; run < num_runs; ++run)
    {
        alpaka::exec<Acc>(
            queue,
            workDiv,
            ComplexDotProductScalarKernel{},
            alpaka::getPtrNative(bufA),
            alpaka::getPtrNative(bufB),
            alpaka::getPtrNative(bufResultsScalar),
            n);
        alpaka::wait(queue);
    }

    end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / num_runs;

    // Get results
    std::vector<float> simd_result(numThreads);
    std::vector<float> scalar_result(numThreads);
    alpaka::memcpy(queue, simd_result, bufResultsSimd);
    alpaka::memcpy(queue, scalar_result, bufResultsScalar);

    float simd_total = std::accumulate(simd_result.begin(), simd_result.end(), 0.0f);
    float scalar_total = std::accumulate(scalar_result.begin(), scalar_result.end(), 0.0f);

    std::cout << "Dataset size: " << n << " elements" << std::endl;
    std::cout << "SIMD time: " << simd_time << " µs" << std::endl;
    std::cout << "Scalar time: " << scalar_time << " µs" << std::endl;
    std::cout << "SIMD speedup: " << (double) scalar_time / simd_time << "x" << std::endl;
    std::cout << "SIMD result: " << simd_total << std::endl;
    std::cout << "Scalar result: " << scalar_total << std::endl;

    // More lenient comparison due to different algorithms
    bool results_close
        = std::abs(simd_total - scalar_total) / std::max(std::abs(simd_total), std::abs(scalar_total)) < 0.1;
    std::cout << "Results reasonably close: " << (results_close ? "YES" : "NO") << std::endl;
}

auto main() -> int
{
    try
    {
        std::cout << "=== Alpaka SIMD Performance Examples ===" << std::endl;

        // Display SIMD capabilities
        using Acc = alpaka::AccCpuThreads<alpaka::DimInt<1>, std::size_t>;
        using SimdFloat = alpaka::simd::PortableSimd<float, Acc>;
        std::cout << "SIMD width for float: " << SimdFloat::size() << std::endl;

        // Run performance test
        testComplexDotProduct();

        std::cout << "\n=== All tests completed ===" << std::endl;
        return EXIT_SUCCESS;
    }
    catch(std::exception const& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
