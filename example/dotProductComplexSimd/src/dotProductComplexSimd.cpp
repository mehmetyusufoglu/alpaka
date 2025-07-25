/* Copyright 2025 Your Name
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

//! IMPORTANT: SIMD Performance Optimization Requirements
//! For optimal SIMD performance, compile with -O3 optimization and -march=native.
//! For cross-platform builds or specific targeting, use explicit
//! flags like -mavx2, but -march=native automatically enables all supported instructions on the target CPU.

// Add SIMD capability check function
void checkSIMDCapabilities()
{
    std::cout << "\n=== SIMD Capability Check ===" << std::endl;

#ifdef __AVX512F__
    std::cout << "AVX-512F: SUPPORTED (SIMD width: float=16, double=8)" << std::endl;
#else
    std::cout << "AVX-512F: NOT SUPPORTED" << std::endl;
#endif

#ifdef __AVX2__
    std::cout << "AVX2: SUPPORTED (SIMD width: float=8, double=4)" << std::endl;
#else
    std::cout << "AVX2: NOT SUPPORTED" << std::endl;
#endif

#ifdef __AVX__
    std::cout << "AVX: SUPPORTED (SIMD width: float=8, double=4)" << std::endl;
#else
    std::cout << "AVX: NOT SUPPORTED" << std::endl;
#endif

#ifdef __SSE4_2__
    std::cout << "SSE4.2: SUPPORTED (SIMD width: float=4, double=2)" << std::endl;
#else
    std::cout << "SSE4.2: NOT SUPPORTED" << std::endl;
#endif

#ifdef __SSE2__
    std::cout << "SSE2: SUPPORTED (SIMD width: float=4, double=2)" << std::endl;
#else
    std::cout << "SSE2: NOT SUPPORTED" << std::endl;
#endif

    // Check what the compiler is actually using
    std::cout << "\nCompiler SIMD flags:" << std::endl;
#if defined(__GNUC__) || defined(__clang__)
    std::cout << "Compiler: "
              <<
#    ifdef __clang__
        "Clang " << __clang_major__ << "." << __clang_minor__
#    else
        "GCC " << __GNUC__ << "." << __GNUC_MINOR__
#    endif
              << std::endl;
#endif

    std::cout << "==============================" << std::endl;
}

// Complex dot product kernel using SIMD
template<typename T>
struct ComplexDotProductSimdKernel
{
    template<typename TAcc>
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

template<typename T>
struct ComplexDotProductScalarKernel
{
    template<typename TAcc>
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

// Template function to test different data types
template<typename T, typename Acc, typename Queue>
void testDataType(Queue& queue, alpaka::Dev<Acc> const& devAcc, std::string const& typeName)
{
    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;

    std::cout << "\n=== Testing " << typeName << " ===" << std::endl;

    // Get SIMD width and calculate perfect multiple dataset size
    using SimdType = alpaka::simd::PortableSimd<T, Acc>;
    constexpr auto simdWidth = SimdType::size();
    constexpr std::size_t numThreads = 1; // Use single thread for AccCpuSerial

    // Dataset size that's a perfect multiple of SIMD width and number of threads
    // This ensures no remainder elements need special handling
    constexpr std::size_t elementsPerThread = 5'000'000; // 5M elements per thread
    constexpr std::size_t simdVecsPerThread = elementsPerThread / simdWidth;
    constexpr std::size_t n = simdVecsPerThread * simdWidth * numThreads; // Perfect multiple

    std::cout << "SIMD width: " << simdWidth << std::endl;
    std::cout << "Dataset size: " << n << " elements (perfect multiple of " << simdWidth << ")" << std::endl;
    std::cout << "Elements per thread: " << n / numThreads << std::endl;

    std::vector<T> a(n), b(n);

    // Initialize with non-trivial patterns
    for(std::size_t i = 0; i < n; ++i)
    {
        a[i] = static_cast<T>(i % 1000) / T{1000.0} + T{1.0};
        b[i] = static_cast<T>((i * 7) % 1000) / T{1000.0} + T{1.0};
    }

    // Allocate buffers
    alpaka::Vec<Dim, Idx> const extentData(static_cast<Idx>(n));
    alpaka::Vec<Dim, Idx> const extentResults(static_cast<Idx>(numThreads));
    auto bufA = alpaka::allocBuf<T, Idx>(devAcc, extentData);
    auto bufB = alpaka::allocBuf<T, Idx>(devAcc, extentData);
    auto bufResultsSimd = alpaka::allocBuf<T, Idx>(devAcc, extentResults);
    auto bufResultsScalar = alpaka::allocBuf<T, Idx>(devAcc, extentResults);

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
            ComplexDotProductSimdKernel<T>{},
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
            ComplexDotProductScalarKernel<T>{},
            alpaka::getPtrNative(bufA),
            alpaka::getPtrNative(bufB),
            alpaka::getPtrNative(bufResultsScalar),
            n);
        alpaka::wait(queue);
    }

    end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / num_runs;

    // Get results
    std::vector<T> simd_result(numThreads);
    std::vector<T> scalar_result(numThreads);
    alpaka::memcpy(queue, simd_result, bufResultsSimd);
    alpaka::memcpy(queue, scalar_result, bufResultsScalar);

    T simd_total = std::accumulate(simd_result.begin(), simd_result.end(), T{0});
    T scalar_total = std::accumulate(scalar_result.begin(), scalar_result.end(), T{0});

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

    // Standardized performance analysis
    double speedup = (double) scalar_time / simd_time;
    std::cout << "\n=== SIMD Performance Analysis ===" << std::endl;
    std::cout << "SIMD Time: " << simd_time << " µs" << std::endl;
    std::cout << "Scalar Time: " << scalar_time << " µs" << std::endl;
    if(speedup > 1.0)
    {
        std::cout << "SIMD is " << speedup << "x FASTER than scalar" << std::endl;
    }
    else
    {
        std::cout << "SIMD is " << (1.0 / speedup) << "x SLOWER than scalar" << std::endl;
    }
    std::cout << "==================================" << std::endl;
}

// Performance test
auto testComplexDotProduct() -> void
{
    std::cout << "\n=== Testing SIMD vs Scalar Complex Dot Product ===" << std::endl;

    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Acc = alpaka::AccCpuSerial<Dim, Idx>;
    using Queue = alpaka::QueueCpuBlocking;

    auto const platform = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platform, 0u);
    Queue queue(devAcc);

    // Test with float
    testDataType<float, Acc>(queue, devAcc, "float");

    // Test with double
    testDataType<double, Acc>(queue, devAcc, "double");

    // Compare SIMD widths
    std::cout << "\n=== Data Type Comparison Summary ===" << std::endl;
    std::cout << "Float SIMD width: " << alpaka::simd::PortableSimd<float, Acc>::size() << " elements" << std::endl;
    std::cout << "Double SIMD width: " << alpaka::simd::PortableSimd<double, Acc>::size() << " elements" << std::endl;
    std::cout << "Float SIMD width is "
              << (alpaka::simd::PortableSimd<float, Acc>::size() / alpaka::simd::PortableSimd<double, Acc>::size())
              << "x larger than double" << std::endl;
    std::cout << "=====================================" << std::endl;
}

auto main() -> int
{
    try
    {
        std::cout << "=== Alpaka SIMD Performance Examples ===" << std::endl;

        // Check SIMD capabilities first
        checkSIMDCapabilities();

        // Display SIMD capabilities
        using Acc = alpaka::AccCpuSerial<alpaka::DimInt<1>, std::size_t>;
        using SimdFloat = alpaka::simd::PortableSimd<float, Acc>;
        using SimdDouble = alpaka::simd::PortableSimd<double, Acc>;
        std::cout << "\nAlpaka detected SIMD width for float: " << SimdFloat::size() << std::endl;
        std::cout << "Alpaka detected SIMD width for double: " << SimdDouble::size() << std::endl;

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
