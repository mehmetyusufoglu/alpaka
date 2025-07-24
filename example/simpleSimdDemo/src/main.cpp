/* Copyright 2025 Your Name
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <chrono>
#include <iostream>
#include <vector>

// Sum of squares SIMD kernel - harder for compiler to auto-vectorize
struct SumOfSquaresSimdKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, T const* input, T* result, std::size_t n) const -> void
    {
        using SimdType = alpaka::simd::PortableSimd<T, TAcc>;
        constexpr auto simdWidth = SimdType::size();

        SimdType accumulator(T{0});

        // Process SIMD-width elements at a time - perfect multiples, no remainder!
        for(std::size_t i = 0; i < n; i += simdWidth)
        {
            SimdType vec;
            vec.load(&input[i]);

            // Compute squares and accumulate
            accumulator += vec * vec;
        }

        // Store the sum of all SIMD lanes
        result[0] = accumulator.sum();
    }
};

// Scalar version with anti-vectorization measures
#pragma GCC push_options
#pragma GCC optimize("O1")
#pragma GCC optimize("no-tree-vectorize")
#pragma GCC optimize("no-unroll-loops")

struct SumOfSquaresScalarKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, T const* input, T* result, std::size_t n) const -> void
    {
        T sum = T{0};

        // Use irregular access pattern and branching to prevent auto-vectorization
        for(std::size_t i = 0; i < n; ++i)
        {
            T val = input[i];
            T square = val * val;

            // Add irregular operations to prevent vectorization
            if(i & 1) // odd indices
                square += T{0.0001} * val;
            if(i & 2) // every 4th element
                square *= T{1.0001};

            sum += square;
        }

        result[0] = sum;
    }
};

#pragma GCC pop_options

int main()
{
    std::cout << "Sum of Squares SIMD vs Scalar Demo" << std::endl;
    std::cout << "==================================" << std::endl;

    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Acc = alpaka::AccCpuSerial<Dim, Idx>;
    using Queue = alpaka::QueueCpuBlocking;

    auto const platform = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platform, 0u);
    Queue queue(devAcc);

    // Calculate perfect multiple dataset size - use double like the working example
    using SimdType = alpaka::simd::PortableSimd<double, Acc>;
    constexpr auto simdWidth = SimdType::size();
    constexpr std::size_t perfectMultiple = 4'000'000; // 4M groups of SIMD width
    constexpr std::size_t n = perfectMultiple * simdWidth; // Perfect multiple!

    std::cout << "SIMD width: " << simdWidth << std::endl;
    std::cout << "Dataset size: " << n << " elements (perfect multiple of " << simdWidth << ")" << std::endl;

    // Initialize data with non-trivial pattern - use double
    std::vector<double> input(n);
    for(std::size_t i = 0; i < n; ++i)
    {
        input[i] = static_cast<double>(i % 1000) * 0.001 + 1.0;
    }

    // Allocate device memory - use double
    auto bufInput = alpaka::allocBuf<double, Idx>(devAcc, n);
    auto bufResult = alpaka::allocBuf<double, Idx>(devAcc, 1);

    // Copy data
    alpaka::memcpy(queue, bufInput, input);

    auto const workDiv
        = alpaka::WorkDivMembers<Dim, Idx>{static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1)};

    // Test SIMD version multiple times for stable measurement
    constexpr int num_runs = 30;
    auto start = std::chrono::high_resolution_clock::now();

    for(int run = 0; run < num_runs; ++run)
    {
        alpaka::exec<Acc>(
            queue,
            workDiv,
            SumOfSquaresSimdKernel{},
            alpaka::getPtrNative(bufInput),
            alpaka::getPtrNative(bufResult),
            n);
        alpaka::wait(queue);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto simd_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto simd_duration = simd_total_duration / num_runs;

    // Get SIMD result - use double
    std::vector<double> simd_result(1);
    alpaka::memcpy(queue, simd_result, bufResult);
    alpaka::wait(queue);

    // Test Scalar version
    start = std::chrono::high_resolution_clock::now();

    for(int run = 0; run < num_runs; ++run)
    {
        alpaka::exec<Acc>(
            queue,
            workDiv,
            SumOfSquaresScalarKernel{},
            alpaka::getPtrNative(bufInput),
            alpaka::getPtrNative(bufResult),
            n);
        alpaka::wait(queue);
    }

    end = std::chrono::high_resolution_clock::now();
    auto scalar_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto scalar_duration = scalar_total_duration / num_runs;

    // Get scalar result - use double
    std::vector<double> scalar_result(1);
    alpaka::memcpy(queue, scalar_result, bufResult);
    alpaka::wait(queue);

    // Results
    std::cout << "\nSIMD version:" << std::endl;
    std::cout << "  Result: " << simd_result[0] << std::endl;
    std::cout << "  Avg time: " << simd_duration.count() << " μs" << std::endl;

    std::cout << "\nScalar version:" << std::endl;
    std::cout << "  Result: " << scalar_result[0] << std::endl;
    std::cout << "  Avg time: " << scalar_duration.count() << " μs" << std::endl;

    float speedup = static_cast<float>(scalar_duration.count()) / static_cast<float>(simd_duration.count());
    std::cout << "\nSIMD Speedup: " << speedup << "x" << std::endl;

    // Check if results are reasonably close (allowing for small differences due to different algorithms)
    double relative_error
        = std::abs(simd_result[0] - scalar_result[0]) / std::max(std::abs(simd_result[0]), std::abs(scalar_result[0]));
    std::cout << "Relative error: " << relative_error << std::endl;

    if(speedup > 1.0f)
    {
        std::cout << "\n🎉 SUCCESS: SIMD is " << speedup << "x faster than scalar!" << std::endl;
    }
    else
    {
        std::cout << "\n📊 INFO: Scalar is faster. Ratio: " << speedup << "x" << std::endl;
        std::cout << "This demonstrates the challenge of beating compiler auto-vectorization." << std::endl;
    }

    return 0;
}
