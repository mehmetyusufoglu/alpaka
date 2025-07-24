/* Copyright 2025 Your Name
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <iostream>
#include <vector>

// Example kernel using SIMD operations
struct VectorAddSimdKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, T const* a, T const* b, T* c, std::size_t n) const -> void
    {
        // Use SIMD for this accelerator type
        using SimdType = alpaka::simd::PortableSimd<T, TAcc>;
        constexpr auto simdWidth = SimdType::size();

        // For single-threaded CPU, process all data in one go
        std::size_t i = 0;

        // Process SIMD-width elements at a time
        for(; i + simdWidth <= n; i += simdWidth)
        {
            // Load SIMD vectors
            SimdType va, vb;
            va.load(&a[i]);
            vb.load(&b[i]);

            // Perform SIMD addition
            auto vc = va + vb;

            // Store result
            vc.store(&c[i]);
        }

        // Handle remaining elements scalar
        for(; i < n; ++i)
        {
            c[i] = a[i] + b[i];
        }
    }
};

int main()
{
    // Define the accelerator type (CPU for SIMD demonstration)
    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Acc = alpaka::AccCpuSerial<Dim, Idx>;

    // Get the platform and device
    auto const platform = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platform, 0u);

    // Create a queue
    using Queue = alpaka::QueueCpuBlocking;
    Queue queue(devAcc);

    // Problem size
    constexpr std::size_t n = 1024;
    constexpr std::size_t numElements = n;

    // Allocate host memory
    std::vector<float> a(numElements, 1.0f);
    std::vector<float> b(numElements, 2.0f);
    std::vector<float> c(numElements, 0.0f);

    // Initialize input data
    for(std::size_t i = 0; i < numElements; ++i)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    auto bufA = alpaka::allocBuf<float, Idx>(devAcc, numElements);
    auto bufB = alpaka::allocBuf<float, Idx>(devAcc, numElements);
    auto bufC = alpaka::allocBuf<float, Idx>(devAcc, numElements);

    // Copy data to device
    alpaka::memcpy(queue, bufA, a);
    alpaka::memcpy(queue, bufB, b);

    // Set up kernel execution parameters (single work item for CPU serial)
    auto const workDiv
        = alpaka::WorkDivMembers<Dim, Idx>{static_cast<Idx>(1), static_cast<Idx>(1), static_cast<Idx>(1)};

    // Get SIMD width for information
    using SimdType = alpaka::simd::PortableSimd<float, Acc>;
    constexpr auto simdWidth = SimdType::size();

    std::cout << "SIMD width for float on this accelerator: " << simdWidth << std::endl;
    std::cout << "Processing " << numElements << " elements\n";

    // Launch kernel
    alpaka::exec<Acc>(
        queue,
        workDiv,
        VectorAddSimdKernel{},
        alpaka::getPtrNative(bufA),
        alpaka::getPtrNative(bufB),
        alpaka::getPtrNative(bufC),
        numElements);

    // Copy result back to host
    alpaka::memcpy(queue, c, bufC);
    alpaka::wait(queue);

    // Verify results
    bool success = true;
    for(std::size_t i = 0; i < numElements; ++i)
    {
        float expected = a[i] + b[i];
        if(std::abs(c[i] - expected) > 1e-5f)
        {
            std::cout << "Error at index " << i << ": expected " << expected << ", got " << c[i] << std::endl;
            success = false;
            break;
        }
    }

    if(success)
    {
        std::cout << "SIMD vector addition completed successfully!" << std::endl;
    }
    else
    {
        std::cout << "SIMD vector addition failed!" << std::endl;
        return 1;
    }

    return 0;
}
