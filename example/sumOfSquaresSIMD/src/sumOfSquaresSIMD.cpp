#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <chrono>
#include <iostream>
#include <random>

//! IMPORTANT: SIMD Performance Optimization Requirements
//! For optimal SIMD performance, compile with -O3 optimization and -march=native to enable auto-vectorization and
//! intrinsics inlining. Without these flags, SIMD operations may fall back to scalar execution, resulting in poor
//! performance. For cross-platform builds or specific targeting, use explicit flags like -mavx2, but -march=native
//! automatically enables all supported instructions on the target CPU.


// SIMD Kernel - One SIMD operation per thread
template<typename T>
class SumOfSquaresSIMDKernel1Thread1SIMD
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename Acc>
    ALPAKA_FN_ACC auto operator()(Acc const& acc, T const* input, T* result, size_t dataSize) const
    {
        size_t globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

        // Use alpaka SIMD
        using SimdType = alpaka::simd::PortableSimd<T, Acc>;
        constexpr size_t simdWidth = SimdType::size();

        // Each thread processes one SIMD-width chunk
        if(globalIdx * simdWidth < dataSize)
        {
            SimdType simd_data;
            simd_data.load(&input[globalIdx * simdWidth]);

            // Square the values and accumulate
            auto simd_squared = simd_data * simd_data;

            // Directly accumulate the result using atomicAdd
            alpaka::atomicAdd(acc, result, simd_squared.sum(), alpaka::hierarchy::Blocks{});
        }
    }
};

// Non-SIMD Kernel
template<typename T>
class SumOfSquaresNonSIMDKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename Acc>
    ALPAKA_FN_ACC auto operator()(Acc const& acc, T const* input, T* result, size_t dataSize) const
    {
        size_t globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        size_t globalSize = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        T localSum = 0.0;

        // Scalar computation for the thread
        for(size_t i = globalIdx; i < dataSize; i += globalSize)
        {
            T value = input[i];
            localSum += value * value; // Compute square and accumulate
        }

        // Directly accumulate the result using atomicAdd
        alpaka::atomicAdd(acc, result, localSum, alpaka::hierarchy::Blocks{});
    }
};

// Template function to test with different data types
template<typename T, typename Acc, typename Queue, typename DevAcc, typename DevHost>
void testDataType(Queue& queue, DevAcc const& devAcc, DevHost const& devHost, std::string const& typeName)
{
    using Dim = alpaka::DimInt<1u>;
    using Idx = unsigned;

    Idx const numElements(32 * 1024 * 1024); // 32M elements
    Idx const elementsPerThread(1);
    alpaka::Vec<Dim, Idx> const extent(numElements);

    // Allocate buffers
    using BufHost = alpaka::Buf<DevHost, T, Dim, Idx>;
    BufHost bufHostA(alpaka::allocBuf<T, Idx>(devHost, extent));

    // Initialize data
    std::random_device rd;
    std::default_random_engine eng{rd()};
    std::uniform_real_distribution<T> dist(1.0, 42.0);
    T referenceSum = 0.0;
    for(Idx i = 0; i < numElements; ++i)
    {
        bufHostA[i] = dist(eng);
        referenceSum += bufHostA[i] * bufHostA[i];
    }

    using BufAcc = alpaka::Buf<DevAcc, T, Dim, Idx>;
    BufAcc bufAccA(alpaka::allocBuf<T, Idx>(devAcc, extent));

    // Allocate result buffer on device (single element)
    using BufResultAcc = alpaka::Buf<DevAcc, T, Dim, Idx>;
    using BufResultHost = alpaka::Buf<DevHost, T, Dim, Idx>;
    BufResultAcc bufResultAcc(alpaka::allocBuf<T, Idx>(devAcc, alpaka::Vec<Dim, Idx>{1}));
    BufResultHost bufResultHost(alpaka::allocBuf<T, Idx>(devHost, alpaka::Vec<Dim, Idx>{1}));
    T* resultPtr = alpaka::getPtrNative(bufResultAcc);

    alpaka::memcpy(queue, bufAccA, bufHostA);

    using SimdType = alpaka::simd::PortableSimd<T, Acc>;
    constexpr size_t simdWidth = SimdType::size();

    std::cout << "\n=== Testing with " << typeName << " ===" << std::endl;
    std::cout << "simdWidth for type " << typeName << " is " << simdWidth << std::endl;
    std::cout << "numElements: " << numElements << std::endl;

    // Variables to store timing for comparison
    double nonSimdTime = 0.0;
    double simdTime = 0.0;

    // Measure Non-SIMD Kernel
    {
        // Initialize result to zero on device
        T zero = 0.0;
        bufResultHost[0] = zero;
        alpaka::memcpy(queue, bufResultAcc, bufResultHost);

        SumOfSquaresNonSIMDKernel<T> nonSimdKernel;
        alpaka::KernelCfg<Acc> const kernelCfg = {extent, elementsPerThread};

        auto const workDiv = alpaka::getValidWorkDiv(
            kernelCfg,
            devAcc,
            nonSimdKernel,
            alpaka::getPtrNative(bufAccA),
            resultPtr,
            numElements);

        auto const taskKernel = alpaka::createTaskKernel<Acc>(
            workDiv,
            nonSimdKernel,
            alpaka::getPtrNative(bufAccA),
            resultPtr,
            numElements);

        alpaka::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        alpaka::enqueue(queue, taskKernel);
        alpaka::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();

        // Copy result back to host
        alpaka::memcpy(queue, bufResultHost, bufResultAcc);
        alpaka::wait(queue);
        T result = *alpaka::getPtrNative(bufResultHost);

        std::cout << " " << std::endl;
        std::cout << "Non-SIMD Kernel Execution Time: " << std::chrono::duration<double>(endT - beginT).count()
                  << "s\n";
        std::cout << "Non-SIMD Kernel Result: " << result << "\n";

        // Store non-SIMD time for comparison
        nonSimdTime = std::chrono::duration<double>(endT - beginT).count();
    }

    // Measure SIMD Kernel (1Thread1SIMD)
    {
        // Initialize result to zero on device
        T zero = 0.0;
        *alpaka::getPtrNative(bufResultHost) = zero;
        alpaka::memcpy(queue, bufResultAcc, bufResultHost);

        SumOfSquaresSIMDKernel1Thread1SIMD<T> simdKernel;

        Idx const simdAdjustedExtent = numElements / simdWidth; // Adjust extent for SIMD processing
        alpaka::Vec<Dim, Idx> const extent(simdAdjustedExtent);

        std::cout << " " << std::endl;
        std::cout << "simdAdjustedExtent = numElements / simdWidth is equal to " << simdAdjustedExtent << std::endl;

        alpaka::WorkDivMembers<Dim, Idx> workDivManual{
            simdAdjustedExtent,
            alpaka::Vec<Dim, Idx>::all(1),
            alpaka::Vec<Dim, Idx>::all(1)};
        auto const taskKernel = alpaka::createTaskKernel<Acc>(
            workDivManual,
            simdKernel,
            alpaka::getPtrNative(bufAccA),
            resultPtr,
            numElements);

        alpaka::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        alpaka::enqueue(queue, taskKernel);
        alpaka::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();

        // Copy result back to host
        alpaka::memcpy(queue, bufResultHost, bufResultAcc);
        alpaka::wait(queue);
        T result = *alpaka::getPtrNative(bufResultHost);

        std::cout << "SIMD Kernel Execution Time: " << std::chrono::duration<double>(endT - beginT).count() << "s\n";
        std::cout << "SIMD Kernel Result: " << result << "\n";

        // Store SIMD time for comparison
        simdTime = std::chrono::duration<double>(endT - beginT).count();
    }

    std::cout << " " << std::endl;
    std::cout << "Reference Sum of Squares: " << referenceSum << "\n";

    // Print SIMD improvement ratio
    std::cout << "\n=== SIMD Performance Analysis ===" << std::endl;
    std::cout << "Non-SIMD Time: " << nonSimdTime << "s" << std::endl;
    std::cout << "SIMD Time: " << simdTime << "s" << std::endl;

    double improvementRatio = nonSimdTime / simdTime;
    if(improvementRatio > 1.0)
    {
        std::cout << "SIMD is " << improvementRatio << "x FASTER than Non-SIMD" << std::endl;
    }
    else
    {
        std::cout << "SIMD is " << (1.0 / improvementRatio) << "x SLOWER than Non-SIMD" << std::endl;
    }
    std::cout << "==================================" << std::endl;
}

// Example function to compare SIMD and non-SIMD kernels
template<alpaka::concepts::Tag TAccTag>
auto example(TAccTag const&) -> int
{
    using Dim = alpaka::DimInt<1u>;
    using Idx = unsigned;

    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
    using DevAcc = alpaka::Dev<Acc>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;

    auto const platform = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platform, 0);
    QueueAcc queue(devAcc);

    using DevHost = alpaka::DevCpu;
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    // Test with double
    testDataType<double, Acc, QueueAcc, DevAcc, DevHost>(queue, devAcc, devHost, "double");

    // Test with float
    testDataType<float, Acc, QueueAcc, DevAcc, DevHost>(queue, devAcc, devHost, "float");

    return EXIT_SUCCESS;
}

auto main() -> int
{
    std::cout << "Check enabled accelerator tags:" << std::endl;
    alpaka::printTagNames<alpaka::EnabledAccTags>();
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}
