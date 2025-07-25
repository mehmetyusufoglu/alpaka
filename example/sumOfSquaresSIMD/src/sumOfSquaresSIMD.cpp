#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <chrono>
#include <iostream>
#include <random>

// SIMD Kernel assuming gridsize is smaller than dataSize/simd_register_size (ie 4 or 8)
class SumOfSquaresSIMDKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename Acc>
    ALPAKA_FN_ACC auto operator()(Acc const& acc, double const* input, double* result, size_t dataSize) const
    {
        size_t globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        size_t gridSize = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        // Use alpaka SIMD
        using SimdType = alpaka::simd::PortableSimd<double, Acc>;
        constexpr size_t simdWidth = SimdType::size();
        double localSum = 0.0;

        // SIMD computation for the thread
        for(size_t i = globalIdx * simdWidth; i < dataSize; i += gridSize * simdWidth)
        {
            // Bounds check for SIMD load
            if(i + simdWidth <= dataSize)
            {
                SimdType simd_data;
                simd_data.load(&input[i]);

                // Square the values and accumulate
                auto simd_squared = simd_data * simd_data;
                localSum += simd_squared.sum();
            }
            else
            {
                // Handle remaining elements individually (scalar fallback)
                for(size_t j = i; j < dataSize && j < i + simdWidth; ++j)
                {
                    double val = input[j];
                    localSum += val * val;
                }
                break;
            }
        }

        // Directly accumulate the result using atomicAdd
        alpaka::atomicAdd(acc, result, localSum, alpaka::hierarchy::Blocks{});
    }
};

// SIMD Kernel - One SIMD operation per thread
class SumOfSquaresSIMDKernel1Thread1SIMD
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename Acc>
    ALPAKA_FN_ACC auto operator()(Acc const& acc, double const* input, double* result, size_t dataSize) const
    {
        size_t globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

        // Use alpaka SIMD
        using SimdType = alpaka::simd::PortableSimd<double, Acc>;
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
class SumOfSquaresNonSIMDKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename Acc>
    ALPAKA_FN_ACC auto operator()(Acc const& acc, double const* input, double* result, size_t dataSize) const
    {
        size_t globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        size_t globalSize = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        double localSum = 0.0;

        // Scalar computation for the thread
        for(size_t i = globalIdx; i < dataSize; i += globalSize)
        {
            double value = input[i];
            localSum += value * value; // Compute square and accumulate
        }

        // Directly accumulate the result using atomicAdd
        alpaka::atomicAdd(acc, result, localSum, alpaka::hierarchy::Blocks{});
    }
};

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

    Idx const numElements(32 * 1024 * 1024); // Increased to 32M for more demanding test
    Idx const elementsPerThread(1);
    alpaka::Vec<Dim, Idx> const extent(numElements);

    using Data = double;
    using DevHost = alpaka::DevCpu;
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    using BufHost = alpaka::Buf<DevHost, Data, Dim, Idx>;
    BufHost bufHostA(alpaka::allocBuf<Data, Idx>(devHost, extent));

    std::random_device rd;
    std::default_random_engine eng{rd()};
    std::uniform_real_distribution<Data> dist(1.0, 42.0);
    double referenceSum = 0.0;
    for(Idx i = 0; i < numElements; ++i)
    {
        bufHostA[i] = dist(eng);
        referenceSum += bufHostA[i] * bufHostA[i];
    }

    using BufAcc = alpaka::Buf<DevAcc, Data, Dim, Idx>;
    BufAcc bufAccA(alpaka::allocBuf<Data, Idx>(devAcc, extent));

    // Allocate result buffer on device (single element)
    using Dim = alpaka::DimInt<1u>; // already defined in your example
    using BufResultAcc = alpaka::Buf<DevAcc, Data, Dim, Idx>;
    using BufResultHost = alpaka::Buf<DevHost, Data, Dim, Idx>;
    BufResultAcc bufResultAcc(alpaka::allocBuf<Data, Idx>(devAcc, alpaka::Vec<Dim, Idx>{1}));
    BufResultHost bufResultHost(alpaka::allocBuf<Data, Idx>(devHost, alpaka::Vec<Dim, Idx>{1}));
    Data* resultPtr = alpaka::getPtrNative(bufResultAcc);

    alpaka::memcpy(queue, bufAccA, bufHostA);

    using SimdType = alpaka::simd::PortableSimd<double, Acc>;

    constexpr size_t simdWidth = SimdType::size();
    std::cout << "simdWidth for type " << typeid(Data).name() << " is " << simdWidth << std::endl;
    std::cout << "numElements: " << numElements << std::endl;

    // Variables to store timing for comparison
    double nonSimdTime = 0.0;
    double simd1Thread1SimdTime = 0.0;
    double simdKernelTime = 0.0;

    // Measure Non-SIMD Kernel
    {
        // Initialize result to zero on device
        Data zero = 0.0;
        bufResultHost[0] = zero;
        alpaka::memcpy(queue, bufResultAcc, bufResultHost);

        SumOfSquaresNonSIMDKernel nonSimdKernel;
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
        Data result = *alpaka::getPtrNative(bufResultHost);

        std::cout << " " << std::endl;
        std::cout << "Non-SIMD Kernel Execution Time: " << std::chrono::duration<double>(endT - beginT).count()
                  << "s\n";
        std::cout << "Non-SIMD Kernel Result: " << result << "\n";

        // Store non-SIMD time for comparison
        nonSimdTime = std::chrono::duration<double>(endT - beginT).count();
    }

    // Measure SumOfSquaresSIMDKernel1Thread1SIMD Kernel
    {
        // Initialize result to zero on device
        Data zero = 0.0;
        *alpaka::getPtrNative(bufResultHost) = zero;
        alpaka::memcpy(queue, bufResultAcc, bufResultHost);

        SumOfSquaresSIMDKernel1Thread1SIMD simdKernel;

        Idx const simdAdjustedExtent = numElements / simdWidth; // Adjust extent for SIMD processing
        alpaka::Vec<Dim, Idx> const extent(simdAdjustedExtent);

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
        Data result = *alpaka::getPtrNative(bufResultHost);

        std::cout << "SIMD Kernel Execution Time (Full Data coverd by simd-size*GridSize): "
                  << std::chrono::duration<double>(endT - beginT).count() << "s\n";
        std::cout << "SIMD Kernel Result: " << result << "\n";

        // Store SIMD 1Thread1SIMD time for comparison
        simd1Thread1SimdTime = std::chrono::duration<double>(endT - beginT).count();
    }


    // Measure SumOfSquaresSIMDKernel Kernel
    {
        // Initialize result to zero on device
        Data zero = 0.0;
        *alpaka::getPtrNative(bufResultHost) = zero;
        alpaka::memcpy(queue, bufResultAcc, bufResultHost);

        SumOfSquaresSIMDKernel simdKernel;

        Idx const simdAdjustedExtent
            = numElements / 32; // much smaller grid then being able to cover all data even with simd
        alpaka::Vec<Dim, Idx> const extent(simdAdjustedExtent);
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
        Data result = *alpaka::getPtrNative(bufResultHost);

        std::cout << " " << std::endl;
        std::cout << "newExtent = (numElements / 32)  is equal to " << simdAdjustedExtent << std::endl;
        std::cout << "SIMD Kernel Execution Time (Not full data covered by simdsize*gridsize): "
                  << std::chrono::duration<double>(endT - beginT).count() << "s\n";
        std::cout << "SIMD Kernel Result: " << result << "\n";

        // Store SIMD kernel time for comparison
        simdKernelTime = std::chrono::duration<double>(endT - beginT).count();
    }

    std::cout << "Reference Sum of Squares: " << referenceSum << "\n";

    // Print SIMD improvement ratios
    std::cout << "\n=== SIMD Performance Analysis ===" << std::endl;
    std::cout << "Non-SIMD Time: " << nonSimdTime << "s" << std::endl;
    std::cout << "SIMD 1Thread1SIMD Time: " << simd1Thread1SimdTime << "s" << std::endl;
    std::cout << "SIMD Kernel Time: " << simdKernelTime << "s" << std::endl;

    // Compare SIMD 1Thread1SIMD vs Non-SIMD
    double improvementRatio1 = nonSimdTime / simd1Thread1SimdTime;
    if(improvementRatio1 > 1.0)
    {
        std::cout << "SIMD 1Thread1SIMD is " << improvementRatio1 << "x FASTER than Non-SIMD" << std::endl;
    }
    else
    {
        std::cout << "SIMD 1Thread1SIMD is " << (1.0 / improvementRatio1) << "x SLOWER than Non-SIMD" << std::endl;
    }

    // Compare SIMD Kernel vs Non-SIMD
    double improvementRatio2 = nonSimdTime / simdKernelTime;
    if(improvementRatio2 > 1.0)
    {
        std::cout << "SIMD Kernel is " << improvementRatio2 << "x FASTER than Non-SIMD" << std::endl;
    }
    else
    {
        std::cout << "SIMD Kernel is " << (1.0 / improvementRatio2) << "x SLOWER than Non-SIMD" << std::endl;
    }
    std::cout << "==================================" << std::endl;

    return EXIT_SUCCESS;
}

auto main() -> int
{
    std::cout << "Check enabled accelerator tags:" << std::endl;
    alpaka::printTagNames<alpaka::EnabledAccTags>();
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}
