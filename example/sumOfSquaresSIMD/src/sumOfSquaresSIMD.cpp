#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>
#include <iostream>
#include <random>
#include <chrono>
#include "simd_library.hpp"

// SIMD Kernel assuming gridsize is smaller than dataSize/simd_register_size (ie 4 or 8)
class SumOfSquaresSIMDKernel {
public:
    ALPAKA_NO_HOST_ACC_WARNING
        template<typename Acc, size_t simdWidth = PortableSimd<float, Acc>::size()>
        ALPAKA_FN_ACC auto operator()(Acc const& acc, const float* input, float* result, size_t dataSize) const {
        size_t globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        size_t gridSize = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        //constexpr size_t simdWidth;
        float localSum = 0.0;

               // SIMD computation for the thread
        for (size_t i = globalIdx * simdWidth; i < dataSize; i += gridSize * simdWidth) {
            PortableSimd<float, Acc> simd_data;
            simd_data.load(&input[i]);

                   // Square the values and accumulate
            PortableSimd<float, Acc> simd_squared = simd_data * simd_data;
            localSum += simd_squared.sum();
        }

               // Directly accumulate the result using atomicAdd
        alpaka::atomicAdd(acc, result, localSum, alpaka::hierarchy::Blocks{});
    }
};


// SIMD Kernel
class SumOfSquaresSIMDKernel1Thread1SIMD {
public:
    ALPAKA_NO_HOST_ACC_WARNING
        template<typename Acc>
        ALPAKA_FN_ACC auto operator()(Acc const& acc, const float* input, float* result, size_t dataSize) const {
        size_t globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        size_t globalSize = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];
        constexpr size_t simdWidth = PortableSimd<float, Acc>::size();
        float localSum = 0.0;
               // SIMD computation for the thread
        //   for (size_t i = ; i < dataSize * simdWidth; i += globalSize * simdWidth) {
        PortableSimd<float, Acc> simd_data;
        simd_data.load(&input[globalIdx * simdWidth]);

               // Square the values and accumulate
        PortableSimd<float, Acc> simd_squared = simd_data * simd_data;
        //localSum += simd_squared.sum();
        // }
               // Directly accumulate the result using atomicAdd
        alpaka::atomicAdd(acc, result, simd_squared.sum(), alpaka::hierarchy::Blocks{});
    }
};

// Non-SIMD Kernel
class SumOfSquaresNonSIMDKernel {
public:
    ALPAKA_NO_HOST_ACC_WARNING
        template<typename Acc>
        ALPAKA_FN_ACC auto operator()(Acc const& acc, const float* input, float* result, size_t dataSize) const {
        size_t globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        size_t globalSize = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        float localSum = 0.0;

               // Scalar computation for the thread
        for (size_t i = globalIdx; i < dataSize; i += globalSize) {
            float value = input[i];
            localSum += value * value; // Compute square and accumulate
        }

               // Directly accumulate the result using atomicAdd
        alpaka::atomicAdd(acc, result, localSum, alpaka::hierarchy::Blocks{});
    }
};

// Example function to compare SIMD and non-SIMD kernels
template<alpaka::concepts::Tag TAccTag>
auto example(TAccTag const&) -> int {
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

    Idx const numElements(32*1024*1024);
    Idx const elementsPerThread(1);
    alpaka::Vec<Dim, Idx> const extent(numElements);

    using Data = float;
    using DevHost = alpaka::DevCpu;
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    using BufHost = alpaka::Buf<DevHost, Data, Dim, Idx>;
    BufHost bufHostA(alpaka::allocBuf<Data, Idx>(devHost, extent));

    std::random_device rd;
    std::default_random_engine eng{rd()};
    std::uniform_real_distribution<Data> dist(1.0, 42.0);
    float referenceSum = 0.0;
    for (Idx i = 0; i < numElements; ++i) {
        bufHostA[i] = dist(eng);
        referenceSum += bufHostA[i] * bufHostA[i];
    }

    using BufAcc = alpaka::Buf<DevAcc, Data, Dim, Idx>;
    BufAcc bufAccA(alpaka::allocBuf<Data, Idx>(devAcc, extent));

    alpaka::memcpy(queue, bufAccA, bufHostA);

    constexpr size_t simdWidth = PortableSimd<Data, Acc>::size();
    std::cout << "simdWidth for type " <<  typeid(Data).name() << " is " << simdWidth << std::endl;
    std::cout << "numElements: " << numElements << std::endl;
           // Measure SumOfSquaresSIMDKernel1Thread1SIMD Kernel
    {
        Data result = 0.0;
        SumOfSquaresSIMDKernel1Thread1SIMD simdKernel;
        // call by deividing to simd dataSize?
        // alpaka::KernelCfg<Acc> const kernelCfg = {extent, elementsPerThread};

        // auto const workDiv = alpaka::getValidWorkDiv(kernelCfg, devAcc, simdKernel, alpaka::getPtrNative(bufAccA), &result, numElements);

        Idx const simdAdjustedExtent = numElements / simdWidth; // Adjust extent for SIMD processing
        alpaka::Vec<Dim, Idx> const extent(simdAdjustedExtent);


        std::cout << "simdAdjustedExtent = numElements / simdWidth is equal to " <<  simdAdjustedExtent << std::endl;

        alpaka::WorkDivMembers<Dim, Idx> workDivManual{simdAdjustedExtent, alpaka::Vec<Dim, Idx>::all(1), alpaka::Vec<Dim, Idx>::all(1)};
        auto const taskKernel = alpaka::createTaskKernel<Acc>(workDivManual, simdKernel, alpaka::getPtrNative(bufAccA), &result, numElements);

        alpaka::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        alpaka::enqueue(queue, taskKernel);
        alpaka::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();

        std::cout << "SIMD Kernel Execution Time (Full Data coverd by simd-size*GridSize): " << std::chrono::duration<float>(endT - beginT).count() << "s\n";
        std::cout << "SIMD Kernel Result: " << result << "\n";
    }


           // Measure SumOfSquaresSIMDKernel Kernel
    {
        Data result = 0.0;
        SumOfSquaresSIMDKernel simdKernel;
        // call by deividing to simd dataSize?
        // alpaka::KernelCfg<Acc> const kernelCfg = {extent, elementsPerThread};

               // auto const workDiv = alpaka::getValidWorkDiv(kernelCfg, devAcc, simdKernel, alpaka::getPtrNative(bufAccA), &result, numElements);

        Idx const simdAdjustedExtent = numElements / 32; // much smaller grid then being able to cover all data even with simd
        alpaka::Vec<Dim, Idx> const extent(simdAdjustedExtent);
        alpaka::WorkDivMembers<Dim, Idx> workDivManual{simdAdjustedExtent, alpaka::Vec<Dim, Idx>::all(1), alpaka::Vec<Dim, Idx>::all(1)};
        auto const taskKernel = alpaka::createTaskKernel<Acc>(workDivManual, simdKernel, alpaka::getPtrNative(bufAccA), &result, numElements);

        alpaka::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        alpaka::enqueue(queue, taskKernel);
        alpaka::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << " " << std::endl;
        std::cout << "newExtent = (numElements / 32)  is equal to " <<  simdAdjustedExtent << std::endl;
        std::cout << "SIMD Kernel Execution Time (Not full data covered by simdsize*gridsize): " << std::chrono::duration<float>(endT - beginT).count() << "s\n";
        std::cout << "SIMD Kernel Result: " << result << "\n";
    }
           // Measure Non-SIMD Kernel
    {
        Data result = 0.0;
        SumOfSquaresNonSIMDKernel nonSimdKernel;
        alpaka::KernelCfg<Acc> const kernelCfg = {extent, elementsPerThread};

        auto const workDiv = alpaka::getValidWorkDiv(kernelCfg, devAcc, nonSimdKernel, alpaka::getPtrNative(bufAccA), &result, numElements);

        auto const taskKernel = alpaka::createTaskKernel<Acc>(workDiv, nonSimdKernel, alpaka::getPtrNative(bufAccA), &result, numElements);

        alpaka::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        alpaka::enqueue(queue, taskKernel);
        alpaka::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << " " << std::endl;
        std::cout << "Non-SIMD Kernel Execution Time: " << std::chrono::duration<float>(endT - beginT).count() << "s\n";
        std::cout << "Non-SIMD Kernel Result: " << result << "\n";
    }

    std::cout << "Reference Sum of Squares: " << referenceSum << "\n";

    return EXIT_SUCCESS;
}

auto main() -> int {
    std::cout << "Check enabled accelerator tags:" << std::endl;
    alpaka::printTagNames<alpaka::EnabledAccTags>();
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}
