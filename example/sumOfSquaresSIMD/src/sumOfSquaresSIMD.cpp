#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>
#include <iostream>
#include <random>
#include <chrono>
#include "simd_library.hpp"

// SIMD Kernel
class SumOfSquaresSIMDKernel {
public:
    ALPAKA_NO_HOST_ACC_WARNING
        template<typename Acc>
        ALPAKA_FN_ACC auto operator()(Acc const& acc, const double* input, double* result, size_t size) const {
        size_t globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        size_t globalSize = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        constexpr size_t simdWidth = PortableSimd<double>::size();
        double localSum = 0.0;

               // SIMD computation for the thread
        for (size_t i = globalIdx * simdWidth; i < size; i += globalSize * simdWidth) {
            PortableSimd<double> simd_data;
            simd_data.load(&input[i]);

                   // Square the values and accumulate
            PortableSimd<double> simd_squared = simd_data * simd_data;
            localSum += simd_squared.sum();
        }

               // Directly accumulate the result using atomicAdd
        alpaka::atomicAdd(acc, result, localSum, alpaka::hierarchy::Blocks{});
    }
};

// Non-SIMD Kernel
class SumOfSquaresNonSIMDKernel {
public:
    ALPAKA_NO_HOST_ACC_WARNING
        template<typename Acc>
        ALPAKA_FN_ACC auto operator()(Acc const& acc, const double* input, double* result, size_t size) const {
        size_t globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        size_t globalSize = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        double localSum = 0.0;

               // Scalar computation for the thread
        for (size_t i = globalIdx; i < size; i += globalSize) {
            double value = input[i];
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
    using Idx = std::size_t;

    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
    using DevAcc = alpaka::Dev<Acc>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;

    auto const platform = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platform, 0);
    QueueAcc queue(devAcc);

    Idx const numElements(22223456);
    Idx const elementsPerThread(256);
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
    for (Idx i = 0; i < numElements; ++i) {
        bufHostA[i] = dist(eng);
        referenceSum += bufHostA[i] * bufHostA[i];
    }

    using BufAcc = alpaka::Buf<DevAcc, Data, Dim, Idx>;
    BufAcc bufAccA(alpaka::allocBuf<Data, Idx>(devAcc, extent));

    alpaka::memcpy(queue, bufAccA, bufHostA);

           // Measure SIMD Kernel
    {
        Data result = 0.0;
        SumOfSquaresSIMDKernel simdKernel;
        alpaka::KernelCfg<Acc> const kernelCfg = {extent, elementsPerThread};

        auto const workDiv = alpaka::getValidWorkDiv(kernelCfg, devAcc, simdKernel, alpaka::getPtrNative(bufAccA), &result, numElements);

        auto const taskKernel = alpaka::createTaskKernel<Acc>(workDiv, simdKernel, alpaka::getPtrNative(bufAccA), &result, numElements);

        alpaka::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        alpaka::enqueue(queue, taskKernel);
        alpaka::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();

        std::cout << "SIMD Kernel Execution Time: " << std::chrono::duration<double>(endT - beginT).count() << "s\n";
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

        std::cout << "Non-SIMD Kernel Execution Time: " << std::chrono::duration<double>(endT - beginT).count() << "s\n";
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
