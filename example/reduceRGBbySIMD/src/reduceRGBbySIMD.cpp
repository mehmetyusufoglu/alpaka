#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>
#include <iostream>
#include <random>
#include <chrono>
#include "simd_library.hpp"

class GrayscaleSIMDKernel {
public:
    ALPAKA_NO_HOST_ACC_WARNING
        template<typename Acc>
        ALPAKA_FN_ACC void operator()(Acc const& acc,
                                      float* r,
                                      float* g,
                                      float* b,
                                      float* grayscale,
                                      size_t size) const
    {
        // Use accelerator-specific SIMD constants
        const PortableSimd<float, Acc> COEFF_R(0.299f);
        const PortableSimd<float, Acc> COEFF_G(0.587f);
        const PortableSimd<float, Acc> COEFF_B(0.114f);

        const size_t globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        const size_t globalSize = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];
        constexpr size_t simdWidth = PortableSimd<float, Acc>::size();

        for(size_t i = globalIdx * simdWidth; i < size; i += globalSize * simdWidth)
        {
            PortableSimd<float, Acc> simdR, simdG, simdB;
            simdR.load(&r[i]);
            simdG.load(&g[i]);
            simdB.load(&b[i]);

            auto simdGray = simdR * COEFF_R + simdG * COEFF_G + simdB * COEFF_B;
            simdGray.store(&grayscale[i]);
        }
    }
};

// Non-SIMD Kernel
class GrayscaleNonSIMDKernel {
public:
    ALPAKA_NO_HOST_ACC_WARNING
        template<typename Acc>
        ALPAKA_FN_ACC auto operator()(Acc const& acc, const float* inputR, const float* inputG, const float* inputB, float* result) const {
        size_t globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

               // Scalar computation for grayscale
        result[globalIdx] = 0.299 * inputR[globalIdx] + 0.587 * inputG[globalIdx] + 0.114 * inputB[globalIdx];
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

    Idx const numElements = 32*1024*1024;
    Idx const elementsPerThread = 1;
    alpaka::Vec<Dim, Idx> const extent(numElements);

    using Data = float;
    using DevHost = alpaka::DevCpu;
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    using BufHost = alpaka::Buf<DevHost, Data, Dim, Idx>;
    BufHost bufHostR(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost bufHostG(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost bufHostB(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost bufHostResult(alpaka::allocBuf<Data, Idx>(devHost, extent));

           // Fill input data with random RGB values
    std::random_device rd;
    std::default_random_engine eng{rd()};
    std::uniform_real_distribution<Data> dist(0.0, 1.0);
    for (Idx i = 0; i < numElements; ++i) {
        bufHostR[i] = dist(eng);
        bufHostG[i] = dist(eng);
        bufHostB[i] = dist(eng);
    }

    using BufAcc = alpaka::Buf<DevAcc, Data, Dim, Idx>;
    BufAcc bufAccR(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccG(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccB(alpaka::allocBuf<Data, Idx>(devAcc, extent));
    BufAcc bufAccResult(alpaka::allocBuf<Data, Idx>(devAcc, extent));

    alpaka::memcpy(queue, bufAccR, bufHostR);
    alpaka::memcpy(queue, bufAccG, bufHostG);
    alpaka::memcpy(queue, bufAccB, bufHostB);
    constexpr size_t simdWidth = PortableSimd<Data,Acc>::size();
    std::cout << "simdWidth for type " <<  typeid(Data).name() << " is " << simdWidth << std::endl;
    std::cout << "numElements: " << numElements << std::endl;

           // Measure SIMD Kernel
    {
        GrayscaleSIMDKernel simdKernel;
        alpaka::KernelCfg<Acc> const kernelCfg = {extent, elementsPerThread};


        Idx const simdAdjustedExtent = numElements / simdWidth; // Adjust extent for SIMD processing
        alpaka::Vec<Dim, Idx> const extent(simdAdjustedExtent);

        alpaka::WorkDivMembers<Dim, Idx> workDivManual{simdAdjustedExtent, alpaka::Vec<Dim, Idx>::all(1), alpaka::Vec<Dim, Idx>::all(1)};
        std::cout << " " << std::endl;

        std::cout << "simdAdjustedExtent = numElements / simdWidth is equal to " <<  simdAdjustedExtent << std::endl;
        std::cout << workDivManual << std::endl;

        auto const taskKernel = alpaka::createTaskKernel<Acc>(workDivManual, simdKernel,
                                                              alpaka::getPtrNative(bufAccR),
                                                              alpaka::getPtrNative(bufAccG),
                                                              alpaka::getPtrNative(bufAccB),
                                                              alpaka::getPtrNative(bufAccResult),
                                                              numElements); // Pass size here

        alpaka::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        alpaka::enqueue(queue, taskKernel);
        alpaka::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();

        std::cout << "SIMD Kernel Execution Time (GridxSimdsize covers full data): " << std::chrono::duration<float>(endT - beginT).count() << "s\n";
    }


           // Measure SIMD Kernel
    {
        GrayscaleSIMDKernel simdKernel;
        alpaka::KernelCfg<Acc> const kernelCfg = {extent, elementsPerThread};

        Idx const simdAdjustedExtent = numElements / 32; // Adjust extent for SIMD processing
        alpaka::Vec<Dim, Idx> const extent(simdAdjustedExtent);

        alpaka::WorkDivMembers<Dim, Idx> workDivManual{simdAdjustedExtent, alpaka::Vec<Dim, Idx>::all(1), alpaka::Vec<Dim, Idx>::all(1)};
        std::cout << " " << std::endl;
        std::cout << workDivManual << std::endl;

        auto const taskKernel = alpaka::createTaskKernel<Acc>(workDivManual, simdKernel,
                                                              alpaka::getPtrNative(bufAccR),
                                                              alpaka::getPtrNative(bufAccG),
                                                              alpaka::getPtrNative(bufAccB),
                                                              alpaka::getPtrNative(bufAccResult),
                                                              numElements); // Pass size here

        alpaka::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        alpaka::enqueue(queue, taskKernel);
        alpaka::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();

        std::cout << "SIMD Kernel Execution Time (Grid x Simdsize does not cover full data): " << std::chrono::duration<float>(endT - beginT).count() << "s\n";
    }



           // Measure Non-SIMD Kernel
    {
        GrayscaleNonSIMDKernel nonSimdKernel;
        alpaka::KernelCfg<Acc> const kernelCfg = {extent, elementsPerThread};

        auto const workDiv = alpaka::getValidWorkDiv(kernelCfg, devAcc, nonSimdKernel,
                                                     alpaka::getPtrNative(bufAccR),
                                                     alpaka::getPtrNative(bufAccG),
                                                     alpaka::getPtrNative(bufAccB),
                                                     alpaka::getPtrNative(bufAccResult));

        auto const taskKernel = alpaka::createTaskKernel<Acc>(workDiv, nonSimdKernel,
                                                              alpaka::getPtrNative(bufAccR),
                                                              alpaka::getPtrNative(bufAccG),
                                                              alpaka::getPtrNative(bufAccB),
                                                              alpaka::getPtrNative(bufAccResult));

        alpaka::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        alpaka::enqueue(queue, taskKernel);
        alpaka::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();

        std::cout << "Non-SIMD Kernel Execution Time: " << std::chrono::duration<float>(endT - beginT).count() << "s\n";
    }

    return EXIT_SUCCESS;
}

auto main() -> int {
    std::cout << "Check enabled accelerator tags:" << std::endl;
    alpaka::printTagNames<alpaka::EnabledAccTags>();
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}
