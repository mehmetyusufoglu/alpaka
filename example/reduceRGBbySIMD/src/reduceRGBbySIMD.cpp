#include "simd_library.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>

// Define global constants
constexpr float scalarR = 0.299f;
constexpr float scalarG = 0.587f;
constexpr float scalarB = 0.114f;

// Template function for fuzzy equality
template<typename T>
[[maybe_unused]] bool FuzzyEqual(T a, T b)
{
    if constexpr(std::is_floating_point_v<T>)
    {
        return std::fabs(a - b) < (std::numeric_limits<T>::epsilon() * static_cast<T>(100.0));
    }
    else if constexpr(std::is_integral_v<T>)
    {
        return a == b;
    }
    else
    {
        static_assert(
            std::is_floating_point_v<T> || std::is_integral_v<T>,
            "FuzzyEqual<T> is only supported for integral or floating-point types.");
    }
}

// SIMD Kernel class template
template<typename T>
class GrayscaleSIMDKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc, T* r, T* g, T* b, T* grayscale, size_t size) const
    {
        // Use accelerator-specific SIMD constants
        PortableSimd<T, Acc> const COEFF_R(scalarR);
        PortableSimd<T, Acc> const COEFF_G(scalarG);
        PortableSimd<T, Acc> const COEFF_B(scalarB);
        const size_t globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        const size_t globalSize = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];
        constexpr size_t simdWidth = PortableSimd<T, Acc>::size();
        for(size_t i = globalIdx * simdWidth; i < size; i += globalSize * simdWidth)
        {
            PortableSimd<T, Acc> simdR, simdG, simdB;
            simdR.load(&r[i]);
            simdG.load(&g[i]);
            simdB.load(&b[i]);
            auto simdGray = simdR * COEFF_R + simdG * COEFF_G + simdB * COEFF_B;
            simdGray.store(&grayscale[i]);
        }
    }
};

// MATIAS SIMD Kernel class template
// SIMD Kernel class template
template<typename T>
class MatiasGrayscaleSIMDKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc, T* argb, T* grayscale, size_t size) const
    {
        constexpr size_t simdWidth = PortableSimd<T, Acc>::size();
        for(auto i = 0; i < size; i += simdWidth)
        {
            PortableSimd<T, Acc> simdA, simdR, simdG, simdB, simdGray, simdARGB;
            simdARGB.load(&argb[i]); // loads {it[0], it[1], it[2], ...}
                                     // Broadcast scalars to PortableSimd<T, Acc>
            PortableSimd<T, Acc> maskFF(0xFFu);

            // Extract components using bitwise operations
            simdA = simdARGB >> 24; // four uint32 becomes four a s again each of 4 a is uint32
            simdR = (simdARGB >> 16) & maskFF;
            simdG = (simdARGB >> 8) & maskFF;
            simdB = simdARGB & maskFF;

            // Broadcast scalars to PortableSimd<T, Acc>
            PortableSimd<T, Acc> coeff11(11u);
            PortableSimd<T, Acc> coeff16(16u);
            PortableSimd<T, Acc> coeff5(5u);
            PortableSimd<T, Acc> coeff32(32u);

            // Perform the calculation
            simdGray = (simdR * coeff11 + simdG * coeff16 + simdB * coeff5) / coeff32;
            simdARGB = simdGray | (simdGray << 8) | (simdGray << 16) | (simdA << 24);
            simdARGB.store(&grayscale[i]);
        }
    }
};

// Non-SIMD Kernel class template
template<typename T>
class GrayscaleNonSIMDKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename Acc>
    ALPAKA_FN_ACC auto operator()(Acc const& acc, T const* inputR, T const* inputG, T const* inputB, T* result) const
    {
        size_t globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        // Scalar computation for grayscale
        result[globalIdx] = scalarR * inputR[globalIdx] + scalarG * inputG[globalIdx] + scalarB * inputB[globalIdx];
    }
};

// Example function to compare SIMD and non-SIMD kernels
template<alpaka::concepts::Tag TAccTag>
auto example(TAccTag const&, size_t numElements) -> int
{
    // Select data type
    using T = float;

    using Dim = alpaka::DimInt<1u>;
    using Idx = std::size_t;
    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
    using DevAcc = alpaka::Dev<Acc>;

    std::cout << "Data type is: " << typeid(T).name() << std::endl;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;
    using QueueProperty = alpaka::Blocking;
    using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
    auto const platform = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platform, 0);
    QueueAcc queue(devAcc);
    Idx const elementsPerThread = 1;
    alpaka::Vec<Dim, Idx> const extent(numElements);
    using DevHost = alpaka::DevCpu;
    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);
    using BufHost = alpaka::Buf<DevHost, T, Dim, Idx>;
    BufHost bufHostR(alpaka::allocBuf<T, Idx>(devHost, extent));
    BufHost bufHostG(alpaka::allocBuf<T, Idx>(devHost, extent));
    BufHost bufHostB(alpaka::allocBuf<T, Idx>(devHost, extent));
    BufHost bufHostResult(alpaka::allocBuf<T, Idx>(devHost, extent));
    std::vector<T> referenceResult(numElements);
    // Fill input data with random RGB values
    std::random_device rd;
    std::default_random_engine eng{rd()};

    using DistributionType = std::conditional_t<
        std::is_floating_point_v<T>,
        std::uniform_real_distribution<T>,
        std::uniform_int_distribution<T>>;
    DistributionType dist(std::is_floating_point_v<T> ? 0.0 : 0, std::is_floating_point_v<T> ? 1.0 : 32);

    for(Idx i = 0; i < numElements; ++i)
    {
        bufHostR[i] = dist(eng);
        bufHostG[i] = dist(eng);
        bufHostB[i] = dist(eng);
        referenceResult.at(i) = scalarR * bufHostR[i] + scalarG * bufHostG[i] + scalarB * bufHostB[i];
    }
    using BufAcc = alpaka::Buf<DevAcc, T, Dim, Idx>;
    BufAcc bufAccR(alpaka::allocBuf<T, Idx>(devAcc, extent));
    BufAcc bufAccG(alpaka::allocBuf<T, Idx>(devAcc, extent));
    BufAcc bufAccB(alpaka::allocBuf<T, Idx>(devAcc, extent));
    BufAcc bufAccResult(alpaka::allocBuf<T, Idx>(devAcc, extent));
    alpaka::memcpy(queue, bufAccR, bufHostR);
    alpaka::memcpy(queue, bufAccG, bufHostG);
    alpaka::memcpy(queue, bufAccB, bufHostB);
    constexpr size_t simdWidth = PortableSimd<T, Acc>::size();
    std::cout << "simdWidth for type " << typeid(T).name() << " is " << simdWidth << std::endl;
    std::cout << "numElements: " << numElements << std::endl;
    // Define the lambda for result verification
    auto verifyResults = [&referenceResult, &numElements, &devHost, &queue](BufAcc const& computed)
    {
        auto computedHost = alpaka::allocBuf<T, Idx>(devHost, numElements);
        alpaka::memcpy(queue, computedHost, computed);
        for(Idx i = 0; i < numElements; ++i)
        {
            if(!FuzzyEqual<T>(referenceResult.at(i), computedHost[i]))
            {
                std::cout << "Result is wrong at index " << i << std::endl;
            }
        }
    };
    // Measure SIMD Kernel
    {
        GrayscaleSIMDKernel<T> simdKernel;
        alpaka::KernelCfg<Acc> const kernelCfg = {extent, elementsPerThread};
        Idx const simdAdjustedExtent = numElements / simdWidth; // Adjust extent for SIMD processing
        alpaka::Vec<Dim, Idx> const extent(simdAdjustedExtent);
        alpaka::WorkDivMembers<Dim, Idx> workDivManual{
            simdAdjustedExtent,
            alpaka::Vec<Dim, Idx>::all(1),
            alpaka::Vec<Dim, Idx>::all(1)};
        std::cout << " " << std::endl;
        std::cout << "simdAdjustedExtent = numElements / simdWidth is equal to " << simdAdjustedExtent << std::endl;
        std::cout << workDivManual << std::endl;
        auto const taskKernel = alpaka::createTaskKernel<Acc>(
            workDivManual,
            simdKernel,
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
        // Call the lambda to verify results
        verifyResults(bufAccResult);
        std::cout << "SIMD1to1: SIMD Kernel Execution Time (GridxSimdsize covers full data): "
                  << std::chrono::duration<float>(endT - beginT).count() << "s\n";
    }
    // Measure SIMD Kernel with different extent
    {
        GrayscaleSIMDKernel<T> simdKernel;
        alpaka::KernelCfg<Acc> const kernelCfg = {extent, elementsPerThread};
        Idx const simdAdjustedExtent = numElements / 32; // Adjust extent for SIMD processing
        alpaka::Vec<Dim, Idx> const extent(simdAdjustedExtent);
        alpaka::WorkDivMembers<Dim, Idx> workDivManual{
            simdAdjustedExtent,
            alpaka::Vec<Dim, Idx>::all(1),
            alpaka::Vec<Dim, Idx>::all(1)};
        std::cout << " " << std::endl;
        std::cout << workDivManual << std::endl;
        auto const taskKernel = alpaka::createTaskKernel<Acc>(
            workDivManual,
            simdKernel,
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
        // Call the lambda to verify results
        verifyResults(bufAccResult);
        std::cout << "SIMD1toN: SIMD Kernel Execution Time (Grid x Simdsize does not cover full data): "
                  << std::chrono::duration<float>(endT - beginT).count() << "s\n";
    }

    // Measure SIMD Kernel MAtias Kretz
    {
        MatiasGrayscaleSIMDKernel<T> simdKernel;
        BufAcc bufAccARGB(alpaka::allocBuf<T, Idx>(devAcc, extent));
        alpaka::KernelCfg<Acc> const kernelCfg = {extent, elementsPerThread};
        Idx const simdAdjustedExtent = numElements / simdWidth; // Adjust extent for SIMD processing
        alpaka::Vec<Dim, Idx> const extent(simdAdjustedExtent);
        alpaka::WorkDivMembers<Dim, Idx> workDivManual{
            simdAdjustedExtent,
            alpaka::Vec<Dim, Idx>::all(1),
            alpaka::Vec<Dim, Idx>::all(1)};
        std::cout << " " << std::endl;
        std::cout << "simdAdjustedExtent = numElements / simdWidth is equal to " << simdAdjustedExtent << std::endl;
        std::cout << workDivManual << std::endl;
        auto const taskKernel = alpaka::createTaskKernel<Acc>(
            workDivManual,
            simdKernel,
            alpaka::getPtrNative(bufAccARGB),
            alpaka::getPtrNative(bufAccResult),
            numElements); // Pass size here
        alpaka::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        alpaka::enqueue(queue, taskKernel);
        alpaka::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        // Call the lambda to verify results
        // verifyResults(bufAccResult);
        std::cout << "MatiasSIMD1to1: SIMD Kernel Execution Time (GridxSimdsize covers full data): "
                  << std::chrono::duration<float>(endT - beginT).count() << "s\n";
    }

    // Measure Non-SIMD Kernel
    {
        GrayscaleNonSIMDKernel<T> nonSimdKernel;
        alpaka::KernelCfg<Acc> const kernelCfg = {extent, elementsPerThread};
        auto const workDiv = alpaka::getValidWorkDiv(
            kernelCfg,
            devAcc,
            nonSimdKernel,
            alpaka::getPtrNative(bufAccR),
            alpaka::getPtrNative(bufAccG),
            alpaka::getPtrNative(bufAccB),
            alpaka::getPtrNative(bufAccResult));
        auto const taskKernel = alpaka::createTaskKernel<Acc>(
            workDiv,
            nonSimdKernel,
            alpaka::getPtrNative(bufAccR),
            alpaka::getPtrNative(bufAccG),
            alpaka::getPtrNative(bufAccB),
            alpaka::getPtrNative(bufAccResult));
        alpaka::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        alpaka::enqueue(queue, taskKernel);
        alpaka::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        // Call the lambda to verify results
        verifyResults(bufAccResult);
        std::cout << "NonSIMD: Non-SIMD Kernel Execution Time: " << std::chrono::duration<float>(endT - beginT).count()
                  << "s\n";
    }
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    // Default number of elements is 2^25
    size_t numElements = 1 << 25; // 2^25

    // Parse command-line argument
    if(argc > 1)
    {
        std::string arg = argv[1];
        if(arg.find("numElements=") == 0)
        {
            try
            {
                numElements = std::stoul(arg.substr(12));
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Invalid number of elements: " << arg.substr(12) << std::endl;
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Number of elements out of range: " << arg.substr(12) << std::endl;
                return EXIT_FAILURE;
            }
        }
        else
        {
            std::cerr << "Usage: " << argv[0] << " numElements=<value>" << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::cout << "Check enabled accelerator tags:" << std::endl;
    alpaka::printTagNames<alpaka::EnabledAccTags>();

    // Use double as the data type
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag, numElements); });
}
