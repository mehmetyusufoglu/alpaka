#include "simd_library.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>

// Define global constants
constexpr auto scalarR = 11u;
constexpr auto scalarB = 5u;
constexpr auto scalarG = 16u;
constexpr auto scalar32 = 32u;

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

// MATIAS SIMD Kernel class template
template<typename T>
class MatiasGrayscaleSIMDKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc, T const* argb, T* grayscale, const size_t size) const
    {
        size_t const globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        size_t const globalSize = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];
        constexpr size_t simdWidth = PortableSimd<T, Acc>::size();

        // Broadcast scalars to PortableSimd<T, Acc>
        PortableSimd<T, Acc> const coeffR(scalarR);
        PortableSimd<T, Acc> const coeffG(scalarG);
        PortableSimd<T, Acc> const coeffB(scalarB);
        PortableSimd<T, Acc> const coeff32(scalar32);
        PortableSimd<T, Acc> const maskFF(0xFFu);

        for(size_t i = globalIdx * simdWidth; i < size; i += globalSize * simdWidth)
        {
            PortableSimd<T, Acc> simdARGB;
            simdARGB.load(&argb[i]); // loads {it[0], it[1], it[2], ...}

            // Extract components using bitwise operations
            PortableSimd<T, Acc> simdA = simdARGB >> 24; // four uint32 becomes four a s again each of 4 a is uint32
            PortableSimd<T, Acc> simdR = (simdARGB >> 16) & maskFF;
            PortableSimd<T, Acc> simdG = (simdARGB >> 8) & maskFF;
            PortableSimd<T, Acc> simdB = simdARGB & maskFF;

            // Perform the calculation
            PortableSimd<T, Acc> const simdGray = (simdR * coeffR + simdG * coeffG + simdB * coeffB) / coeff32;

            // Reconstruct ARGB value
            PortableSimd<T, Acc> const simdARGBResult = simdGray | (simdGray << 8) | (simdGray << 16) | (simdA << 24);

            // Store the result
            simdARGBResult.store(&grayscale[i]);
        }
    }
};

// Non-SIMD Kernel class template
template<typename T>
class MatiasGrayscaleKernelNonSIMD
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc, T const* argb, T* grayscale, const size_t size) const
    {
        const size_t globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        const size_t globalSize = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];

        for(size_t i = globalIdx; i < size; i += globalSize)
        {
            // Extract components using bitwise operations
            T a = (argb[i] >> 24) & 0xFFu;
            T r = (argb[i] >> 16) & 0xFFu;
            T g = (argb[i] >> 8) & 0xFFu;
            T b = argb[i] & 0xFFu;

            // Compute grayscale value
            const T gray = (r * scalarR + g * scalarG + b * scalarB) / scalar32;

            // Reconstruct ARGB value
            grayscale[i] = (gray << 16) | (gray << 8) | gray | (a << 24);
        }
    }
};

// Example function to compare SIMD and non-SIMD kernels
template<alpaka::concepts::Tag TAccTag>
auto example(TAccTag const&, size_t numElements) -> int
{
    // Select data type
    using T = uint32_t;
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
    std::cout << "Device:" << alpaka::getName(devAcc) << std::endl;
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
    DistributionType dist(std::is_floating_point_v<T> ? 0.0 : 0, std::is_floating_point_v<T> ? 1.0 : 255);
    for(Idx i = 0; i < numElements; ++i)
    {
        bufHostR[i] = dist(eng);
        bufHostG[i] = dist(eng);
        bufHostB[i] = dist(eng);
        // Pack ARGB value
        T argb = (0xFFu << 24) | (bufHostR[i] << 16) | (bufHostG[i] << 8) | bufHostB[i];
        // Extract components
        T a = (argb >> 24) & 0xFFu;
        T r = (argb >> 16) & 0xFFu;
        T g = (argb >> 8) & 0xFFu;
        T b = argb & 0xFFu;
        // Compute grayscale value
        T gray = (r * scalarR + g * scalarG + b * scalarB) / scalar32;
        // Reconstruct ARGB value
        referenceResult.at(i) = (gray << 16) | (gray << 8) | gray | (a << 24);
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

    BufAcc bufAccARGB(alpaka::allocBuf<T, Idx>(devAcc, extent));
    for(Idx i = 0; i < numElements; ++i)
    {
        bufAccARGB[i] = (bufHostR[i] << 16) | (bufHostG[i] << 8) | bufHostB[i];
    }

    // Measure SIMD Kernel MAtias Kretz
    {
        MatiasGrayscaleSIMDKernel<T> simdKernel;


        Idx const simdAdjustedExtent = numElements / simdWidth; // Adjust extent for SIMD processing
        alpaka::Vec<Dim, Idx> const extent(simdAdjustedExtent);
        alpaka::KernelCfg<Acc> const kernelCfg = {extent, elementsPerThread};
        alpaka::WorkDivMembers<Dim, Idx> workDivManual{
            extent,
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
        //   verifyResults(bufAccResult);
        std::cout << "SIMD Kernel Execution Time (GridxSimdsize covers full data)" << std::endl;
        std::cout << "MatiasSIMD1to1:" << std::chrono::duration<float>(endT - beginT).count() << "s\n";
    }

    // Measure Non-SIMD Kernel
    {
        MatiasGrayscaleKernelNonSIMD<T> nonSimdKernel;
        alpaka::Vec<Dim, Idx> const extent(numElements);
        alpaka::KernelCfg<Acc> const kernelCfg = {extent, elementsPerThread};
        alpaka::WorkDivMembers<Dim, Idx> workDivManual{
            extent,
            alpaka::Vec<Dim, Idx>::all(1),
            alpaka::Vec<Dim, Idx>::all(1)};
        auto const taskKernel = alpaka::createTaskKernel<Acc>(
            workDivManual,
            nonSimdKernel,
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
        std::cout << "Non-SIMD:" << std::chrono::duration<float>(endT - beginT).count() << "s\n";
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


    std::cout << "numElements: " << numElements << std::endl;

    // Use double as the data type
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag, numElements); });
}
