#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>

//! IMPORTANT: SIMD Performance Optimization Requirements
//! For optimal SIMD performance, compile with -O3 optimization and -march=native.
//! performance (1.1-2.3x slower instead of 2-8x faster). For cross-platform builds or specific targeting, use explicit
//! flags like -mavx2, but -march=native automatically enables all supported instructions on the target CPU.

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
        alpaka::simd::PortableSimd<T, Acc> const COEFF_R(static_cast<T>(scalarR));
        alpaka::simd::PortableSimd<T, Acc> const COEFF_G(static_cast<T>(scalarG));
        alpaka::simd::PortableSimd<T, Acc> const COEFF_B(static_cast<T>(scalarB));
        const size_t globalIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        const size_t globalSize = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0];
        constexpr size_t simdWidth = alpaka::simd::PortableSimd<T, Acc>::size();
        for(size_t i = globalIdx * simdWidth; i < size; i += globalSize * simdWidth)
        {
            alpaka::simd::PortableSimd<T, Acc> simdR, simdG, simdB;
            simdR.load(&r[i]);
            simdG.load(&g[i]);
            simdB.load(&b[i]);
            auto simdGray = simdR * COEFF_R + simdG * COEFF_G + simdB * COEFF_B;
            simdGray.store(&grayscale[i]);
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
        result[globalIdx] = static_cast<T>(scalarR) * inputR[globalIdx] + static_cast<T>(scalarG) * inputG[globalIdx]
                            + static_cast<T>(scalarB) * inputB[globalIdx];
    }
};

// Template function to test different data types
template<typename T, typename Acc, typename Queue>
void testDataType(Queue& queue, alpaka::Dev<Acc> const& devAcc, size_t numElements, std::string const& typeName)
{
    using Dim = alpaka::DimInt<1u>;
    using Idx = std::size_t;

    std::cout << "\n=== Testing with " << typeName << " ===" << std::endl;

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
        referenceResult.at(i) = static_cast<T>(scalarR) * bufHostR[i] + static_cast<T>(scalarG) * bufHostG[i]
                                + static_cast<T>(scalarB) * bufHostB[i];
    }

    using BufAcc = alpaka::Buf<alpaka::Dev<Acc>, T, Dim, Idx>;
    BufAcc bufAccR(alpaka::allocBuf<T, Idx>(devAcc, extent));
    BufAcc bufAccG(alpaka::allocBuf<T, Idx>(devAcc, extent));
    BufAcc bufAccB(alpaka::allocBuf<T, Idx>(devAcc, extent));
    BufAcc bufAccResult(alpaka::allocBuf<T, Idx>(devAcc, extent));

    alpaka::memcpy(queue, bufAccR, bufHostR);
    alpaka::memcpy(queue, bufAccG, bufHostG);
    alpaka::memcpy(queue, bufAccB, bufHostB);

    constexpr size_t simdWidth = alpaka::simd::PortableSimd<T, Acc>::size();
    std::cout << "SIMD width for " << typeName << " is " << simdWidth << std::endl;
    std::cout << "Number of elements: " << numElements << std::endl;

    // Variables to store timing for comparison
    float simdTime = 0.0f;
    float nonSimdTime = 0.0f;

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
        Idx const simdAdjustedExtent = numElements / simdWidth;
        alpaka::Vec<Dim, Idx> const extentSIMD(simdAdjustedExtent);
        alpaka::WorkDivMembers<Dim, Idx> workDivManual{
            simdAdjustedExtent,
            alpaka::Vec<Dim, Idx>::all(1),
            alpaka::Vec<Dim, Idx>::all(1)};

        std::cout << "SIMD adjusted extent = " << simdAdjustedExtent << std::endl;

        auto const taskKernel = alpaka::createTaskKernel<Acc>(
            workDivManual,
            simdKernel,
            alpaka::getPtrNative(bufAccR),
            alpaka::getPtrNative(bufAccG),
            alpaka::getPtrNative(bufAccB),
            alpaka::getPtrNative(bufAccResult),
            numElements);

        alpaka::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();
        alpaka::enqueue(queue, taskKernel);
        alpaka::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();

        verifyResults(bufAccResult);
        simdTime = std::chrono::duration<float>(endT - beginT).count();
        std::cout << "SIMD Kernel Time: " << simdTime << "s\n";
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

        verifyResults(bufAccResult);
        nonSimdTime = std::chrono::duration<float>(endT - beginT).count();
        std::cout << "Non-SIMD Kernel Time: " << nonSimdTime << "s\n";
    }

    // Print SIMD improvement ratios
    std::cout << "\n=== SIMD Performance Analysis ===" << std::endl;
    std::cout << "SIMD Time: " << simdTime << "s" << std::endl;
    std::cout << "Non-SIMD Time: " << nonSimdTime << "s" << std::endl;

    float improvementRatio = nonSimdTime / simdTime;
    if(improvementRatio > 1.0f)
    {
        std::cout << "SIMD is " << improvementRatio << "x FASTER than Non-SIMD" << std::endl;
    }
    else
    {
        std::cout << "SIMD is " << (1.0f / improvementRatio) << "x SLOWER than Non-SIMD" << std::endl;
    }
    std::cout << "==================================" << std::endl;
}

// Example function to compare SIMD and non-SIMD kernels
template<alpaka::concepts::Tag TAccTag>
auto example(TAccTag const&, size_t numElements) -> int
{
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

    // Test with float
    testDataType<float, Acc>(queue, devAcc, numElements, "float");

    // Test with double
    testDataType<double, Acc>(queue, devAcc, numElements, "double");

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

    // Assert numElements to be a power of 2
    if((numElements & (numElements - 1)) != 0)
    {
        std::cerr << "Error: numElements must be a power of 2." << std::endl;
        return EXIT_FAILURE;
    }

    // Calculate the power of 2
    size_t powerOf2 = 0;
    size_t temp = numElements;
    while(temp >>= 1)
    {
        powerOf2++;
    }

    // Print numElements in format of power of 2
    std::cout << "numElements: 2^" << powerOf2 << " (" << numElements << ")" << std::endl;

    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag, numElements); });
}
