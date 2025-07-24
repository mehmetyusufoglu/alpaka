# Alpaka SIMD Support

This document describes the SIMD (Single Instruction, Multiple Data) support integrated into Alpaka.

## Overview

The Alpaka SIMD implementation provides a portable interface for SIMD operations across different accelerator types. It automatically adapts to use:

- **CPU Accelerators**: Real SIMD instructions (SSE, AVX, AVX2, AVX-512, NEON) via `std::experimental::simd`
- **GPU Accelerators**: Scalar operations (SIMD width = 1) since GPUs handle parallelism through many threads

## Features

- **Portable API**: Same interface across all accelerator types
- **Automatic SIMD width detection**: Optimal SIMD width based on data type and target architecture
- **Type safety**: Template-based design ensures type correctness
- **Zero overhead**: GPU implementations compile to simple scalar operations
- **Standard compliance**: Uses `std::experimental::simd` for CPU implementations

## API Reference

### Core Classes

#### `alpaka::simd::PortableSimd<T, TAcc>`

The main SIMD class template providing vectorized operations.

**Template Parameters:**
- `T`: Element type (float, double, int32_t, etc.)
- `TAcc`: Accelerator type

**Static Methods:**
- `size()`: Returns the SIMD width (number of elements processed in parallel)

**Methods:**
- `PortableSimd()`: Default constructor (initializes to zero)
- `PortableSimd(T scalar)`: Scalar constructor (broadcasts scalar to all lanes)
- `load(const T* ptr)`: Load data from memory
- `store(T* ptr)`: Store data to memory
- `sum()`: Horizontal sum of all elements
- `operator[](size_t idx)`: Access individual element

**Operators:**
- `+`, `-`, `*`, `/`: Element-wise arithmetic
- `+=`, `-=`, `*=`, `/=`: Compound assignment

### Utility Functions and Traits

#### `alpaka::simd::simd_width_v<T, TAcc>`
Compile-time constant for SIMD width.

#### `alpaka::simd::isSimdAligned<T, TAcc>(size_t size)`
Check if a size is SIMD-aligned.

#### `alpaka::simd::roundUpToSimdWidth<T, TAcc>(size_t size)`
Round up size to next SIMD-aligned boundary.

#### `alpaka::simd::getSimdOpCount<T, TAcc>(size_t size)`
Get number of SIMD operations needed for given size.

## Usage Examples

### Basic Vector Addition

```cpp
struct VectorAddKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, T const* a, T const* b, T* c, std::size_t n) const -> void
    {
        auto const idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        
        using SimdType = alpaka::simd::PortableSimd<T, TAcc>;
        constexpr auto simdWidth = SimdType::size();
        
        for(std::size_t i = idx * simdWidth; i < n; i += simdWidth)
        {
            if(i + simdWidth <= n)
            {
                SimdType va, vb;
                va.load(&a[i]);
                vb.load(&b[i]);
                
                auto vc = va + vb;
                vc.store(&c[i]);
            }
            else
            {
                // Handle remaining elements
                for(std::size_t j = i; j < n && j < i + simdWidth; ++j)
                {
                    c[j] = a[j] + b[j];
                }
            }
        }
    }
};
```

### Reduction Operations

```cpp
struct DotProductKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, T const* a, T const* b, T* result, std::size_t n) const -> void
    {
        using SimdType = alpaka::simd::PortableSimd<T, TAcc>;
        constexpr auto simdWidth = SimdType::size();
        
        SimdType accumulator(T{0});
        
        std::size_t i = 0;
        for(; i + simdWidth <= n; i += simdWidth)
        {
            SimdType va, vb;
            va.load(&a[i]);
            vb.load(&b[i]);
            accumulator += va * vb;
        }
        
        // Handle remaining elements + horizontal reduction
        T scalar_sum = accumulator.sum();
        for(; i < n; ++i)
        {
            scalar_sum += a[i] * b[i];
        }
        
        *result = scalar_sum;
    }
};
```

## Performance Considerations

### CPU Performance
- Enable appropriate compiler flags: `-march=native`, `-mavx2`, etc.
- Ensure data alignment for optimal performance
- Use SIMD-friendly memory access patterns (contiguous, aligned)

### GPU Performance
- SIMD operations compile to scalar operations with no overhead
- Focus on thread-level parallelism rather than SIMD parallelism
- The same kernel code works on both CPU and GPU

## SIMD Widths by Architecture

| Architecture | float | double | int32_t | int8_t |
|--------------|-------|---------|---------|--------|
| SSE          | 4     | 2       | 4       | 16     |
| AVX          | 8     | 4       | 8       | 32     |
| AVX2         | 8     | 4       | 8       | 32     |
| AVX-512      | 16    | 8       | 16      | 64     |
| ARM NEON     | 4     | 2       | 4       | 16     |
| GPU (CUDA/HIP) | 1   | 1       | 1       | 1      |

## Compiler Requirements

- **CPU**: Requires `std::experimental::simd` support
  - GCC 11+ with `-fconcepts`
  - Clang 14+ with appropriate flags
  - MSVC 2022+ (latest versions)
- **GPU**: Standard CUDA/HIP compiler support

## Integration with Existing Code

The SIMD implementation is designed to integrate seamlessly with existing Alpaka code:

1. **Include**: Add `#include <alpaka/alpaka.hpp>` (SIMD is included automatically)
2. **Replace scalar operations**: Use `PortableSimd<T, TAcc>` instead of `T`
3. **Handle remainder elements**: Always handle non-SIMD-aligned data
4. **Compile**: Add appropriate SIMD compiler flags

## Examples

See the following example directories:
- `example/vectorAddSIMD/`: Basic vector addition with SIMD
- `example/simdExamples/`: Advanced SIMD examples including dot product and matrix multiplication

## Future Extensions

Planned improvements include:
- Support for more SIMD operations (min, max, comparison, etc.)
- Better memory alignment utilities
- SYCL backend integration
- Performance benchmarking tools
