/* Copyright 2022  Andrea Bocci, Mehmet Yusufoglu, René Widera, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Implementation details
#include "alpaka/core/Sycl.hpp"

//! ALPAKA_DEVICE_THROW either aborts(terminating the program and creating a core dump) or throws std::runtime_error
//! depending on the Acc. The std::runtime_error exception can be catched in the catch block.
//!
//! CUDA __trap function triggers std::runtime_error but can be catched during wait not exec.
//!
//! The OpenMP specification mandates that exceptions thrown by some thread must be handled by the same thread.
//! Therefore std::runtime_error thrown by ALPAKA_DEVICE_THROW aborts the the program for OpenMP backends. If needed
//! the SIGABRT signal can be catched by signal handler.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)
#    define ALPAKA_DEVICE_THROW(MSG)                                                                                  \
        {                                                                                                             \
            printf(                                                                                                   \
                "alpaka encountered a user-defined error condition while running on the CUDA back-end:\n%s",          \
                (MSG));                                                                                               \
            __trap();                                                                                                 \
        }
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__)
#    define ALPAKA_DEVICE_THROW(MSG)                                                                                  \
        {                                                                                                             \
            printf(                                                                                                   \
                "alpaka encountered a user-defined error condition while running on the HIP back-end:\n%s",           \
                (MSG));                                                                                               \
            assert(false);                                                                                            \
        }
#elif defined(ALPAKA_ACC_SYCL_ENABLED) && defined(__SYCL_DEVICE_ONLY__)
#    define ALPAKA_DEVICE_THROW(MSG)                                                                                  \
        {                                                                                                             \
            printf(                                                                                                   \
                "alpaka encountered a user-defined error condition while running on the SYCL back-end:\n%s",          \
                (MSG));                                                                                               \
            assert(false);                                                                                            \
        }
#else
#    define ALPAKA_DEVICE_THROW(MSG)                                                                                  \
        {                                                                                                             \
            printf("alpaka encountered a user-defined error condition:\n%s", (MSG));                                  \
            throw std::runtime_error(MSG);                                                                            \
        }
#endif
