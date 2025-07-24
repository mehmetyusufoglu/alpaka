################################################################################
# Copyright 2025 Your Name
# SPDX-License-Identifier: MPL-2.0
################################################################################

cmake_minimum_required(VERSION 3.22)

set(_TARGET_NAME simdExamples)

append_recursive_files_add_to_src_group("src" "src" "cpp" _FILES_SOURCE)

alpaka_add_executable(
    ${_TARGET_NAME}
    ${_FILES_SOURCE})
target_link_libraries(
    ${_TARGET_NAME}
    PUBLIC "alpaka::alpaka")

# Enable C++20 for std::experimental::simd support
target_compile_features(${_TARGET_NAME} PRIVATE cxx_std_20)

# Add compiler-specific flags for SIMD
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${_TARGET_NAME} PRIVATE -march=native)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(${_TARGET_NAME} PRIVATE /arch:AVX2)
endif()
