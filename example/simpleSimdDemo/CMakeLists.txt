# Copyright 2025 Your Name
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required(VERSION 3.22)

set(_TARGET_NAME simpleSimdDemo)

alpaka_add_executable(
    ${_TARGET_NAME}
    src/main.cpp)
target_link_libraries(
    ${_TARGET_NAME}
    PRIVATE common)

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "example")

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME})
