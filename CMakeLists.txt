# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(paci_arm C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(PACI_BUILD_TESTS "Build the C-level regression tests" ON)
option(PACI_BUILD_BENCH "Build the measurement harness (bench_main)" ON)
option(PACI_TRACE_REQUANT "Instrument CMSIS-NN requantization for bias tracing (section 5)" OFF)

# Host builds (aarch64 Linux, or any native dev machine) must never carry
# -ffast-math: it breaks NaN/Inf handling in the EKF and destroys
# host/target bit-exactness (G12). ffp-contract=off keeps FMA contraction
# from silently diverging between host and target.
add_compile_options(-O2 -fno-fast-math -ffp-contract=off -Wall -Wextra)

if(PACI_TRACE_REQUANT)
    add_compile_definitions(PACI_TRACE_REQUANT=1)
endif()

# paci_core/ (Phase 1), CMSIS-NN/CMSIS-DSP submodules (Phase 2), bench/
# (Phase 3) and tests/ (Phase 1+) are added here as each phase lands.
# Guarded with EXISTS so a partial checkout at any phase boundary still
# configures cleanly.
if(EXISTS "${CMAKE_SOURCE_DIR}/paci_core/CMakeLists.txt")
    add_subdirectory(paci_core)
endif()

if(PACI_BUILD_BENCH AND EXISTS "${CMAKE_SOURCE_DIR}/bench/CMakeLists.txt")
    add_subdirectory(bench)
endif()

if(PACI_BUILD_TESTS AND EXISTS "${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif()
