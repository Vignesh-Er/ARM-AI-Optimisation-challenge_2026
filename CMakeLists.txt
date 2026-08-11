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
if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/CMSIS-NN/Source")
    add_subdirectory(third_party)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/paci_core/CMakeLists.txt")
    add_subdirectory(paci_core)
endif()

# arm_nn_requantize (Include/arm_nnsupportfunctions.h) is header-only
# (__STATIC_FORCEINLINE), so this shim needs only the include path, no
# CMSIS-NN sources to link. Exists purely so tools/ref_cmsisnn.py's Python
# reimplementation of the requantizer can be cross-checked against the real
# compiled CMSIS-NN arithmetic (tests/test_requantize_matches_cmsisnn.py) —
# not shipped, not part of paci_core's public build output.
if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/CMSIS-NN/Include/arm_nnsupportfunctions.h")
    add_library(requantize_shim SHARED tools/native/requantize_shim.c)
    target_include_directories(requantize_shim PRIVATE third_party/CMSIS-NN/Include)
    target_compile_definitions(requantize_shim PRIVATE CMSIS_NN_USE_SINGLE_ROUNDING)
endif()

if(PACI_BUILD_BENCH AND EXISTS "${CMAKE_SOURCE_DIR}/bench/CMakeLists.txt")
    add_subdirectory(bench)
endif()

if(PACI_BUILD_TESTS AND EXISTS "${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif()
