// SPDX-License-Identifier: Apache-2.0
// Tiny shim exposing CMSIS-NN's real arm_nn_requantize as a shared-library
// symbol, so tools/ref_cmsisnn.py's Python reimplementation can be checked
// against the actual compiled CMSIS-NN arithmetic (not just against itself)
// before anything is built on top of it as Oracle B for Tier-1 (Task 2.3).
// CMSIS_NN_USE_SINGLE_ROUNDING is provided by the CMake target
// (CMakeLists.txt) so both this shim and the eventual real paci_infer.c
// build define it identically, rather than each hardcoding their own copy.
#include "arm_nnsupportfunctions.h"

#if defined(_WIN32)
#define PACI_EXPORT __declspec(dllexport)
#else
#define PACI_EXPORT
#endif

PACI_EXPORT int32_t paci_test_requantize(int32_t val, int32_t multiplier, int32_t shift) {
    return arm_nn_requantize(val, multiplier, shift);
}
