# SPDX-License-Identifier: Apache-2.0
# Builds the pinned CMSIS-NN checkout as a static library. Host builds (this
# target) have none of ARM_MATH_DSP/ARM_MATH_MVEI defined, so CMSIS-NN's own
# #ifdef guards select its portable C reference path for every kernel; the
# Cortex-M55 FVP build (Phase 3) defines ARM_MATH_MVEI/ARM_MATH_DSP so the
# same sources instead compile the Helium/DSP-optimised paths. Compiling the
# whole Source/ tree rather than cherry-picking files avoids having to track
# CMSIS-NN's internal support-function dependency graph by hand.
file(GLOB_RECURSE CMSIS_NN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Source/*.c")

add_library(cmsisnn STATIC ${CMSIS_NN_SOURCES})
target_include_directories(cmsisnn PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/CMSIS-NN/Include"
)
target_compile_definitions(cmsisnn PUBLIC CMSIS_NN_USE_SINGLE_ROUNDING)
# CMSIS-NN's reference (non-DSP/MVE) path is plain portable C; disable
# -Wall/-Wextra's stricter warnings-as-errors-adjacent noise from code we
# don't own and can't fix upstream from inside this repo.
target_compile_options(cmsisnn PRIVATE -w)
