# SPDX-License-Identifier: Apache-2.0

set(PACI_CORE_SOURCES
    src/paci_physics.c
    src/paci_ekf.c
    src/paci_ring.c
    src/paci_cascade.c
    src/paci_infer.c
)

add_library(paci_core_static STATIC ${PACI_CORE_SOURCES})
target_include_directories(paci_core_static PUBLIC include)
set_target_properties(paci_core_static PROPERTIES OUTPUT_NAME paci_core)

# Shared object/DLL: this is what tests/*.py load via ctypes, per G1 — the
# Python simulation path calls into the C core instead of maintaining a
# second EKF implementation, so host and target share exactly one
# implementation of the physics/EKF/ring logic.
add_library(paci_core_shared SHARED ${PACI_CORE_SOURCES})
target_include_directories(paci_core_shared PUBLIC include)
set_target_properties(paci_core_shared PROPERTIES OUTPUT_NAME paci_core)

if(NOT WIN32)
    # mingw/MSVC provide libm functions as part of the C runtime; only
    # POSIX-style toolchains (the aarch64 Linux target) need an explicit -lm.
    target_link_libraries(paci_core_static PUBLIC m)
    target_link_libraries(paci_core_shared PUBLIC m)
endif()

# paci_infer.c calls real CMSIS-NN kernels (Task 2.4) — link both targets
# against the vendored library (third_party/CMakeLists.txt) when it's
# present. Falls back to compiling without Tier-1/2 inference (paci_infer.c
# itself guards on __has_include("tier2_weights.h") etc.) if the submodule
# isn't checked out.
if(TARGET cmsisnn)
    target_link_libraries(paci_core_static PUBLIC cmsisnn)
    target_link_libraries(paci_core_shared PUBLIC cmsisnn)
    # arm_nn_requantize is __STATIC_FORCEINLINE (header-only, gated behind
    # this macro — see Include/arm_nnsupportfunctions.h) — paci_infer.c
    # calls it directly for the hand-written avgpool step, so this
    # translation unit needs the same macro the cmsisnn target itself
    # compiles with (third_party/CMakeLists.txt), not just the link.
    target_compile_definitions(paci_core_static PRIVATE CMSIS_NN_USE_SINGLE_ROUNDING)
    target_compile_definitions(paci_core_shared PRIVATE CMSIS_NN_USE_SINGLE_ROUNDING)

    # G6 regression: a deliberately undersized scratch buffer, built as its
    # own shared library so tests/test_infer_t2.py can load it and confirm
    # paci_infer_t2_s8 actually returns PACI_E_BUFSIZE rather than
    # overrunning the buffer. 16 bytes is smaller than any single layer's
    # real CMSIS-NN scratch requirement for this architecture.
    add_library(paci_core_tiny_scratch SHARED ${PACI_CORE_SOURCES})
    target_include_directories(paci_core_tiny_scratch PUBLIC include)
    target_link_libraries(paci_core_tiny_scratch PUBLIC cmsisnn)
    target_compile_definitions(paci_core_tiny_scratch PRIVATE CMSIS_NN_USE_SINGLE_ROUNDING PACI_SCRATCH_BYTES=16)
    if(NOT WIN32)
        target_link_libraries(paci_core_tiny_scratch PUBLIC m)
    endif()
endif()
