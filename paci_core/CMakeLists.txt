# SPDX-License-Identifier: Apache-2.0

set(PACI_CORE_SOURCES
    src/paci_physics.c
    src/paci_ekf.c
    src/paci_ring.c
    src/paci_cascade.c
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
