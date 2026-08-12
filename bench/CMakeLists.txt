# SPDX-License-Identifier: Apache-2.0

set(BENCH_SOURCES
    bench_main.c
    bench_timing.c
    bench_json.c
)

add_executable(bench_main ${BENCH_SOURCES})
target_include_directories(bench_main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/paci_core/include
)
target_link_libraries(bench_main PRIVATE paci_core_static)

if(TARGET cmsisnn)
    target_link_libraries(bench_main PRIVATE cmsisnn)
endif()

if(NOT WIN32)
    target_compile_definitions(bench_main PRIVATE _GNU_SOURCE)
    target_link_libraries(bench_main PRIVATE m pthread)
endif()

# STEP B (GATE 3.2) - Footprint differencing build variants
# Built with -ffunction-sections -fdata-sections and linked with -Wl,--gc-sections

set(GC_SECTIONS_FLAGS "-ffunction-sections" "-fdata-sections")
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(GC_LINK_FLAGS "-Wl,--gc-sections")
else()
    set(GC_LINK_FLAGS "")
endif()

set(PACI_CORE_BASE_SOURCES
    ${CMAKE_SOURCE_DIR}/paci_core/src/paci_physics.c
    ${CMAKE_SOURCE_DIR}/paci_core/src/paci_ekf.c
    ${CMAKE_SOURCE_DIR}/paci_core/src/paci_ring.c
    ${CMAKE_SOURCE_DIR}/paci_core/src/paci_cascade.c
    ${CMAKE_SOURCE_DIR}/paci_core/src/paci_infer.c
)

# 1. core_only: Physics + EKF + Ring + Cascade, no inference engine (neither tier1 nor tier2 weights in include path)
add_executable(core_only
    footprint/main_core_only.c
    ${PACI_CORE_BASE_SOURCES}
)
target_include_directories(core_only PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/footprint/inc_core_only)
target_compile_options(core_only PRIVATE ${GC_SECTIONS_FLAGS})
if(GC_LINK_FLAGS)
    target_link_options(core_only PRIVATE ${GC_LINK_FLAGS})
endif()
if(TARGET cmsisnn)
    target_link_libraries(core_only PRIVATE cmsisnn)
    target_compile_definitions(core_only PRIVATE CMSIS_NN_USE_SINGLE_ROUNDING)
endif()
if(NOT WIN32)
    target_link_libraries(core_only PRIVATE m)
endif()

# 2. core+tier1: Core + Tier 1 INT4 inference engine (tier1_weights.h in include path, no tier2_weights.h)
add_executable(core+tier1
    footprint/main_core_tier1.c
    ${PACI_CORE_BASE_SOURCES}
)
target_include_directories(core+tier1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/footprint/inc_tier1)
target_compile_options(core+tier1 PRIVATE ${GC_SECTIONS_FLAGS})
if(GC_LINK_FLAGS)
    target_link_options(core+tier1 PRIVATE ${GC_LINK_FLAGS})
endif()
if(TARGET cmsisnn)
    target_link_libraries(core+tier1 PRIVATE cmsisnn)
    target_compile_definitions(core+tier1 PRIVATE CMSIS_NN_USE_SINGLE_ROUNDING)
endif()
if(NOT WIN32)
    target_link_libraries(core+tier1 PRIVATE m)
endif()

# 3. core+tier2: Core + Tier 1 INT4 + Tier 2 INT8 inference engines (both weight headers present)
add_executable(core+tier2
    footprint/main_core_tier2.c
    ${PACI_CORE_BASE_SOURCES}
)
target_include_directories(core+tier2 PRIVATE ${CMAKE_SOURCE_DIR}/paci_core/include)
target_compile_options(core+tier2 PRIVATE ${GC_SECTIONS_FLAGS})
if(GC_LINK_FLAGS)
    target_link_options(core+tier2 PRIVATE ${GC_LINK_FLAGS})
endif()
if(TARGET cmsisnn)
    target_link_libraries(core+tier2 PRIVATE cmsisnn)
    target_compile_definitions(core+tier2 PRIVATE CMSIS_NN_USE_SINGLE_ROUNDING)
endif()
if(NOT WIN32)
    target_link_libraries(core+tier2 PRIVATE m)
endif()

# 4. full: Core + Tier 1 + Tier 2 + bench harness
add_executable(full
    ${BENCH_SOURCES}
    ${PACI_CORE_BASE_SOURCES}
)
target_include_directories(full PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/paci_core/include
)
target_compile_options(full PRIVATE ${GC_SECTIONS_FLAGS})
if(GC_LINK_FLAGS)
    target_link_options(full PRIVATE ${GC_LINK_FLAGS})
endif()
if(TARGET cmsisnn)
    target_link_libraries(full PRIVATE cmsisnn)
    target_compile_definitions(full PRIVATE CMSIS_NN_USE_SINGLE_ROUNDING)
endif()
if(NOT WIN32)
    target_compile_definitions(full PRIVATE _GNU_SOURCE)
    target_link_libraries(full PRIVATE m pthread)
endif()
