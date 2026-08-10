# SPDX-License-Identifier: Apache-2.0

# The regression tests (D2/D3/D4/G1/G13, currently) drive paci_core over
# ctypes rather than through a separate C test framework, so `ctest` here is
# a thin wrapper around pytest rather than a native test binary. This keeps
# a single test implementation instead of duplicating C-level unit tests
# that would just re-check the same ctypes-reachable behaviour.
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# ctest does not build targets before running tests, only before "cmake
# --build build" has already produced libpaci_core.{dll,so}; run the normal
# build step first (CI does: cmake --build build, then ctest).
add_test(
    NAME paci_core_pytest
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/tests -v
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
