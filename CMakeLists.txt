# Optional CMake build for the aoeViewer core.
#
# CMake is NOT required. The Makefile (Linux/macOS) and build.bat (Windows) are
# the supported paths and need nothing but a compiler, which is what the target
# machine has. This file exists for machines that do have CMake and prefer it,
# and to host the optional pybind11 module.
#
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ctest --test-dir build
#
# Optional pybind11 module (needs network access or a local pybind11):
#   cmake -B build -DAOE_BUILD_PYBIND=ON

cmake_minimum_required(VERSION 3.16)
project(aoecore VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(AOE_SOURCES
    core/src/mapped_file.cpp
    core/src/dataset.cpp
    core/src/formula.cpp
    core/src/iso.cpp
    core/src/decimate.cpp
    core/src/session.cpp
    core/src/c_api.cpp
)

# The shared library the Python front end loads through ctypes.
add_library(aoecore SHARED ${AOE_SOURCES})
target_include_directories(aoecore PUBLIC core/include)
target_compile_definitions(aoecore PRIVATE AOE_BUILD_SHARED)

# Output straight into the build directory so the Python side finds it with no
# extra configuration. The default prefix is already right on both platforms
# (libaoecore.so on Unix, aoecore.dll on Windows), so it is left alone --
# PREFIX does not accept generator expressions and setting one breaks the
# generated makefiles.
set_target_properties(aoecore PROPERTIES
    OUTPUT_NAME "aoecore"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# Multi-config generators (Visual Studio, Xcode) otherwise append Release/ or
# Debug/, which is not where the Python loader looks.
foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER "${config}" config_upper)
    set_target_properties(aoecore PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}"
        LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}"
    )
endforeach()

if(MSVC)
    target_compile_options(aoecore PRIVATE /W4 /O2 /fp:fast)
else()
    target_compile_options(aoecore PRIVATE -Wall -Wextra -O3 -fno-math-errno)
endif()

if(MINGW)
    # Without these the DLL depends on libstdc++-6.dll / libgcc_s_seh-1.dll, and
    # ctypes reports only a bare "DLL load failed" when they are absent.
    target_link_options(aoecore PRIVATE -static -static-libgcc -static-libstdc++)
endif()

find_package(Threads REQUIRED)
target_link_libraries(aoecore PRIVATE Threads::Threads)

# ------------------------------------------------------------------ tests

enable_testing()

add_executable(aoe_tests core/tests/test_core.cpp ${AOE_SOURCES})
target_include_directories(aoe_tests PRIVATE core/include core/tests)
target_link_libraries(aoe_tests PRIVATE Threads::Threads)
add_test(NAME core COMMAND aoe_tests)

add_executable(aoe_bench core/tests/bench_parse.cpp ${AOE_SOURCES})
target_include_directories(aoe_bench PRIVATE core/include core/tests)
target_link_libraries(aoe_bench PRIVATE Threads::Threads)

# -------------------------------------------------- optional pybind11 module
#
# Not the path the application uses -- the ctypes binding over the C ABI is.
# This target exists for callers who prefer a native extension module, and is
# off by default because pybind11 may not be reachable on a locked-down machine.

option(AOE_BUILD_PYBIND "Also build the optional pybind11 module" OFF)

if(AOE_BUILD_PYBIND)
    find_package(pybind11 QUIET CONFIG)
    if(NOT pybind11_FOUND)
        # Try the pip-installed copy before reaching for the network.
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(PYBIND11_CMAKE_DIR)
            list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
            find_package(pybind11 QUIET CONFIG)
        endif()
    endif()

    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.13.6
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    pybind11_add_module(aoecore_py core/bindings/pybind_module.cpp ${AOE_SOURCES})
    target_include_directories(aoecore_py PRIVATE core/include)
    target_link_libraries(aoecore_py PRIVATE Threads::Threads)
endif()

# ---------------------------------------------------------------- install

include(GNUInstallDirs)
install(TARGETS aoecore
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY core/include/aoe DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
