#!/bin/sh
# Author: Paulo R. A. Gilz
#
# Build the aoeViewer core on Linux and macOS. The counterpart of build.bat.
#
# Neither CMake nor make is required -- only a C++17 compiler. There is a
# Makefile too, and it is the better choice for repeated builds because it
# tracks header dependencies and recompiles only what changed. This script
# exists for the first build on a machine that has a compiler and not much
# else, and so that a clone has an obvious thing to run on either platform.
# Application 2.2.1; these commands are unchanged.
#
#   ./build.sh            build the core shared library into build/
#   ./build.sh test       build and run the C++ self-tests
#   ./build.sh bench      build and run the parser benchmark
#   ./build.sh pytest     run the Python tests
#   ./build.sh clean      delete build output
#
# The compiler is found on PATH. One that is not can be named with AOE_CXX:
#
#   AOE_CXX=/opt/gcc-13/bin/g++ ./build.sh
#
# Written for /bin/sh rather than bash: this has to run on a minimal container
# or a BSD-ish shell as readily as on a desktop distribution.

set -eu

cd "$(dirname "$0")"

# --------------------------------------------------------------- compiler
#
# Deliberately no path baked in. This file is checked in and shared, and one
# developer's install location is noise to everyone else.
CXX="${AOE_CXX:-}"

if [ -n "$CXX" ]; then
    if ! command -v "$CXX" >/dev/null 2>&1 && [ ! -x "$CXX" ]; then
        echo "[ERROR] AOE_CXX is set to '$CXX' but there is no such compiler." >&2
        exit 1
    fi
else
    for candidate in g++ clang++ c++; do
        if command -v "$candidate" >/dev/null 2>&1; then
            CXX="$candidate"
            break
        fi
    done
fi

if [ -z "$CXX" ]; then
    # echo rather than a cat heredoc: this is the branch that runs on a machine
    # missing its build tools, and echo is a shell builtin that cannot itself
    # be missing.
    echo "[ERROR] No C++ compiler found on PATH." >&2
    echo "" >&2
    echo "        Install one, then run this again:" >&2
    echo "" >&2
    echo "            Debian/Ubuntu   sudo apt install build-essential" >&2
    echo "            Fedora/RHEL     sudo dnf install gcc-c++" >&2
    echo "            Arch            sudo pacman -S gcc" >&2
    echo "            macOS           xcode-select --install" >&2
    echo "" >&2
    echo "        A compiler that is not on PATH can be named instead:" >&2
    echo "" >&2
    echo "            AOE_CXX=/path/to/g++ ./build.sh" >&2
    exit 1
fi

# --------------------------------------------------------------- platform

BUILD=build
CXXFLAGS="-std=c++17 -O3 -DNDEBUG -Wall -Wextra -fno-math-errno -fvisibility=hidden -fPIC"
INCLUDES="-Icore/include"
LDLIBS="-lpthread"

case "$(uname -s)" in
    Darwin)
        LIBNAME="libaoecore.dylib"
        LDFLAGS="-dynamiclib"
        ;;
    *)
        LIBNAME="libaoecore.so"
        LDFLAGS="-shared"
        ;;
esac

SOURCES="core/src/mapped_file.cpp \
         core/src/cache.cpp \
         core/src/dataset.cpp \
         core/src/formula.cpp \
         core/src/iso.cpp \
         core/src/decimate.cpp \
         core/src/session.cpp \
         core/src/c_api.cpp"

PYTHON="${PYTHON:-python3}"

mkdir -p "$BUILD"

# ------------------------------------------------------------------ tasks

case "${1:-all}" in

    clean)
        rm -rf "$BUILD" dist
        echo "Cleaned."
        exit 0
        ;;

    test)
        echo "Building the self-tests with $CXX ..."
        # shellcheck disable=SC2086
        "$CXX" $CXXFLAGS $INCLUDES core/tests/test_core.cpp $SOURCES \
               -o "$BUILD/aoe_tests" $LDLIBS
        exec "$BUILD/aoe_tests"
        ;;

    bench)
        echo "Building the benchmark with $CXX ..."
        # shellcheck disable=SC2086
        "$CXX" $CXXFLAGS $INCLUDES core/tests/bench_parse.cpp $SOURCES \
               -o "$BUILD/aoe_bench" $LDLIBS
        exec "$BUILD/aoe_bench"
        ;;

    pytest)
        # The core has to exist first: the Python package loads it through
        # ctypes at import, and the failure otherwise is a confusing one about
        # a missing library rather than a missing build.
        if [ ! -f "$BUILD/$LIBNAME" ]; then
            echo "The core is not built yet; building it first."
            "$0"
        fi
        exec "$PYTHON" -m unittest discover -s tests -v
        ;;

    all|"")
        ;;

    *)
        echo "[ERROR] Unknown command '$1'." >&2
        echo "        Use: build.sh [ test | bench | pytest | clean ]" >&2
        exit 1
        ;;
esac

# ------------------------------------------------------------------ build
#
# Every source is compiled in a single command, so there are no stale object
# files to disagree about a changed header. Do not split this into separate
# compile steps without adding dependency tracking -- objects that disagree
# about a struct's layout show up as memory corruption at run time, not as a
# build error. (The Makefile does track dependencies, via -MMD -MP, which is
# why it is the better choice once you are editing the core regularly.)

echo "Building $LIBNAME with $CXX ..."
# shellcheck disable=SC2086
"$CXX" $CXXFLAGS -DAOE_BUILD_SHARED $INCLUDES $SOURCES $LDFLAGS \
       -o "$BUILD/$LIBNAME" $LDLIBS

echo
echo "Build successful: $BUILD/$LIBNAME"
echo
echo "Next:"
echo "  PYTHONPATH=python $PYTHON -m aoeviewer    run from source"
echo "  ./build.sh test                           run the self-tests"
echo "  $PYTHON packaging/build_exe.py            build the standalone binary (one folder)"
echo
echo "  Add --onefile for a single binary instead of a folder:"
echo "    $PYTHON packaging/build_exe.py --onefile"
echo "  It is one file to copy, but every launch unpacks ~60 MB to a temp"
echo "  directory first, so it starts in seconds rather than instantly."
echo "  Unless you specifically need one file, take the folder."
