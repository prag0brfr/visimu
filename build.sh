#!/bin/sh
# Author: Paulo R. A. Gilz
#
# Build the visimu core on Linux and macOS. The counterpart of build.bat.
#
# Neither CMake nor make is required -- only a C++17 compiler. There is a
# Makefile too, and it is the better choice for repeated builds because it
# tracks header dependencies and recompiles only what changed. This script
# exists for the first build on a machine that has a compiler and not much
# else, and so that a clone has an obvious thing to run on either platform.
# Application 2.4.0; these commands are unchanged.
#
#   ./build.sh            build the core AND the standalone single-file binary
#   ./build.sh exe-folder  the same, as a folder instead of one file
#   ./build.sh core        build only the core shared library into build/
#   ./build.sh test        build and run the C++ self-tests
#   ./build.sh bench       build and run the parser benchmark
#   ./build.sh pytest      run the Python tests
#   ./build.sh clean       delete build output
#
# With no argument this does the whole job, because that is the job: the
# executable embeds the Python front end but loads the core library, so the
# core has to be built first and building one without the other ships
# yesterday's core inside today's bundle. `core` is there for the times you
# only want the library -- to run the C++ tests against, say -- since
# packaging takes a minute against a couple of seconds for the core.
#
# Anything after the command is passed to packaging/build_exe.py, so
# `./build.sh exe --clean` works.
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

# Whether this run packages the executable, and how. On by default -- see the
# commands below -- and anything left in "$@" after the command is handed to
# build_exe.py.
BUILD_EXE=1
EXE_ARGS="--onefile"

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
            # `core`, not a bare run: the tests need the library, not a
            # packaged executable, and packaging one would take a minute.
            "$0" core
        fi
        exec "$PYTHON" -m unittest discover -s tests -v
        ;;

    exe|onefile|--onefile)
        # The default, named explicitly.
        shift || true
        ;;

    exe-folder|exedir|folder)
        EXE_ARGS=""
        shift || true
        ;;

    core|lib|dll|so)
        BUILD_EXE=0
        shift || true
        ;;

    -*)
        # A flag and no command word, now that the default is the whole job:
        # "./build.sh --clean" means build everything and hand --clean to the
        # packaging step. Not shifted, so it travels in "$@" below.
        ;;

    all|"")
        ;;

    *)
        echo "[ERROR] Unknown command '$1'." >&2
        echo "        Use: build.sh [ exe-folder | core | test | bench |" >&2
        echo "                       pytest | clean ]" >&2
        echo "        With no argument it builds the core and the" >&2
        echo "        single-file executable." >&2
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

# ------------------------------------------------------------- packaging
#
# Only when asked, because this is the slow half: PyInstaller analyses every
# import and writes tens of megabytes, which is a minute or more against a
# couple of seconds for the core. Someone building the core to run the tests
# should not pay for it.
#
# No `exec`: the core build above has to have happened first, and its exit
# status is already checked by `set -e`.
if [ "$BUILD_EXE" -eq 1 ]; then
    echo
    echo "Packaging the executable with $PYTHON ..."
    # shellcheck disable=SC2086
    "$PYTHON" packaging/build_exe.py $EXE_ARGS "$@"
    echo
    echo "Core and executable both built."
    exit 0
fi

# Only the core-only path reaches here; the packaging one exits above.
echo
echo "The core only, as asked. Next:"
echo "  PYTHONPATH=python $PYTHON -m visimu    run from source"
echo "  ./build.sh test                           run the self-tests"
echo "  ./build.sh                                core + the standalone binary"
echo "  ./build.sh exe-folder                     the same, as a folder"
echo
echo "  One file is one thing to copy, but every launch unpacks ~60 MB to a"
echo "  temp directory first, so it starts in seconds rather than instantly."
echo "  Where that matters, take the folder."
