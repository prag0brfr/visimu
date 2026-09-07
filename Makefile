# aoeViewer core - portable Makefile
#
# Deliberately does not require cmake: the target machine has only a compiler.
# Works with GNU make on Linux/macOS and with mingw32-make.exe on Windows.
#
#   make              build the shared library into build/
#   make test         build and run the C++ self-tests
#   make bench        build and run the parser benchmark
#   make clean
#
# Override the compiler if it is not on PATH, e.g. on Windows:
#   mingw32-make CXX="C:/.../bin/x86_64-w64-mingw32-g++.exe"

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -DNDEBUG -Wall -Wextra -fno-math-errno -fvisibility=hidden
INCLUDES  = -Icore/include
BUILD     = build

# ---------------------------------------------------------------- platform

ifeq ($(OS),Windows_NT)
    LIBNAME  = aoecore.dll
    # Windows headers gate every declaration on _WIN32_WINNT, and MinGW-w64's
    # default for it varies between distributions -- which is why this once
    # built on one machine and failed on another with no source change.
    # platform_thread.hpp needs at least 0x0600 (Vista) for SRWLOCK and
    # SleepConditionVariableSRW; 0x0A00 is Windows 10, which the GUI already
    # requires through PySide6. core/include/aoe/win32.hpp sets the same value
    # if the build does not, so a build outside this file still works.
    CXXFLAGS += -D_WIN32_WINNT=0x0A00
    LDFLAGS += -shared -static -static-libgcc -static-libstdc++
    LDLIBS   =
    EXE      = .exe
    RM       = del /Q /F
    MKDIR    = if not exist "$(subst /,\,$1)" mkdir "$(subst /,\,$1)"
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        LIBNAME  = libaoecore.dylib
        LDFLAGS += -dynamiclib
    else
        LIBNAME  = libaoecore.so
        LDFLAGS += -shared
    endif
    CXXFLAGS += -fPIC
    LDLIBS    = -lpthread
    EXE       =
    RM        = rm -f
    MKDIR     = mkdir -p $1
endif

# -static-libstdc++ on Windows matters: without it the DLL needs
# libstdc++-6.dll and libgcc_s_seh-1.dll beside it, and ctypes gives a bare
# "DLL load failed" if they are missing.

SOURCES = core/src/mapped_file.cpp \
          core/src/cache.cpp \
          core/src/dataset.cpp \
          core/src/formula.cpp \
          core/src/iso.cpp \
          core/src/decimate.cpp \
          core/src/session.cpp \
          core/src/c_api.cpp

OBJECTS = $(patsubst core/src/%.cpp,$(BUILD)/%.o,$(SOURCES))
TARGET  = $(BUILD)/$(LIBNAME)

PYTHON ?= python3

.PHONY: all clean test bench dirs gendata flightdata icon pytest

all: dirs $(TARGET)

dirs:
	@$(call MKDIR,$(BUILD))

# -MMD -MP makes the compiler emit a .d file listing every header each object
# depends on; including them below means editing a header rebuilds everything
# that includes it. Without this, changing a struct in a header rebuilt only the
# .cpp that changed, and the stale objects disagreed about the struct's layout --
# which shows up as memory corruption at run time, not as a build error.
$(BUILD)/%.o: core/src/%.cpp
	$(CXX) $(CXXFLAGS) -MMD -MP -DAOE_BUILD_SHARED $(INCLUDES) -c $< -o $@

-include $(OBJECTS:.o=.d)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJECTS) $(LDLIBS)

# ------------------------------------------------------------------ tests

TEST_BIN = $(BUILD)/aoe_tests$(EXE)

test: dirs
	$(CXX) $(CXXFLAGS) $(INCLUDES) core/tests/test_core.cpp $(SOURCES) -o $(TEST_BIN) $(LDLIBS)
	$(TEST_BIN)

BENCH_BIN = $(BUILD)/aoe_bench$(EXE)

bench: dirs
	$(CXX) $(CXXFLAGS) $(INCLUDES) core/tests/bench_parse.cpp $(SOURCES) -o $(BENCH_BIN) $(LDLIBS)
	$(BENCH_BIN)

GENDATA_BIN = $(BUILD)/gen_data$(EXE)

gendata: dirs
	$(CXX) $(CXXFLAGS) -o $(GENDATA_BIN) tools/gen_data.cpp

# Flight-dynamics test data. Three 500 MB files of simulated aircraft, which is
# what the performance and ISO behaviour are actually measured against.
flightdata:
	$(PYTHON) tools/gen_flight_data.py data/flight_a.aoe data/flight_b.aoe data/flight_c.aoe --size 500 --runs 25

# The application icon, drawn from tools/make_icon.py rather than committed.
icon:
	$(PYTHON) tools/make_icon.py

pytest:
	$(PYTHON) -m unittest discover -s tests -v

clean:
	-$(RM) $(subst /,$(if $(filter Windows_NT,$(OS)),\,/),$(wildcard $(BUILD)/*))
