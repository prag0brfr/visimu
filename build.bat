@echo off
REM ---------------------------------------------------------------------------
REM Build the aoeViewer core on Windows with MinGW-w64.
REM
REM Application 2.3.0; these commands are unchanged.
REM No cmake required -- only a compiler. g++ on PATH is enough; a portable
REM MinGW-w64 that is not on PATH can be named with the AOE_CXX variable:
REM
REM   set AOE_CXX=C:\tools\mingw64\bin\x86_64-w64-mingw32-g++.exe
REM   build.bat
REM
REM   build.bat            build the core DLL
REM   build.bat test       build and run the C++ self-tests
REM   build.bat bench      build and run the parser benchmark
REM   build.bat clean      delete build output
REM ---------------------------------------------------------------------------
setlocal enabledelayedexpansion

REM --- Find a compiler -------------------------------------------------------
REM Deliberately no path baked in here. This file is checked in and shared, and
REM one developer's install location is noise to everyone else -- worse, it is a
REM home directory, which carries an account name with it.
set COMPILER=%AOE_CXX%

if not "%COMPILER%"=="" (
    if not exist "%COMPILER%" (
        echo [ERROR] AOE_CXX is set to "%COMPILER%" but there is nothing there.
        exit /b 1
    )
) else (
    where g++.exe >nul 2>nul
    if !ERRORLEVEL! EQU 0 (
        set COMPILER=g++.exe
    ) else (
        where x86_64-w64-mingw32-g++.exe >nul 2>nul
        if !ERRORLEVEL! EQU 0 (
            set COMPILER=x86_64-w64-mingw32-g++.exe
        ) else (
            echo [ERROR] No C++ compiler found on PATH.
            echo.
            echo         Install MinGW-w64, then either add its bin folder to PATH
            echo         or point AOE_CXX at the compiler:
            echo.
            echo             set AOE_CXX=C:\path\to\mingw64\bin\x86_64-w64-mingw32-g++.exe
            echo             build.bat
            exit /b 1
        )
    )
)

set SOURCES=core\src\mapped_file.cpp core\src\cache.cpp core\src\dataset.cpp core\src\formula.cpp core\src\iso.cpp core\src\decimate.cpp core\src\session.cpp core\src\c_api.cpp
REM -D_WIN32_WINNT=0x0A00 targets the Windows 10 API. Windows headers gate every
REM declaration on this, and MinGW-w64's default varies between distributions --
REM which is why this once built on one machine and failed on another with no
REM source change, on errors like "'SRWLOCK' does not name a type" from
REM platform_thread.hpp. 0x0600 (Vista) is the real floor, set by SRWLOCK and
REM SleepConditionVariableSRW; 0x0A00 matches what the GUI already needs through
REM PySide6. core\include\aoe\win32.hpp sets the same value when a build does
REM not, so this flag is insurance rather than the only line of defence.
set CXXFLAGS=-std=c++17 -O3 -DNDEBUG -Wall -Wextra -fno-math-errno -D_WIN32_WINNT=0x0A00
set INCLUDES=-Icore\include

REM -static-libgcc/-static-libstdc++ matter here: without them the DLL needs
REM libstdc++-6.dll and libgcc_s_seh-1.dll next to it, and ctypes reports only
REM a bare "DLL load failed" when they are missing.
set LDFLAGS=-shared -static -static-libgcc -static-libstdc++

if not exist build mkdir build

if "%1"=="clean" (
    if exist build rmdir /s /q build
    if exist dist rmdir /s /q dist
    echo Cleaned.
    exit /b 0
)

if "%1"=="test" (
    echo Building tests...
    "%COMPILER%" %CXXFLAGS% %INCLUDES% core\tests\test_core.cpp %SOURCES% -o build\aoe_tests.exe
    if !ERRORLEVEL! NEQ 0 exit /b 1
    build\aoe_tests.exe
    exit /b !ERRORLEVEL!
)

if "%1"=="bench" (
    echo Building benchmark...
    "%COMPILER%" %CXXFLAGS% %INCLUDES% core\tests\bench_parse.cpp %SOURCES% -o build\aoe_bench.exe
    if !ERRORLEVEL! NEQ 0 exit /b 1
    build\aoe_bench.exe
    exit /b !ERRORLEVEL!
)

REM Every source is compiled in a single command, so there are no stale object
REM files to disagree about a changed header. Do not "optimise" this into
REM separate compile steps without adding dependency tracking.
echo Building aoecore.dll with %COMPILER% ...
"%COMPILER%" %CXXFLAGS% -DAOE_BUILD_SHARED %INCLUDES% %SOURCES% %LDFLAGS% -o build\aoecore.dll
if %ERRORLEVEL% NEQ 0 (
    echo Build FAILED.
    exit /b 1
)

echo.
echo Build successful: build\aoecore.dll
echo.
echo Next:
echo   python -m aoeviewer            run from source ^(set PYTHONPATH=python^)
echo   build.bat test                 run the self-tests
echo   python packaging\build_exe.py  build the standalone executable ^(one folder^)
echo.
echo   Add --onefile for a single .exe instead of a folder:
echo     python packaging\build_exe.py --onefile
echo   It is one file to copy, but every launch unpacks ~60 MB to a temp
echo   directory first, so it starts in seconds rather than instantly.
echo   Unless you specifically need one file, take the folder.
exit /b 0
