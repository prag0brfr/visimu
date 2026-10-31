@echo off
REM ---------------------------------------------------------------------------
REM Build the visimu core on Windows with MinGW-w64.
REM
REM Application 2.4.0; these commands are unchanged.
REM No cmake required -- only a compiler. g++ on PATH is enough; a portable
REM MinGW-w64 that is not on PATH can be named with the AOE_CXX variable:
REM
REM   set AOE_CXX=C:\tools\mingw64\bin\x86_64-w64-mingw32-g++.exe
REM   build.bat
REM
REM   build.bat             build the core DLL AND the standalone single .exe
REM   build.bat exe-folder  the same, as a folder instead of one file
REM   build.bat core        build only the core DLL
REM   build.bat test        build and run the C++ self-tests
REM   build.bat bench       build and run the parser benchmark
REM   build.bat clean       delete build output
REM
REM With no argument this does the whole job, because that is the job: the
REM executable embeds the Python front end but loads aoecore.dll, so the core
REM has to be built first and building one without the other ships
REM yesterday's core inside today's bundle. "core" is there for the times you
REM only want the DLL -- to run the C++ tests against, say -- since packaging
REM takes a minute against a couple of seconds for the core.
REM
REM Anything after the command is passed on to packaging\build_exe.py, so
REM "build.bat exe --clean" works.
REM
REM Python is found on PATH, or named with AOE_PYTHON:
REM
REM   set AOE_PYTHON=C:\Python312\python.exe
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

REM --- What this run is going to do -------------------------------------------
REM
REM The whole job by default: the core, then the single-file executable.
set BUILD_EXE=1
set EXE_MODE=--onefile

REM A folder instead of one file.
if /I "%1"=="exe-folder" set EXE_MODE=
if /I "%1"=="exedir" set EXE_MODE=
if /I "%1"=="folder" set EXE_MODE=

REM The core on its own, for the times that is all you want.
if /I "%1"=="core" set BUILD_EXE=0
if /I "%1"=="lib" set BUILD_EXE=0
if /I "%1"=="dll" set BUILD_EXE=0

REM A flag and no command word, now that the default is the whole job:
REM "build.bat --clean" means build everything and hand --clean to the
REM packaging step, so the first argument has to travel with the rest.
REM Substring expansion needs a variable of its own; it does not work on %1.
set FIRSTARG=%1
set FIRSTCHAR=
if not "%FIRSTARG%"=="" set FIRSTCHAR=!FIRSTARG:~0,1!
set FORWARD_FIRST=0
if "!FIRSTCHAR!"=="-" set FORWARD_FIRST=1
if /I "%FIRSTARG%"=="--onefile" set FORWARD_FIRST=0

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

if !BUILD_EXE! EQU 1 goto :package

echo.
echo The core only, as asked. Next:
echo   python -m visimu       run from source ^(set PYTHONPATH=python^)
echo   build.bat test            run the self-tests
echo   build.bat                 core + the standalone .exe
echo   build.bat exe-folder      the same, as a folder
echo.
echo   One file is one thing to copy, but every launch unpacks ~60 MB to a
echo   temp directory first, so it starts in seconds rather than instantly.
echo   Where that matters, take the folder.
exit /b 0

REM --- Packaging -------------------------------------------------------------
REM
REM Skipped only when asked, because this is the slow half: PyInstaller
REM analyses every import and writes tens of megabytes, which is a minute or
REM more against a couple of seconds for the core. Someone building the core
REM to run the tests
REM should not pay for it -- which is also why Python is looked for here
REM rather than at the top: "build.bat core" needs no Python at all.
:package

set PY=%AOE_PYTHON%
set PYARGS=

if not "%PY%"=="" (
    if not exist "%PY%" (
        echo [ERROR] AOE_PYTHON is set to "%PY%" but there is nothing there.
        exit /b 1
    )
) else (
    where python.exe >nul 2>nul
    if !ERRORLEVEL! EQU 0 (
        set PY=python.exe
    ) else (
        REM The Windows launcher, which is on PATH on installs where python.exe
        REM is not -- the usual shape of a managed machine.
        where py.exe >nul 2>nul
        if !ERRORLEVEL! EQU 0 (
            set PY=py.exe
            set PYARGS=-3
        ) else (
            echo [ERROR] No Python found on PATH.
            echo.
            echo         The core built fine; only the packaging step needs
            echo         Python. Add it to PATH or name it:
            echo.
            echo             set AOE_PYTHON=C:\Python312\python.exe
            echo             build.bat exe
            exit /b 1
        )
    )
)

echo.
echo Packaging the executable with !PY! ...
REM %2 onwards is passed through, so "build.bat exe --clean" reaches
REM build_exe.py. Eight is more flags than it has.
set EXTRA=%2 %3 %4 %5 %6 %7 %8 %9
if !FORWARD_FIRST! EQU 1 set EXTRA=%*
"!PY!" !PYARGS! packaging\build_exe.py !EXE_MODE! !EXTRA!
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo Executable build FAILED. The core in build\aoecore.dll is fine.
    exit /b 1
)

echo.
echo Core and executable both built.
exit /b 0
