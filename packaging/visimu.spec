# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for visimu.

Build with the helper script rather than calling PyInstaller directly:

    python packaging/build_exe.py            # one folder (fast start)
    python packaging/build_exe.py --onefile  # single file (slower start)
    python packaging/build_exe.py --debug    # console build, prints tracebacks

PyInstaller does not cross-compile: run this on Windows for a .exe and on Linux
for an ELF binary. Same spec, same sources, two builds.

Notes on the choices here:

  * The core shared library is added explicitly. PyInstaller cannot see it --
    it is loaded by ctypes at runtime from a path computed at import, not by a
    Python import statement, so nothing in the dependency graph mentions it.
  * NumPy is collected by hand as well as by the hook; see NUMPY below. This is
    the fix for "Importing the numpy C-extensions failed".
  * The Qt exclusions matter more than they look. A default PySide6 bundle
    drags in WebEngine, Quick/QML, Multimedia, 3D and the SQL drivers, which
    together are the majority of the download and none of which this app uses.
  * console=False on Windows only, unless VISIMU_CONSOLE=1. On Linux a
    console build keeps stderr visible, which is where a missing-library
    message would appear.

Environment switches, all set by build_exe.py:

    VISIMU_ONEFILE=1          single-file build
    VISIMU_CONSOLE=1          keep a console window (Windows) for diagnosis
    VISIMU_RUNTIME_TMPDIR=... where --onefile unpacks itself at run time
"""

import glob
import os
import sys

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

block_cipher = None

PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(SPEC)), ".."))
BUILD_DIR = os.path.join(PROJECT, "build")

if sys.platform == "win32":
    CORE_NAME = "aoecore.dll"
elif sys.platform == "darwin":
    CORE_NAME = "libaoecore.dylib"
else:
    CORE_NAME = "libaoecore.so"

CORE_PATH = os.path.join(BUILD_DIR, CORE_NAME)
if not os.path.isfile(CORE_PATH):
    raise SystemExit(
        "The core library is missing: {0}\n"
        "Build it first with 'make' (Linux) or 'build.bat' (Windows).".format(CORE_PATH)
    )

ONEFILE = os.environ.get("VISIMU_ONEFILE", "0") == "1"
CONSOLE = os.environ.get("VISIMU_CONSOLE", "0") == "1"
RUNTIME_TMPDIR = os.environ.get("VISIMU_RUNTIME_TMPDIR") or None

# ---------------------------------------------------------------------------
# NumPy
# ---------------------------------------------------------------------------
# "ImportError: Importing the numpy C-extensions failed" is the least helpful
# error message in this entire toolchain, because it is what NumPy reports for
# *any* failure of `from . import _core` -- including a missing pure-Python
# submodule, which has nothing to do with C extensions. That is exactly what
# happened here: NumPy 2.5 with PyInstaller 6.3 collected the compiled
# `_multiarray_umath.pyd` but not all of `numpy._core`'s Python modules, and
# the resulting ModuleNotFoundError was reported as a C-extension problem.
#
# So collect NumPy explicitly instead of trusting whichever hook ships with the
# installed PyInstaller:
#
#   1. Every submodule, because NumPy 2.x renamed `numpy.core` to
#      `numpy._core` and an older hook collects the wrong list.
#   2. The DLLs `_multiarray_umath` links against, which do not live inside the
#      `numpy` package at all -- they sit in a sibling `numpy.libs` directory
#      (Windows/Linux) or `numpy/.dylibs` (macOS), where `collect_dynamic_libs`
#      never looks because it walks the package.
#
# Submodules NumPy only reaches through a lazy `__getattr__` are dropped again
# below; collecting them pulls in setuptools and distutils for 14 MB of code
# that is never executed.
NUMPY_SKIP_PREFIXES = ("numpy.f2py", "numpy.distutils")

NUMPY_HIDDEN = []
NUMPY_BINARIES = []
NUMPY_DATAS = []
try:
    import numpy as _numpy

    NUMPY_HIDDEN = [
        name for name in collect_submodules("numpy")
        if not name.startswith(NUMPY_SKIP_PREFIXES)
    ]
    NUMPY_BINARIES = collect_dynamic_libs("numpy")
    NUMPY_DATAS = collect_data_files("numpy")

    _numpy_pkg = os.path.dirname(os.path.abspath(_numpy.__file__))
    _site = os.path.dirname(_numpy_pkg)
    # The sibling directory, whatever this platform calls it.
    for _pattern, _dest in (
        (os.path.join(_site, "numpy.libs", "*"), "numpy.libs"),
        (os.path.join(_numpy_pkg, ".dylibs", "*"), "numpy/.dylibs"),
    ):
        for _path in glob.glob(_pattern):
            if os.path.isfile(_path):
                NUMPY_BINARIES.append((_path, _dest))
                # Also next to the executable: Windows resolves a dependent DLL
                # from the loading module's directory and the application
                # directory, not from an arbitrary bundled subfolder.
                if sys.platform == "win32":
                    NUMPY_BINARIES.append((_path, "."))
except Exception as exc:  # pragma: no cover - build-time only
    print("WARNING: could not enumerate numpy explicitly ({0}). "
          "Relying on the PyInstaller hook alone.".format(exc))

# pyqtgraph loads colour maps from .csv/.txt data files at runtime and picks its
# Qt binding by import, so both the data and the binding need naming.
PYQTGRAPH_DATAS = []
try:
    PYQTGRAPH_DATAS = collect_data_files("pyqtgraph")
except Exception:  # pragma: no cover - build-time only
    pass

# Everything Qt that this application never touches. Trimming these takes the
# bundle from roughly 150 MB to roughly 60 MB.
EXCLUDES = [
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel", "PySide6.QtWebSockets",
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQuickWidgets",
    "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic", "PySide6.Qt3DAnimation", "PySide6.Qt3DExtras",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtGraphs",
    "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtDesigner", "PySide6.QtHelp",
    "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtPositioning",
    "PySide6.QtSerialPort", "PySide6.QtSensors", "PySide6.QtRemoteObjects",
    "PySide6.QtScxml", "PySide6.QtSpatialAudio", "PySide6.QtTextToSpeech",
    "PySide6.QtPdf", "PySide6.QtPdfWidgets",
    # Scientific stack that pyqtgraph can optionally use but we do not.
    "scipy", "pandas", "matplotlib", "IPython", "jupyter",
    "tkinter",
    "PyQt5", "PyQt6", "PySide2",
    # Build-time only: reachable from numpy.f2py / numpy.distutils, which the
    # application never calls. Roughly 14 MB.
    "setuptools", "pkg_resources", "numpy.f2py", "numpy.distutils",
]

# "distutils" is deliberately NOT in that list, and must not be put back.
#
# PEP 632 removed distutils from the standard library in Python 3.12, so
# PyInstaller supplies it instead: a pre-safe-import hook aliases
# setuptools._distutils onto the name distutils as soon as anything in the
# graph reaches for it, which numpy's own hook does by way of
# importlib_metadata and packaging.
#
# Listing it as an exclude puts an ExcludedModule node under that name first,
# and the alias then refuses to overwrite it:
#
#   ValueError: Target module "distutils" already imported as
#   ExcludedModule('distutils',)
#
# -- which aborts the build during Analysis, before anything is written. On
# 3.12 the exclusion saves nothing anyway, there being no stdlib distutils to
# leave out; what the comment above is really excluding is setuptools and
# pkg_resources, and those two stay. Everything distutils would have dragged in
# comes through setuptools._distutils, which the setuptools exclude already
# covers.

# Deliberately NOT excluded, despite being tempting:
#   unittest  - numpy imports it through numpy.testing, and excluding it makes
#               the frozen build fail at startup with the thoroughly misleading
#               "Importing the numpy C-extensions failed".
#   test, pydoc_data - same family of risk for a few hundred KB of savings.

# The window icon. Two separate jobs, and only one of them was being done:
#
#   * `icon=` on EXE sets the icon Windows shows for the *file* in Explorer and
#     on the taskbar shortcut. It is embedded as a resource in the binary.
#   * bundling the .ico as data lets the running application call
#     setWindowIcon(), which is what puts it in the title bar and on the
#     running taskbar button. Without it the frozen app shows Qt's default.
#
# The icon is generated (tools/make_icon.py), so it can legitimately be absent
# from a fresh checkout -- but silently producing an unbranded executable is
# not the right response to that. build_exe.py generates it before calling
# PyInstaller; this is the backstop for anyone invoking the spec directly.
ICON = os.path.join(PROJECT, "packaging", "visimu.ico")
if not os.path.isfile(ICON):
    raise SystemExit(
        "The application icon is missing: {0}\n"
        "Generate it first:  python tools/make_icon.py".format(ICON)
    )
icon_arg = ICON

# A version resource, so Explorer's details pane and the file's Properties
# dialog show a name and a version rather than blanks. Written next to the
# build rather than committed, because it has to track __version__.
VERSION_FILE = None
if sys.platform == "win32":
    try:
        sys.path.insert(0, os.path.join(PROJECT, "python"))
        from visimu import __version__ as _app_version

        _parts = [int(p) for p in (_app_version.split(".") + ["0", "0", "0"])[:4]]
        VERSION_FILE = os.path.join(BUILD_DIR, "version_info.txt")
        with open(VERSION_FILE, "w", encoding="utf-8") as _handle:
            _handle.write(
                "VSVersionInfo(\n"
                "  ffi=FixedFileInfo(filevers={0}, prodvers={0}, mask=0x3f,\n"
                "                    flags=0x0, OS=0x40004, fileType=0x1,\n"
                "                    subtype=0x0, date=(0, 0)),\n"
                "  kids=[\n"
                "    StringFileInfo([StringTable('040904B0', [\n"
                "      StringStruct('CompanyName', 'Paulo R. A. Gilz'),\n"
                "      StringStruct('FileDescription', 'visimu'),\n"
                "      StringStruct('FileVersion', '{1}'),\n"
                "      StringStruct('InternalName', 'visimu'),\n"
                "      StringStruct('OriginalFilename', 'visimu.exe'),\n"
                "      StringStruct('ProductName', 'visimu'),\n"
                "      StringStruct('ProductVersion', '{1}'),\n"
                "    ])]),\n"
                "    VarFileInfo([VarStruct('Translation', [1033, 1200])])\n"
                "  ]\n"
                ")\n".format(tuple(_parts), _app_version)
            )
    except Exception as exc:  # pragma: no cover - build-time only
        print("WARNING: no version resource ({0})".format(exc))
        VERSION_FILE = None

# h5py is optional: it is only needed to open HDF5 result files, and it is not
# part of the base install. Naming it unconditionally would make every build
# fail on a machine that does not have it, so it is added only when it is there
# -- a frozen build made without it simply cannot open those files, and says so.
try:
    import h5py as _h5py  # noqa: F401

    HDF5_HIDDEN = ["h5py", "h5py.defs", "h5py.utils", "h5py._proxy",
                   "h5py.h5ac", "visimu.hdf5"]
    print("spec: h5py found, HDF5 support will be included")
except Exception:  # pragma: no cover - build-time only
    HDF5_HIDDEN = []
    print("spec: h5py not installed, the build will not read HDF5 files")

a = Analysis(
    [os.path.join(PROJECT, "python", "visimu", "__main__.py")],
    pathex=[os.path.join(PROJECT, "python")],
    binaries=[(CORE_PATH, ".")] + NUMPY_BINARIES,
    datas=NUMPY_DATAS + PYQTGRAPH_DATAS + [(ICON, ".")],
    # pyqtgraph resolves its Qt binding at import time, so the static analysis
    # misses it; naming it here keeps the frozen build from failing on start.
    hiddenimports=(
        ["visimu", "pyqtgraph", "PySide6.QtWidgets",
         "PySide6.QtCore", "PySide6.QtGui"]
        + NUMPY_HIDDEN
        + HDF5_HIDDEN
    ),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

SHOW_CONSOLE = CONSOLE or sys.platform != "win32"

if ONEFILE:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name="visimu",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,           # UPX-packed binaries trip corporate antivirus
        # Normally None, meaning %TEMP%. Locked-down Windows machines often
        # forbid loading DLLs from %TEMP% (AppLocker), which breaks every
        # extension module in the bundle; pointing this somewhere allowed
        # fixes that without touching the code.
        runtime_tmpdir=RUNTIME_TMPDIR,
        console=SHOW_CONSOLE,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=icon_arg,
        version=VERSION_FILE,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="visimu",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=SHOW_CONSOLE,
        disable_windowed_traceback=False,
        icon=icon_arg,
        version=VERSION_FILE,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        name="visimu",
    )
