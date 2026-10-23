# Author: Paulo R. A. Gilz
"""ctypes bindings for the aoeViewer C++ core.

The core is a plain C shared library, so this module needs nothing beyond the
standard library plus NumPy. That is deliberate: on a locked-down machine you
cannot count on being able to install pybind11 or rebuild a Python-ABI-specific
extension, and one ``.dll`` / ``.so`` here works with any CPython 3.8+.

Everything performance-sensitive stays on the C++ side. Column data crosses the
boundary as a pointer that NumPy wraps *without copying*, so plotting a
million-sample channel costs one function call, not a million conversions.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import (
    POINTER,
    c_char_p,
    c_double,
    c_float,
    c_int,
    c_size_t,
    c_void_p,
)
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["lib", "CoreError", "library_path", "as_array", "as_pinned_array", "Pin"]


class CoreError(RuntimeError):
    """Raised when the core reports a failure through aoe_last_error()."""


# --------------------------------------------------------------------------
# Locating the shared library
# --------------------------------------------------------------------------

_LIB_NAMES = {
    "win32": "aoecore.dll",
    "darwin": "libaoecore.dylib",
}


def _library_name() -> str:
    return _LIB_NAMES.get(sys.platform, "libaoecore.so")


def _candidate_dirs() -> List[str]:
    """Directories to search, most specific first.

    Order matters. A PyInstaller bundle puts the library in ``sys._MEIPASS``;
    a source checkout has it in ``build/``. An explicit AOEVIEWER_CORE wins over
    both so a developer can point at a debug build without reinstalling.
    """
    dirs: List[str] = []

    override = os.environ.get("AOEVIEWER_CORE")
    if override:
        dirs.append(override if os.path.isdir(override) else os.path.dirname(override))

    # Frozen with PyInstaller: everything is unpacked beside the executable.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs.append(meipass)
        dirs.append(os.path.join(meipass, "aoeviewer"))
    if getattr(sys, "frozen", False):
        dirs.append(os.path.dirname(sys.executable))

    here = os.path.dirname(os.path.abspath(__file__))
    dirs.append(here)                                            # installed beside the package
    project = os.path.abspath(os.path.join(here, "..", ".."))
    dirs.append(os.path.join(project, "build"))                  # source checkout
    dirs.append(os.getcwd())

    seen = set()
    unique = []
    for d in dirs:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            unique.append(d)
    return unique


def library_path() -> str:
    """Absolute path of the core library, or raise with somewhere to look."""
    override = os.environ.get("AOEVIEWER_CORE")
    if override and os.path.isfile(override):
        return override

    name = _library_name()
    for directory in _candidate_dirs():
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate

    searched = "\n  ".join(_candidate_dirs())
    raise CoreError(
        "could not find the aoeViewer core library ({0}).\n"
        "Build it first:\n"
        "    make            (Linux/macOS)\n"
        "    build.bat       (Windows)\n"
        "or set AOEVIEWER_CORE to its location.\n"
        "Searched:\n  {1}".format(name, searched)
    )


def _load() -> ctypes.CDLL:
    path = library_path()
    directory = os.path.dirname(path)

    # On Windows the core is built with -static-libgcc/-static-libstdc++ so it
    # has no sibling runtime DLLs, but adding its directory costs nothing and
    # rescues a dynamically linked build.
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(directory)
        except (OSError, FileNotFoundError):
            pass

    try:
        return ctypes.CDLL(path)
    except OSError as exc:
        hint = ""
        # On Windows this exact message means the file was found but something
        # it links against was not -- almost always the MinGW runtime DLLs. The
        # message itself never says which, so point at the tool that does.
        if sys.platform == "win32" and "dependencies" in str(exc):
            hint = (
                "\n\nThe library exists but a DLL it depends on is missing.\n"
                "List what it needs with:\n"
                "    objdump -p \"{0}\" | findstr \"DLL Name\"\n"
                "Anything beyond KERNEL32.dll and msvcrt.dll (typically\n"
                "libwinpthread-1.dll, libstdc++-6.dll or libgcc_s_seh-1.dll) means the\n"
                "runtime was linked dynamically. Rebuild with build.bat, which passes\n"
                "-static -static-libgcc -static-libstdc++."
            ).format(path)
        raise CoreError("failed to load {0}: {1}{2}".format(path, exc, hint)) from exc


lib = _load()

# --------------------------------------------------------------------------
# Prototypes
#
# Declaring every signature is not optional. Without argtypes/restype ctypes
# assumes int-sized arguments and returns, which silently truncates pointers on
# 64-bit builds -- the classic way these bindings "work" until they segfault.
# --------------------------------------------------------------------------

_FLOAT_P = POINTER(c_float)
_DOUBLE_P = POINTER(c_double)
_INT_P = POINTER(c_int)
_SIZE_P = POINTER(c_size_t)
_STR_ARRAY = POINTER(c_char_p)

_PROTOTYPES = [
    ("aoe_version", [], c_char_p),
    ("aoe_worker_count", [], c_int),
    ("aoe_session_create", [], c_void_p),
    ("aoe_session_destroy", [c_void_p], None),
    ("aoe_last_error", [c_void_p], c_char_p),

    ("aoe_add_file", [c_void_p, c_char_p], c_int),
    # Pushed files: a format read in Python and handed over as arrays.
    ("aoe_begin_arrays",
     [c_void_p, c_char_p, POINTER(c_char_p), POINTER(c_char_p), c_int, c_int], c_int),
    ("aoe_add_arrays_sim", [c_void_p, c_int, c_char_p], c_int),
    ("aoe_set_arrays_column",
     [c_void_p, c_int, c_int, c_int, _FLOAT_P, c_size_t], c_int),
    ("aoe_end_arrays", [c_void_p, c_int], None),
    ("aoe_column_loaded", [c_void_p, c_int, c_int, c_int], c_int),
    ("aoe_all_columns_resident", [c_void_p, c_int], c_int),
    ("aoe_required_columns",
     [c_void_p, c_int, POINTER(c_char_p), c_int, _INT_P, c_int], c_int),
    ("aoe_remove_file", [c_void_p, c_int], None),
    ("aoe_clear_files", [c_void_p], None),
    ("aoe_file_count", [c_void_p], c_int),
    ("aoe_file_name", [c_void_p, c_int], c_char_p),
    ("aoe_file_path", [c_void_p, c_int], c_char_p),
    ("aoe_stale_files", [c_void_p, _INT_P, c_int], c_int),
    ("aoe_reload_file", [c_void_p, c_int], c_int),

    ("aoe_set_file_name", [c_void_p, c_int, c_char_p], None),
    ("aoe_set_sim_name", [c_void_p, c_int, c_int, c_char_p], None),
    ("aoe_sim_name", [c_void_p, c_int, c_int], c_char_p),
    ("aoe_sim_count", [c_void_p, c_int], c_int),
    ("aoe_sim_id", [c_void_p, c_int, c_int], c_char_p),
    ("aoe_sim_key", [c_void_p, c_int, c_int], c_char_p),
    ("aoe_header_count", [c_void_p, c_int], c_int),
    ("aoe_header", [c_void_p, c_int, c_int], c_char_p),
    ("aoe_header_unit", [c_void_p, c_int, c_int], c_char_p),
    ("aoe_param_unit", [c_void_p, c_char_p], c_char_p),

    ("aoe_param_count", [c_void_p], c_int),
    ("aoe_param", [c_void_p, c_int], c_char_p),
    ("aoe_set_alias", [c_void_p, c_char_p, c_char_p], None),
    ("aoe_remove_alias", [c_void_p, c_char_p], None),
    ("aoe_resolve", [c_void_p, c_char_p], c_char_p),
    ("aoe_set_computed", [c_void_p, c_char_p, c_char_p], c_int),
    ("aoe_remove_computed", [c_void_p, c_char_p], c_int),
    ("aoe_removed_name", [c_void_p, c_int], c_char_p),
    ("aoe_is_computed", [c_void_p, c_char_p], c_int),
    ("aoe_computed_source", [c_void_p, c_char_p], c_char_p),
    ("aoe_validate_formula", [c_void_p, c_char_p], c_int),

    ("aoe_channel", [c_void_p, c_int, c_int, c_char_p, _SIZE_P], _FLOAT_P),
    ("aoe_channel_pinned",
     [c_void_p, c_int, c_int, c_char_p, _SIZE_P, POINTER(c_void_p)], _FLOAT_P),
    ("aoe_pin_release", [c_void_p], None),
    ("aoe_pin_sorted", [c_void_p], c_int),
    ("aoe_pin_bounds", [c_void_p, _FLOAT_P, _FLOAT_P], c_int),
    ("aoe_pin_count", [c_void_p], c_size_t),
    ("aoe_preload", [c_void_p, c_int, _INT_P, c_int, _STR_ARRAY, c_int], None),

    ("aoe_decimate",
     [_FLOAT_P, _FLOAT_P, c_size_t, c_float, c_float, c_size_t, _FLOAT_P, _FLOAT_P],
     c_size_t),
    ("aoe_decimate_ex",
     [_FLOAT_P, _FLOAT_P, c_size_t, c_float, c_float, c_size_t, c_int, _FLOAT_P,
      _FLOAT_P],
     c_size_t),
    ("aoe_series_sorted", [_FLOAT_P, c_size_t], c_int),
    ("aoe_series_bounds", [_FLOAT_P, c_size_t, _FLOAT_P, _FLOAT_P], c_int),
    ("aoe_nearest_index", [_FLOAT_P, c_size_t, c_float], c_size_t),
    ("aoe_series_stats",
     [_FLOAT_P, _FLOAT_P, c_size_t, c_float, c_float, _DOUBLE_P], c_int),
    ("aoe_series_stats_ex",
     [_FLOAT_P, _FLOAT_P, c_size_t, c_float, c_float, c_int, _DOUBLE_P], c_int),

    ("aoe_resident_bytes", [c_void_p], c_size_t),
    ("aoe_trim_memory", [c_void_p, c_size_t], c_size_t),

    ("aoe_cache_enabled", [], c_int),
    ("aoe_cache_active", [c_void_p, c_int], c_int),
    ("aoe_cache_path", [c_void_p, c_int], c_char_p),
    ("aoe_build_cache", [c_void_p, c_int], c_int),
    ("aoe_discard_cache", [c_void_p, c_int], c_int),

    ("aoe_iso_values", [c_void_p, c_int, c_char_p, _FLOAT_P, c_int], c_int),
    ("aoe_iso_distinct", [c_void_p, c_char_p, _FLOAT_P, c_int], c_int),
    ("aoe_iso_auto_group", [_FLOAT_P, c_int, c_int, _FLOAT_P, c_int], c_int),
    ("aoe_iso_group_index", [c_float, _FLOAT_P, c_int], c_int),
    ("aoe_iso_format", [c_void_p, c_float], c_char_p),
]

for _name, _args, _ret in _PROTOTYPES:
    _fn = getattr(lib, _name)
    _fn.argtypes = _args
    _fn.restype = _ret


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def as_array(pointer, count: int) -> np.ndarray:
    """Wrap a ``float*`` from the core as a NumPy array **without copying**.

    The returned array is a live view of the C++ buffer, which stays valid until
    that file is reloaded or removed. It is marked read-only, because writing
    through it would corrupt the core's cache -- callers that need to modify
    values should ``.copy()`` first.
    """
    if not pointer or count <= 0:
        return np.empty(0, dtype=np.float32)

    array = np.ctypeslib.as_array(pointer, shape=(int(count),))
    array.flags.writeable = False
    return array


class Pin:
    """Keeps a channel's samples alive for as long as a view of them exists.

    A plotted curve holds its NumPy array for as long as it is on screen, and
    that array is a window onto a C++ buffer. Without something like this, the
    memory cap could free the buffer a visible curve was reading from -- and it
    evicted least-recently-used columns first, which are precisely the ones the
    tiles are pointing at, so the next mouse move read freed memory.

    The object exposes ``__array_interface__``, which is how NumPy wraps a raw
    address without copying *and* records this object as the array's ``base``.
    That is the whole mechanism: the array keeps the Pin, the Pin keeps the
    C++ block, and when the last view goes the block is released.

    It also carries the summary the core computed while parsing -- extent and
    whether the samples are ordered -- so the draw path can answer questions
    that would otherwise mean rescanning a million samples per curve.
    """

    __slots__ = ("_handle", "count", "__array_interface__", "__weakref__")

    def __init__(self, address: int, count: int, handle):
        self._handle = handle
        self.count = int(count)
        self.__array_interface__ = {
            "data": (int(address), True),   # True: read-only
            "shape": (int(count),),
            "typestr": "<f4",
            "version": 3,
        }

    @property
    def sorted(self) -> bool:
        """True when the samples never decrease -- the precondition of every
        binary search over an x channel. Free: computed when the block was
        parsed."""
        if self._handle is None:
            return False
        return bool(lib.aoe_pin_sorted(self._handle))

    def bounds(self):
        """(min, max) over the finite samples, or None. Also free."""
        if self._handle is None:
            return None
        low = c_float(0.0)
        high = c_float(0.0)
        if not lib.aoe_pin_bounds(self._handle, ctypes.byref(low), ctypes.byref(high)):
            return None
        return (low.value, high.value)

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            lib.aoe_pin_release(handle)

    def __del__(self):
        # Reached during interpreter shutdown as well, when the module globals
        # this needs may already be gone. Never let that raise.
        try:
            self.release()
        except Exception:
            pass


def as_pinned_array(pointer, count: int, handle) -> np.ndarray:
    """A read-only view of ``count`` floats that owns its samples' lifetime."""
    if not pointer or count <= 0:
        if handle is not None:
            lib.aoe_pin_release(handle)
        return np.empty(0, dtype=np.float32)
    address = ctypes.cast(pointer, c_void_p).value
    return np.asarray(Pin(address, count, handle))


def pin_of(array) -> Optional[Pin]:
    """The Pin behind an array, when it has one. None for anything else."""
    base = getattr(array, "base", None)
    return base if isinstance(base, Pin) else None


def encode(text: Optional[str]) -> Optional[bytes]:
    """UTF-8 encode for the C ABI. Paths with accents must survive this."""
    if text is None:
        return None
    return text.encode("utf-8")


def decode(raw: Optional[bytes]) -> str:
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace")


def float_array(values: Sequence[float]):
    """A C float[] from any Python sequence."""
    return (c_float * len(values))(*values)


def int_array(values: Sequence[int]):
    return (c_int * len(values))(*values)


def string_array(values: Sequence[str]):
    """A C ``const char*[]``. Keeps the encoded bytes alive via the array."""
    encoded = [v.encode("utf-8") for v in values]
    array = (c_char_p * len(encoded))(*encoded)
    return array


VERSION = decode(lib.aoe_version())
WORKER_COUNT = int(lib.aoe_worker_count())
