# Author: Paulo R. A. Gilz
"""Where the time goes when a large HDF5 file is opened.

    python tools/hdf5_timing.py /path/to/file.hdf5

Times each phase of an open separately and reports peak memory after each, so
that "the app took two minutes" becomes a number against a named step. Needs
nothing but the core library and h5py -- no Qt, no display.

The phases, and what a slow one would mean:

  inspect          walking the tree to work out which datasets are which.
                   Touches every dataset's header. Slow here means the walk is
                   the problem and the fix is to read names rather than objects.

  add_file         inspect again (the session does its own), plus handing the
                   channel and run names to the core. The difference between
                   this and inspect is the push, which should be milliseconds.

  one column       the lazy read: what the app does the first time a curve is
                   drawn. This is the cost you pay per parameter, not per open.

  fill every       what the sidecar builder does before it can write anything.
  column           Watch the memory figure here: it holds the whole file.

  build_cache      writing the sidecar itself.

Only the first three happen while you wait for the window. The last two run in
a separate process in 2.2.1, so they cost you responsiveness rather than a
frozen window.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))


def peak_mb():
    """Peak resident memory of this process, in MB. 0.0 if unavailable.

    The Windows path needs its argument and return types declared. Without
    them ctypes assumes a C int, and GetCurrentProcess()'s pseudo-handle
    (-1) is truncated on a 64-bit build, so the call fails and the figure
    comes back as zero -- which is how this reported 0 MB for every phase
    the first time it was run.
    """
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD),
                            ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.GetCurrentProcess.argtypes = []

            # K32GetProcessMemoryInfo lives in kernel32 on anything modern and
            # avoids depending on psapi.dll being present under that name.
            try:
                query = kernel32.K32GetProcessMemoryInfo
            except AttributeError:
                query = ctypes.WinDLL("psapi", use_last_error=True) \
                    .GetProcessMemoryInfo
            query.restype = wintypes.BOOL
            query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters),
                              wintypes.DWORD]

            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if not query(kernel32.GetCurrentProcess(),
                         ctypes.byref(counters), counters.cb):
                return 0.0
            return counters.PeakWorkingSetSize / (1024.0 * 1024.0)
        except Exception:
            return 0.0
    try:
        import resource

        # Linux reports kilobytes here; macOS reports bytes.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1024.0 if sys.platform != "darwin" else peak / 1048576.0
    except Exception:
        return 0.0


def step(label, function):
    started = time.time()
    result = function()
    elapsed = time.time() - started
    print("  %-26s %8.1f s     peak %7.0f MB" % (label, elapsed, peak_mb()))
    return result, elapsed


def main(path):
    print("file: %s" % path)
    try:
        print("size: %.2f GB" % (os.path.getsize(path) / 1e9))
    except OSError as exc:
        print("cannot stat it: %s" % exc)
        return 1

    from aoeviewer import hdf5

    print("\nphases")
    # Cold: no layout written down, so this is the real cost of working it out.
    cache = hdf5._layout_cache_path(path)
    if cache and os.path.exists(cache):
        os.remove(cache)
    hdf5.release()

    described, inspect_s = step("inspect, from names",
                                lambda: hdf5.inspect(path, use_cache=False))
    layout = described.layout
    print("     %d runs x %d channels x %d samples, read as '%s'"
          % (len(layout.run_keys), len(described.channels),
             layout.samples, layout.rule))

    # And the two cheaper routes, for comparison.
    hdf5.release()
    _thorough, object_s = step(
        "inspect, opening every dataset",
        lambda: hdf5._inspect_by_object(hdf5._open(path), path))
    hdf5.release()
    hdf5.inspect(path)                     # writes the layout down
    hdf5.release()
    _cached, cached_s = step("inspect, from the layout cache",
                             lambda: hdf5.inspect(path))

    from aoeviewer import Session

    session = Session()
    handle, add_s = step("add_file (inspect + push)",
                         lambda: session.add_file(path))

    first = described.channels[0]
    _column, column_s = step("one column, one run",
                             lambda: session.channel(handle.index, 0, first))

    print("\n--- the rest happens in another process in 2.2.1 ---")
    _ok, fill_s = step("fill every column",
                       lambda: session.fill_pushed_completely(handle.index))
    _built, build_s = step("build_cache (write sidecar)",
                           lambda: session.build_cache(handle.index))

    waited = add_s + column_s
    print("\nWorking the layout out:")
    print("  opening every dataset   %7.1f s" % object_s)
    print("  from names              %7.1f s" % inspect_s)
    print("  from the layout cache   %7.1f s" % cached_s)
    print("\nYou wait for %.1f s (add_file + first column); a second open of"
          % waited)
    print("this file pays %.1f s for the layout instead of %.1f s."
          % (cached_s, object_s))
    print("The sidecar costs a further %.1f s and %.0f MB, in the background."
          % (fill_s + build_s, peak_mb()))

    session.close()
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
