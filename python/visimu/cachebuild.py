# Author: Paulo R. A. Gilz
"""Building a ``.aoec`` sidecar in a process of its own.

Why out of process at all. A sidecar is a rewrite of a file's samples in the
order this program reads them, so writing one for a 2.9 GB source means reading
2.9 GB and writing about as much. Done inside the application that is minutes
during which the core session is held exclusively -- the C ABI takes the write
lock for the whole of ``aoe_build_cache`` -- so every panel and every plot that
wants a channel waits behind it, and the window is disabled for the duration.

A separate process shares no session, so there is nothing to wait for: the
application stays usable throughout, and the next open of that file finds the
sidecar and takes the cached path.

The rules this obeys, in order of importance:

1. It never writes to the source. It opens it read-only; the only file it
   creates is the sidecar.
2. It never stops anyone else writing to the source. HDF5 takes a lock on a
   file it opens, even read-only, unless told not to, and the memory mapping
   the core uses for .aoe files is opened with ``FILE_SHARE_WRITE``.
3. If the source changes while it works, it abandons the sidecar and removes
   it. A sidecar mixing a file's before and after is the one failure that would
   show numbers which were never in the file together.
4. It fails quietly. A sidecar is an optimisation nobody asked for; the file is
   open and working in the application without it.
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Optional, Sequence

from .logging_setup import get_logger

_log = get_logger("cachebuild")

# The flag, in one place, so the application and the subprocess cannot disagree
# about its spelling.
FLAG = "--build-cache"


def _fingerprint(path: str):
    """``(size, mtime)`` of the source, or None when it cannot be read.

    Deliberately not the core's content hash. This only has to notice a change
    across our own run, and the core applies its own strict three-part check
    around the write itself; two numbers that cost nothing are the right tool
    for "did this move while we worked".
    """
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (stat.st_size, int(stat.st_mtime))


def build_one(path: str) -> bool:
    """Read one file and write its sidecar. True if one was written."""
    # Imported for its import side effect as much as its use: visimu.hdf5
    # sets HDF5_USE_FILE_LOCKING before h5py -- and therefore before the HDF5
    # library -- is loaded, which is what keeps rule 2 above true. Importing it
    # first means that holds however the rest of this function goes.
    from . import hdf5                                            # noqa: F401
    from .session import CoreError, Session

    before = _fingerprint(path)
    if before is None:
        _log.warning("%s: cannot be read", path)
        return False

    session = None
    try:
        session = Session()
        handle = session.add_file(path)

        # A pushed file -- HDF5 -- holds no samples until something asks for
        # them, and the core writes a sidecar from resident columns only. A
        # parsed file needs none of this: the core reads it itself, one
        # parameter at a time.
        #
        # This is the one step here that can want the whole file in memory at
        # once, and it is the reason this runs in a process of its own rather
        # than merely on a thread: the address space it fills is not the
        # application's, and it is handed back in full when this exits.
        if session.is_pushed_index(handle.index):
            if not session.fill_pushed_completely(handle.index):
                _log.warning("%s: could not read every column; no sidecar", path)
                return False

        # The source must still be the file it was when we started reading it.
        # The core checks this too, around its own write; this check covers the
        # read above, which happens before that one begins.
        if _fingerprint(path) != before:
            _log.warning("%s: changed while being read; no sidecar", path)
            return False

        if not session.build_cache(handle.index):
            _log.info("%s: no sidecar written (%s)", path,
                      session.last_error or "nowhere writable")
            return False

        # And once more afterwards. The core deletes a sidecar whose source
        # moved during its own write, so reaching here with a changed source
        # means the change fell between its check and ours -- in which case the
        # sidecar on disk describes a file that no longer exists and has to go.
        # Only the sidecar: the source is never ours to touch.
        if _fingerprint(path) != before:
            _log.warning("%s: changed as the sidecar was written; discarding it",
                         path)
            _discard(session, handle.index, path)
            return False

        _log.info("%s: sidecar written", path)
        return True

    except (CoreError, OSError, ValueError, MemoryError) as exc:
        # MemoryError belongs in that list: filling every column of a very large
        # file is the one thing here that can exhaust memory, and when it does
        # the right outcome is no sidecar, not a crash report about a feature
        # nobody asked for.
        _log.warning("%s: %s", path, exc)
        return False
    finally:
        if session is not None:
            session.close()


def _discard(session, index: int, path: str) -> None:
    """Take the sidecar away again, by the route that also closes it.

    Through the core rather than ``os.remove``: on Windows the file it just
    wrote is mapped by this process, and a mapped file cannot be deleted while
    the mapping is open. ``discard_cache`` drops the mapping first, which is
    the whole reason to go through it.
    """
    try:
        session.discard_cache(index)
        _log.info("%s: sidecar discarded", path)
    except Exception as exc:                        # pragma: no cover - cleanup
        _log.warning("%s: could not discard the sidecar: %s", path, exc)


def build_many(paths: Sequence[str]) -> int:
    """Build sidecars for several files, one at a time. Returns how many.

    Sequentially, deliberately: three large files at once is three times the
    memory and the same one disk, read in an order no read-ahead can follow.
    """
    done = 0
    for path in paths:
        started = time.time()
        if build_one(path):
            done += 1
            _log.info("%s: took %.1fs", path, time.time() - started)
    return done


def main(paths: Sequence[str]) -> int:
    """The subprocess entry point. Exit status is how many files failed.

    Logging is already installed by ``__main__.main`` before it dispatches
    here, so there is nothing to set up.
    """
    paths = [p for p in paths if p]
    if not paths:
        return 0
    _log.info("building sidecars for %d file(s)", len(paths))
    return len(paths) - build_many(paths)


# ---------------------------------------------------------------- the caller

def command_for(paths: Sequence[str]) -> Optional[List[str]]:
    """How to start this module as a subprocess, frozen or from a checkout.

    Frozen, the executable is its own interpreter and re-executes itself with
    the flag. From a checkout it is ``python -m visimu``, which needs the
    package's parent on PYTHONPATH -- see `spawn`.
    """
    if not sys.executable:
        return None
    if getattr(sys, "frozen", False):
        return [sys.executable, FLAG] + list(paths)
    return [sys.executable, "-m", "visimu", FLAG] + list(paths)


def spawn(paths: Sequence[str]):
    """Start a detached sidecar build. Returns the process, or None.

    Detached because it depends on nothing it leaves behind: if the user closes
    the application while a sidecar is being written, finishing it is still the
    right outcome, and a build that is interrupted removes its own half-written
    file rather than leaving one to be found later.

    No console window on Windows. This is a background chore of a windowed
    program, and a console flashing up is how a background chore announces
    itself as a bug.
    """
    import subprocess

    paths = [p for p in paths if p]
    command = command_for(paths)
    if not paths or command is None:
        return None

    env = dict(os.environ)
    frozen = bool(getattr(sys, "frozen", False))
    package_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not frozen:
        # The package's parent, so `-m visimu` resolves in a source
        # checkout. Not when frozen: there, that path is inside the unpacked
        # bundle, which in a one-file build is a temporary directory the parent
        # removes when it exits -- so handing it to a child that outlives the
        # parent points it at a directory that will not be there. The frozen
        # executable needs no import path anyway; it carries its own.
        existing = env.get("PYTHONPATH", "")
        if package_parent not in existing.split(os.pathsep):
            env["PYTHONPATH"] = (package_parent + os.pathsep + existing
                                 if existing else package_parent)
    # Belt and braces: visimu.hdf5 sets this before importing h5py, and
    # setting it in the environment too means it is right even if something
    # else reaches h5py first.
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    kwargs = {"env": env, "stdin": subprocess.DEVNULL,
              "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if not frozen:
        # Same reasoning: a working directory that is about to be deleted is
        # worse than inheriting the parent's. Every path passed in is absolute.
        kwargs["cwd"] = package_parent
    if os.name == "nt":
        kwargs["creationflags"] = (getattr(subprocess, "CREATE_NO_WINDOW", 0)
                                   | getattr(subprocess, "DETACHED_PROCESS", 0))
    else:
        kwargs["start_new_session"] = True

    try:
        process = subprocess.Popen(command, **kwargs)
    except OSError as exc:
        _log.warning("could not start the sidecar build: %s", exc)
        return None
    _log.info("sidecar build started for %d file(s), pid %s",
              len(paths), process.pid)
    return process
