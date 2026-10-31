# Author: Paulo R. A. Gilz
"""Pythonic wrapper over the C core.

This is the only module the UI talks to. It hides ctypes entirely and presents
files, simulations and channels as ordinary Python objects, while keeping every
heavy operation on the C++ side.

The one rule worth remembering: arrays returned by :meth:`Session.channel` are
*views* of C++ memory, not copies. They are fast because of that, and they stop
being valid when the owning file is reloaded or removed.
"""

from __future__ import annotations

import ctypes
import threading
from ctypes import byref, c_char_p, c_double, c_float, c_int, c_size_t
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import _core
from . import hdf5
from .logging_setup import get_logger
from ._core import CoreError, as_array, as_pinned_array, decode, encode, lib, pin_of

__all__ = ["Session", "FileHandle", "SimulationHandle", "IsoGroup", "CoreError"]

_log = get_logger("session")


class SimulationHandle:
    """One simulation block inside a file."""

    __slots__ = ("_session", "file_index", "index", "sim_id", "name", "key")

    def __init__(self, session: "Session", file_index: int, index: int):
        self._session = session
        self.file_index = file_index
        self.index = index
        # sim_id is what the file says; name is what the user sees and may have
        # renamed. key is the identity styles are attached to and never changes.
        self.sim_id = decode(lib.aoe_sim_id(session._handle, file_index, index))
        self.name = decode(lib.aoe_sim_name(session._handle, file_index, index))
        self.key = decode(lib.aoe_sim_key(session._handle, file_index, index))

    def rename(self, name: str) -> None:
        lib.aoe_set_sim_name(self._session._handle, self.file_index, self.index,
                             encode(name))
        self.name = name

    def channel(self, name: str) -> np.ndarray:
        return self._session.channel(self.file_index, self.index, name)

    def __repr__(self) -> str:
        return "<Simulation {0!r}>".format(self.sim_id)


class FileHandle:
    """One loaded AOE result file."""

    __slots__ = ("_session", "index")

    def __init__(self, session: "Session", index: int):
        self._session = session
        self.index = index

    @property
    def name(self) -> str:
        return decode(lib.aoe_file_name(self._session._handle, self.index))

    @property
    def path(self) -> str:
        return decode(lib.aoe_file_path(self._session._handle, self.index))

    @property
    def headers(self) -> List[str]:
        handle = self._session._handle
        count = lib.aoe_header_count(handle, self.index)
        return [decode(lib.aoe_header(handle, self.index, i)) for i in range(count)]

    @property
    def units(self) -> List[str]:
        """Units declared in *PARAMETRES, parallel to `headers`.

        Empty strings where the file's dialect carries none, so the two lists
        can always be zipped.
        """
        handle = self._session._handle
        count = lib.aoe_header_count(handle, self.index)
        return [decode(lib.aoe_header_unit(handle, self.index, i)) for i in range(count)]

    @property
    def simulation_count(self) -> int:
        return int(lib.aoe_sim_count(self._session._handle, self.index))

    @property
    def simulations(self) -> List[SimulationHandle]:
        """Handles for every run, cached on the session.

        Building one handle is three calls across the C boundary, so a file of
        a thousand runs is three thousand -- and this is read once per file per
        tile on every redraw, which on a 3x3 grid multiplies by nine. The cache
        is dropped whenever the set of files or their names change.
        """
        return self._session._simulations(self.index)

    def rename(self, name: str) -> None:
        """Change the displayed file name. Identity keys are untouched."""
        lib.aoe_set_file_name(self._session._handle, self.index, encode(name))
        self._session._invalidate_handles()

    def reload(self) -> bool:
        # A pushed file has no parser inside the core to re-run, so the core
        # refuses and the reader that produced it runs again instead. The file
        # keeps its index, and with it every style and selection keyed on it.
        path = self.path
        if self._session.is_pushed(path):
            try:
                self._session._push_hdf5(path, replace=self.index)
            except (CoreError, ValueError, OSError):
                return False
            return True

        ok = bool(lib.aoe_reload_file(self._session._handle, self.index))
        self._session._invalidate_handles()
        return ok

    def __repr__(self) -> str:
        return "<File {0!r} sims={1}>".format(self.name, self.simulation_count)


class IsoGroup:
    """A value interval used to bucket and colour runs by an ISO parameter."""

    __slots__ = ("name", "low", "high", "active")

    def __init__(self, name: str, low: float, high: float, active: bool = True):
        self.name = name
        self.low = low
        self.high = high
        self.active = active

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def __repr__(self) -> str:
        return "<IsoGroup {0!r} [{1}, {2}]>".format(self.name, self.low, self.high)


def _c_floats(values: np.ndarray) -> np.ndarray:
    """A float32 array the core can read as a flat C buffer.

    The core takes a pointer and a count, and knows nothing about NumPy
    strides. Handing it a strided view -- `channel(...)[::step]`, which is
    exactly what the 3D projection builds -- makes it read `count` *contiguous*
    floats from the first element, so it silently answers about the first
    `count` samples of the underlying buffer rather than every step-th one of
    the slice. The bounds of a million-sample trajectory then came from its
    first thousandth, and the projection was scaled to fit a fragment.

    Contiguous input is returned untouched, so the zero-copy path -- which is
    every plotted curve -- is unaffected; only an already-cheap strided slice
    pays for a copy.
    """
    array = np.asarray(values)
    if array.dtype != np.float32 or not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array, dtype=np.float32)
    return array


def _sorted_flag(x: np.ndarray) -> int:
    """1 / 0 / -1 for "x is non-decreasing", as the core's `x_sorted` argument.

    Whether x is ordered decides whether the visible range can be binary
    searched, and getting it wrong on a phase-plane tile silently deletes half
    the trajectory. A channel straight from the core already knows the answer --
    it was worked out while the column was parsed -- so ask it rather than
    rescanning a million samples on every pan. -1 means "work it out", which is
    what anything synthesised in Python gets.
    """
    pin = pin_of(x)
    if pin is not None and pin.count == x.size:
        return 1 if pin.sorted else 0
    return -1


class Session:
    """Owns the loaded files and everything derived from them."""

    def __init__(self) -> None:
        handle = lib.aoe_session_create()
        if not handle:
            raise CoreError("could not create the core session")
        self._handle = handle
        # Scratch buffers for decimation, grown on demand and reused so that
        # panning a plot does not allocate on every frame.
        self._dx: Optional[np.ndarray] = None
        self._dy: Optional[np.ndarray] = None
        # file index -> run handles. See FileHandle.simulations.
        self._sim_cache: Dict[int, List[SimulationHandle]] = {}
        # Paths read outside the core and pushed in as arrays. Reloading one
        # means running its reader again, which only this side can do.
        self._pushed: Dict[str, bool] = {}
        # file index -> what the reader needs to fetch a column of that file:
        # its path, its layout, and the channel order the core was given.
        self._pushed_plans: Dict[int, tuple] = {}
        # One reader at a time. The columns of a pushed file are fetched from
        # whichever thread asked for them -- the UI thread for a tooltip, a
        # worker for an open -- and HDF5 is not reentrant unless the library
        # was built thread-safe, which is not something this program can know.
        self._pull_lock = threading.Lock()
        # (file index, channel) -> one value per run, for pushed files. An ISO
        # filter asks for these on every change, and each answer means opening
        # the source; the set of files is what makes them stale, which is what
        # _invalidate_handles already tracks.
        self._iso_cache: Dict[tuple, np.ndarray] = {}

    # ------------------------------------------------------------ lifetime

    def close(self) -> None:
        hdf5.release()
        if getattr(self, "_handle", None):
            lib.aoe_session_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def last_error(self) -> str:
        return decode(lib.aoe_last_error(self._handle))

    # --------------------------------------------------------------- files

    def _invalidate_handles(self) -> None:
        """Drop cached run handles. Anything that changes which files exist, or
        what they are called, has to call this."""
        self._sim_cache.clear()
        self._iso_cache.clear()
        # And the open HDF5 handles, which belong to files that may just have
        # been removed, reloaded or renumbered. Reopening costs one round trip
        # against the risk of reading through a handle onto a file this
        # session no longer believes in.
        hdf5.release()

    def release_sources(self) -> None:
        """Close the HDF5 files held open for reuse.

        Called when a background task finishes. Reads only happen inside a
        task, so a task is the span over which keeping a handle open pays --
        a page of six parameters opens the file once rather than six times --
        and outside one there is nothing to gain by holding it.

        It is not only tidiness. This program must never be the reason a
        solver cannot write its own output, and the surest way to honour that
        is to hold nothing open while nothing is being read.
        """
        hdf5.release()

    def _simulations(self, file_index: int) -> List[SimulationHandle]:
        cached = self._sim_cache.get(file_index)
        if cached is not None:
            return cached
        count = int(lib.aoe_sim_count(self._handle, int(file_index)))
        handles = [SimulationHandle(self, file_index, i) for i in range(count)]
        self._sim_cache[file_index] = handles
        return handles

    def add_file(self, path: str) -> FileHandle:
        # The core parses the .aoe dialect. Anything else is read here and
        # pushed down as arrays; from the core's side the two are the same
        # file, which is why nothing downstream of this call has to know.
        if hdf5.looks_like_hdf5(path):
            return self._push_hdf5(path, replace=-1)

        index = lib.aoe_add_file(self._handle, encode(path))
        if index < 0:
            raise CoreError(self.last_error or "could not read {0!r}".format(path))
        self._invalidate_handles()
        return FileHandle(self, index)

    def _push_hdf5(self, path: str, replace: int) -> FileHandle:
        """Read an HDF5 file here and hand its columns to the core.

        Every way this can fail -- h5py absent, a tree whose layout cannot be
        worked out, an unreadable file -- surfaces as the same CoreError the
        parsing route raises, because to everything above this it is the same
        event: that file could not be read, and the message says why.
        """
        try:
            described = hdf5.inspect(path)
        except (ValueError, OSError) as exc:
            raise CoreError(str(exc))
        layout = described.layout

        headers = described.channels
        header_array = (c_char_p * len(headers))(*[encode(n) for n in headers])
        unit_array = (c_char_p * len(headers))(
            *[encode(u) for u in described.units])

        index = lib.aoe_begin_arrays(self._handle, encode(path), header_array,
                                     unit_array, len(headers), int(replace))
        if index < 0:
            raise CoreError(self.last_error or "could not open {0!r}".format(path))

        for run_key in layout.run_keys:
            if lib.aoe_add_arrays_sim(self._handle, index, encode(run_key)) < 0:
                raise CoreError(self.last_error or "could not add run {0!r}"
                                .format(run_key))

        # No samples yet. Reading the whole file here is what made a large one
        # take minutes to appear, and most of what it read was never plotted:
        # a page of four tiles asks for four channels out of forty. The headers
        # and the run list are enough to draw the DATA tree and fill the
        # parameter lists, and a column is fetched the first time something
        # asks for it -- see _fill_pushed.
        #
        # end_arrays looks for a sidecar once the runs are known, so a file
        # opened a second time is served from that instead and this reader is
        # never asked for anything.
        lib.aoe_end_arrays(self._handle, index)
        self._pushed[path] = True
        self._pushed_plans[index] = (path, layout, headers)
        self._invalidate_handles()
        return FileHandle(self, index)

    # --------------------------------------------- pushed files, on demand

    def _fill_pushed(self, file_index: int, sim_indices: Sequence[int],
                     names: Sequence[str]) -> None:
        """Fetch the columns of a pushed file that are not resident yet.

        The pull half of the array route, and the reason a large HDF5 file
        opens at once: nothing is read until something asks to draw it.

        The core is asked first -- aoe_preload serves anything a sidecar can
        cover -- and only what is still missing afterwards is read out of the
        source. So the three ways a column can arrive are tried in order of
        cost: already in memory, in the sidecar, or read from the file.
        """
        plan = self._pushed_plans.get(int(file_index))
        if plan is None:
            return
        path, layout, headers = plan

        # Asked of the core rather than worked out here: a computed channel
        # needs whatever its formula depends on, through aliases and through
        # other computed channels, and the formula table is the only thing that
        # knows. "ALT_FT = ALT * 3.28" has to fetch ALT.
        wanted = self.required_columns(file_index, names)
        if not wanted:
            return

        runs = ([int(s) for s in sim_indices] if sim_indices
                else list(range(len(layout.run_keys))))

        missing = [(run, column) for run in runs for column in wanted
                   if not lib.aoe_column_loaded(self._handle, int(file_index),
                                                run, column)]
        if not missing:
            return

        with self._pull_lock:
            # Checked again inside the lock: another thread may have fetched
            # exactly these while this one waited, and reading them twice is
            # the cost this whole mechanism exists to avoid.
            missing = [(run, column) for run, column in missing
                       if not lib.aoe_column_loaded(
                           self._handle, int(file_index), run, column)]
            if not missing:
                return
            try:
                for run, column, values in hdf5.read_selected(
                        path, layout, missing):
                    lib.aoe_set_arrays_column(
                        self._handle, int(file_index), run, column,
                        values.ctypes.data_as(ctypes.POINTER(c_float)),
                        values.size)
            except (ValueError, OSError) as exc:
                # The file has moved or changed under us. Nothing to do but
                # leave the columns empty, which draws as a gap rather than as
                # a wrong number.
                _log.warning("could not read %s: %s", path, exc)

    def required_columns(self, file_index: int,
                         names: Sequence[str]) -> List[int]:
        """Which physical columns of a file the named channels need.

        Follows aliases and formulas, so a computed channel reports the columns
        underneath it rather than itself.
        """
        if not names:
            return []
        name_array = _core.string_array(list(names))
        count = lib.aoe_required_columns(
            self._handle, int(file_index), name_array, len(names), None, 0)
        if count <= 0:
            return []
        buffer = (c_int * count)()
        written = lib.aoe_required_columns(
            self._handle, int(file_index), name_array, len(names), buffer, count)
        return [int(buffer[i]) for i in range(min(written, count))]

    def fill_pushed_completely(self, file_index: int) -> bool:
        """Fetch every column of a pushed file. False if it is not one.

        What a sidecar needs: the core writes one from resident columns, and
        has no reader of its own to fill a gap it finds.
        """
        plan = self._pushed_plans.get(int(file_index))
        if plan is None:
            return False
        _path, _layout, headers = plan
        self._fill_pushed(file_index, [], headers)
        return bool(lib.aoe_all_columns_resident(self._handle, int(file_index)))

    def is_pushed_index(self, file_index: int) -> bool:
        """Whether this file was read outside the core, by index."""
        return int(file_index) in self._pushed_plans

    def is_pushed(self, path: str) -> bool:
        """True when this path was read outside the core and handed to it.

        Such a file has no sidecar and cannot be reloaded by the core, because
        the reader that produced it lives here.
        """
        return bool(self._pushed.get(path))

    def remove_file(self, index: int) -> None:
        handle = self.file(int(index))
        if handle is not None:
            self._pushed.pop(handle.path, None)
        lib.aoe_remove_file(self._handle, int(index))
        # Removing a file shifts every later index, and the plans are keyed by
        # index, so they are rebuilt from the paths rather than renumbered.
        self._reindex_pushed_plans(int(index))
        # Removing a file shifts every later index, so the whole cache goes.
        self._invalidate_handles()

    def _reindex_pushed_plans(self, removed: int) -> None:
        plans = {}
        for index, plan in self._pushed_plans.items():
            if index == removed:
                continue
            plans[index - 1 if index > removed else index] = plan
        self._pushed_plans = plans

    def clear(self) -> None:
        self._pushed.clear()
        self._pushed_plans.clear()
        lib.aoe_clear_files(self._handle)
        self._invalidate_handles()

    @property
    def file_count(self) -> int:
        return int(lib.aoe_file_count(self._handle))

    @property
    def files(self) -> List[FileHandle]:
        return [FileHandle(self, i) for i in range(self.file_count)]

    def file(self, index: int) -> Optional[FileHandle]:
        if 0 <= index < self.file_count:
            return FileHandle(self, index)
        return None

    def stale_files(self) -> List[int]:
        """Indices of files whose contents changed on disk since they were read."""
        count = lib.aoe_stale_files(self._handle, None, 0)
        if count <= 0:
            return []
        buffer = (c_int * count)()
        written = lib.aoe_stale_files(self._handle, buffer, count)
        return [int(buffer[i]) for i in range(min(written, count))]

    # ---------------------------------------------------------- parameters

    @property
    def parameters(self) -> List[str]:
        """Every selectable channel: file headers, computed params and aliases."""
        count = lib.aoe_param_count(self._handle)
        return [decode(lib.aoe_param(self._handle, i)) for i in range(count)]

    @property
    def plottable_parameters(self) -> List[str]:
        """`parameters`, but empty while no file is open.

        Computed channels and aliases outlive the files they were defined
        against -- deliberately, so that closing a study and opening the next
        one does not throw away the formulas and the readable names set up for
        it. What they do not outlive is the ability to plot anything: a
        formula over columns that are not loaded evaluates to nothing, and an
        alias resolves to a channel no file carries.

        So `parameters` is the right list for "what may the user name", and a
        list of names that can carry an axis is a different question.
        Answering both with the same call is what let a tile's x axis be
        re-pointed at an alias while the session held no files at all: closing
        the old study left `parameters` non-empty, so the code meant to leave
        a closed session alone did not recognise one.

        Use this wherever a channel is being *chosen* on the user's behalf.
        """
        return self.parameters if self.file_count else []

    def set_alias(self, alias: str, target: str) -> None:
        lib.aoe_set_alias(self._handle, encode(alias), encode(target))

    def remove_alias(self, alias: str) -> None:
        lib.aoe_remove_alias(self._handle, encode(alias))

    def resolve(self, name: str) -> str:
        return decode(lib.aoe_resolve(self._handle, encode(name)))

    def unit(self, name: str) -> str:
        """The unit of a parameter, or "" if the file did not declare one."""
        return decode(lib.aoe_param_unit(self._handle, encode(name)))

    def label(self, name: str) -> str:
        """A parameter with its unit, as "ALTITUDE [ft]".

        One place decides how a unit is shown, so axis labels, the parameter
        list, statistics and the inspector cannot drift apart.
        """
        unit = self.unit(name)
        return "{0} [{1}]".format(name, unit) if unit else name

    def set_computed(self, name: str, expression: str) -> None:
        """Define a computed channel. Raises CoreError with the parse message."""
        if not lib.aoe_set_computed(self._handle, encode(name), encode(expression)):
            raise CoreError(self.last_error or "invalid expression")
        # The ISO values of a computed channel are computed from it, so a
        # redefinition makes them stale -- and the same name can be given a
        # different formula at any time.
        self._iso_cache.clear()

    def remove_computed(self, name: str) -> List[str]:
        """Remove a computed channel; returns it plus every dependant removed."""
        count = lib.aoe_remove_computed(self._handle, encode(name))
        self._iso_cache.clear()          # see set_computed
        return [decode(lib.aoe_removed_name(self._handle, i)) for i in range(count)]

    def is_computed(self, name: str) -> bool:
        return bool(lib.aoe_is_computed(self._handle, encode(name)))

    def computed_source(self, name: str) -> str:
        return decode(lib.aoe_computed_source(self._handle, encode(name)))

    def validate_formula(self, expression: str) -> Tuple[bool, str]:
        """(ok, message) without evaluating anything. For live editor feedback."""
        ok = bool(lib.aoe_validate_formula(self._handle, encode(expression)))
        return ok, ("" if ok else self.last_error)

    # ---------------------------------------------------------------- data

    def channel(self, file_index: int, sim_index: int, name: str) -> np.ndarray:
        """Samples of one channel as a **read-only view** of the C++ buffer.

        The view owns its samples' lifetime: the core hands back a reference
        alongside the pointer, the array holds it, and the memory cap can no
        longer free a buffer a visible curve is still reading from. Dropping
        every reference to the array releases it.

        Returns an empty array when the channel does not exist for that run,
        which callers should treat as "nothing to plot" rather than an error --
        different files in one study legitimately hold different channels.
        """
        if self._pushed_plans:
            self._fill_pushed(file_index, [int(sim_index)], [name])
        count = c_size_t(0)
        handle = ctypes.c_void_p()
        pointer = lib.aoe_channel_pinned(
            self._handle, int(file_index), int(sim_index), encode(name),
            byref(count), byref(handle)
        )
        return as_pinned_array(pointer, count.value, handle if handle else None)

    def preload(
        self,
        file_index: int,
        sim_indices: Sequence[int],
        names: Sequence[str],
    ) -> None:
        """Load every column the named parameters need, in one parallel pass.

        Call this before drawing a tile. It turns N lazy single-column loads into
        one batched multi-threaded read, which is the difference between a plot
        appearing instantly and appearing a channel at a time.
        """
        if not names:
            return
        sims = _core.int_array(list(sim_indices)) if sim_indices else None
        n_sims = len(sim_indices) if sim_indices else 0
        name_array = _core.string_array(list(names))
        lib.aoe_preload(
            self._handle, int(file_index), sims, n_sims, name_array, len(names)
        )
        # After the core's own pass, not instead of it: a sidecar covers these
        # far faster than the reader can, and what it covered is resident by
        # now and will not be fetched again.
        if self._pushed_plans:
            self._fill_pushed(file_index, sim_indices, names)

    # ------------------------------------------------------------ plotting

    def _scratch(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._dx is None or self._dx.size < size:
            self._dx = np.empty(size, dtype=np.float32)
            self._dy = np.empty(size, dtype=np.float32)
        return self._dx, self._dy

    def decimate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_points: int,
        x_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reduce a series to at most ``2 * max_points`` plot points.

        Min/max bucketing, so narrow spikes survive -- unlike stride sampling,
        which drops whatever falls between the samples it keeps. This is what
        lets the Qt front end stay responsive on million-sample channels: the
        plot widget never sees more points than the tile has pixels.
        """
        count = min(x.size, y.size)
        if count == 0:
            empty = np.empty(0, dtype=np.float32)
            return empty, empty

        # Not routed through _c_floats: this is the innermost path of the whole
        # application and a silent copy here would be a performance bug rather
        # than a correctness one. Every caller passes a contiguous channel view.
        x = _c_floats(x)
        y = _c_floats(y)

        max_points = max(2, int(max_points))
        capacity = 2 * max_points + 4
        out_x, out_y = self._scratch(max(capacity, count if count <= capacity else capacity))

        lo, hi = (0.0, 0.0) if x_range is None else (float(x_range[0]), float(x_range[1]))

        written = lib.aoe_decimate_ex(
            x.ctypes.data_as(ctypes.POINTER(c_float)),
            y.ctypes.data_as(ctypes.POINTER(c_float)),
            c_size_t(count),
            c_float(lo),
            c_float(hi),
            c_size_t(max_points),
            c_int(_sorted_flag(x)),
            out_x.ctypes.data_as(ctypes.POINTER(c_float)),
            out_y.ctypes.data_as(ctypes.POINTER(c_float)),
        )
        written = int(written)
        # Copy out of the reusable scratch so the caller can hold on to it.
        return out_x[:written].copy(), out_y[:written].copy()

    @staticmethod
    def bounds(values: np.ndarray) -> Optional[Tuple[float, float]]:
        """(min, max) ignoring NaN and infinity, or None if there are no finite samples."""
        if values.size == 0:
            return None

        # Autoscaling asks for this once per visible curve on every replot, and
        # answering it by rescanning a million samples per curve was the largest
        # avoidable cost in the draw path. A whole channel already carries its
        # extent, computed in parallel when the column was parsed.
        pin = pin_of(values)
        if pin is not None and pin.count == values.size:
            return pin.bounds()

        low = c_float(0.0)
        high = c_float(0.0)
        values = _c_floats(values)
        ok = lib.aoe_series_bounds(
            values.ctypes.data_as(ctypes.POINTER(c_float)),
            c_size_t(values.size),
            byref(low),
            byref(high),
        )
        return (low.value, high.value) if ok else None

    @staticmethod
    def nearest_index(x: np.ndarray, target: float) -> int:
        """Index of the sample closest to ``target``, for crosshair readout."""
        if x.size == 0:
            return 0
        x = _c_floats(x)
        return int(
            lib.aoe_nearest_index(
                x.ctypes.data_as(ctypes.POINTER(c_float)),
                c_size_t(x.size),
                c_float(float(target)),
            )
        )

    @staticmethod
    def stats(y: np.ndarray, x: Optional[np.ndarray] = None,
              x_range: Optional[Tuple[float, float]] = None) -> Optional[dict]:
        """min/max/mean/stddev/first/last over the samples in view.

        Computed in C++ because the useful version is "statistics of what I am
        currently looking at", which is recomputed on every pan and zoom of
        every visible tile.

        The returned dict carries ``ranged``: True when the x window was
        actually applied. A run whose x channel is shorter than its y channel
        cannot be windowed -- the core is given no x pointer at all -- and the
        figures are then for the whole channel. Reporting that lets the
        inspector say so rather than captioning whole-channel statistics with
        an x range they do not respect.
        """
        if y.size == 0:
            return None
        y = _c_floats(y)
        if x is not None:
            x = _c_floats(x)
        low, high = (0.0, 0.0) if x_range is None else (float(x_range[0]), float(x_range[1]))
        buffer = (c_double * 7)()

        usable_x = x is not None and x.size >= y.size
        x_pointer = (
            x.ctypes.data_as(ctypes.POINTER(c_float)) if usable_x else None
        )
        ok = lib.aoe_series_stats_ex(
            x_pointer,
            y.ctypes.data_as(ctypes.POINTER(c_float)),
            c_size_t(min(y.size, x.size) if x is not None else y.size),
            c_float(low),
            c_float(high),
            c_int(_sorted_flag(x) if usable_x else -1),
            buffer,
        )
        if not ok:
            return None
        return {
            "min": buffer[0], "max": buffer[1], "mean": buffer[2],
            "std": buffer[3], "first": buffer[4], "last": buffer[5],
            "count": int(buffer[6]),
            "ranged": bool(usable_x and low < high),
        }

    # -------------------------------------------------------------- memory

    @property
    def resident_bytes(self) -> int:
        """Sample data currently cached by the core, in bytes."""
        return int(lib.aoe_resident_bytes(self._handle))

    def trim_memory(self, keep_bytes: int) -> int:
        """Drop least-recently-used columns until at most `keep_bytes` remain.

        Always safe -- evicted columns are re-read on demand. This is the valve
        that keeps a session over several half-gigabyte files from growing
        without bound.
        """
        return int(lib.aoe_trim_memory(self._handle, c_size_t(max(0, int(keep_bytes)))))

    # --------------------------------------------------------------- cache

    @property
    def cache_enabled(self) -> bool:
        """False when the AOE_CACHE environment variable switches caching off."""
        return bool(lib.aoe_cache_enabled())

    def cache_active(self, file_index: int) -> bool:
        """True when this file is being served from a matching `.aoec` sidecar.

        A sidecar stores the file's numbers the way the viewer reads them --
        one parameter's samples for every run laid out together, already
        float32 -- so opening skips the index scan and loading a column skips
        the parser entirely.
        """
        return bool(lib.aoe_cache_active(self._handle, int(file_index)))

    def cache_path(self, file_index: int) -> str:
        """Where this file's sidecar lives, or "" when it has none."""
        raw = lib.aoe_cache_path(self._handle, int(file_index))
        return raw.decode("utf-8", "replace") if raw else ""

    def build_cache(self, file_index: int) -> bool:
        """Parse the file once and write its sidecar, so later opens are fast.

        This costs a full parse of the file, so it belongs on a worker thread,
        not on the one drawing the window. Memory stays bounded to a single
        parameter across every run however large the file is, so it is safe to
        call on the files this is actually for.

        Returns False, with `last_error` set, if the file cannot be parsed or
        nowhere is writable -- solver output on a read-only share is the usual
        reason, and it is not an error worth showing the user.
        """
        return bool(lib.aoe_build_cache(self._handle, int(file_index)))

    def discard_cache(self, file_index: int) -> bool:
        """Delete this file's sidecar and go back to reading the text."""
        return bool(lib.aoe_discard_cache(self._handle, int(file_index)))

    # ----------------------------------------------------------------- iso

    def iso_values(self, file_index: int, name: str) -> np.ndarray:
        """Value of an ISO parameter for each simulation of a file (NaN if absent)."""
        if int(file_index) in self._pushed_plans:
            return self._pushed_iso_values(int(file_index), name)
        count = lib.aoe_iso_values(self._handle, int(file_index), encode(name), None, 0)
        if count <= 0:
            return np.empty(0, dtype=np.float32)
        buffer = (c_float * count)()
        written = lib.aoe_iso_values(
            self._handle, int(file_index), encode(name), buffer, count
        )
        return np.frombuffer(buffer, dtype=np.float32, count=min(written, count)).copy()

    def _pushed_iso_values(self, file_index: int, name: str) -> np.ndarray:
        """The same answer for a pushed file, read here rather than in the core.

        The core finds an ISO value by reading the first record of a run
        through its index, and a pushed file has no index there -- but the
        reader can read element zero of a dataset, which costs a chunk rather
        than a column. Doing it that way keeps an ISO scan cheap on a file
        whose columns are otherwise fetched lazily.
        """
        plan = self._pushed_plans.get(file_index)
        if plan is None:
            return np.empty(0, dtype=np.float32)
        path, layout, headers = plan

        resolved = self.resolve(name)
        try:
            column = headers.index(resolved)
        except ValueError:
            try:
                column = headers.index(name)
            except ValueError:
                # Not a channel this file stores. A computed one is the case
                # that matters: it is not in the headers and never will be, and
                # answering NaN for every run meant an ISO filter on a computed
                # channel admitted nothing at all -- NaN reads as "this run has
                # no value", so every run was excluded while the ISO tab, which
                # gets its value list from the core, listed the values quite
                # correctly. A filter that lists 1 and then hides everything
                # holding 1 is the shape of that bug.
                for candidate in (resolved, name):
                    if candidate and self.is_computed(candidate):
                        return self._pushed_computed_iso_values(
                            file_index, candidate, len(layout.run_keys))
                return np.full(len(layout.run_keys), np.nan, dtype=np.float32)

        key = (file_index, column)
        cached = self._iso_cache.get(key)
        if cached is not None:
            return cached.copy()

        with self._pull_lock:
            try:
                values = hdf5.read_first_values(path, layout, column)
            except (ValueError, OSError) as exc:
                _log.warning("could not read %s: %s", path, exc)
                values = np.full(len(layout.run_keys), np.nan, dtype=np.float32)
        self._iso_cache[key] = values
        return values.copy()

    def _pushed_computed_iso_values(self, file_index: int, name: str,
                                    count: int) -> np.ndarray:
        """One value per run for a channel this file computes rather than holds.

        Element zero of the computed channel, per run, which is what the core
        answers for a file it parsed itself: a formula over a whole run --
        max(NZ), or a difference between its ends -- is constant across the
        run, so its first sample *is* the run's value.

        It costs more than the stored case, which reads one element of one
        dataset per run: a computed channel has to be evaluated, and that reads
        whatever columns the formula names. There is no cheaper correct answer
        -- the value does not exist anywhere until it is computed -- so it is
        cached like the stored one, and the cache is dropped whenever the files
        or the definitions change.
        """
        key = (file_index, "computed", name)
        cached = self._iso_cache.get(key)
        if cached is not None:
            return cached.copy()

        values = np.full(max(0, int(count)), np.nan, dtype=np.float32)
        for sim_index in range(values.size):
            try:
                column = self.channel(file_index, sim_index, name)
            except CoreError:
                continue
            if column is not None and column.size:
                first = float(column[0])
                # A formula that cannot be evaluated for a run gives nothing
                # rather than a wrong number, and nothing is what NaN means
                # here: that run is not admitted by a filter on this channel.
                if first == first:
                    values[sim_index] = first

        self._iso_cache[key] = values
        return values.copy()

    def iso_distinct(self, name: str) -> np.ndarray:
        """Sorted distinct values of an ISO parameter across every open file."""
        count = lib.aoe_iso_distinct(self._handle, encode(name), None, 0)
        if count <= 0:
            values = np.empty(0, dtype=np.float32)
        else:
            buffer = (c_float * count)()
            written = lib.aoe_iso_distinct(self._handle, encode(name), buffer,
                                           count)
            values = np.frombuffer(buffer, dtype=np.float32,
                                   count=min(written, count)).copy()
        if not self._pushed_plans:
            return values

        # The core's answer covers the files it parsed. A pushed file's values
        # are known here, so the union is assembled here -- otherwise an ISO
        # filter on an HDF5 file would offer no values to group by.
        parts = [values]
        for file_index in sorted(self._pushed_plans):
            parts.append(self._pushed_iso_values(file_index, name))
        merged = np.concatenate([p for p in parts if p.size])
        merged = merged[np.isfinite(merged)]
        if merged.size == 0:
            return np.empty(0, dtype=np.float32)
        return np.unique(merged).astype(np.float32)

    def iso_auto_group(self, values: np.ndarray, group_count: int) -> List[IsoGroup]:
        """Evenly spaced intervals over the observed range of ``values``."""
        values = np.ascontiguousarray(values, dtype=np.float32)
        if values.size == 0 or group_count <= 0:
            return []

        bounds = (c_float * (2 * group_count))()
        n = lib.aoe_iso_auto_group(
            values.ctypes.data_as(ctypes.POINTER(c_float)),
            c_int(values.size),
            c_int(int(group_count)),
            bounds,
            c_int(group_count),
        )
        groups = []
        for i in range(int(n)):
            low = float(bounds[2 * i])
            high = float(bounds[2 * i + 1])
            groups.append(IsoGroup(self.format_value(low) + " - " + self.format_value(high),
                                   low, high))
        return groups

    def iso_group_index(self, value: float, groups: Sequence[IsoGroup]) -> int:
        """Index of the group containing ``value``, or -1."""
        if not groups:
            return -1
        bounds = (c_float * (2 * len(groups)))()
        for i, group in enumerate(groups):
            bounds[2 * i] = group.low
            bounds[2 * i + 1] = group.high
        return int(lib.aoe_iso_group_index(c_float(float(value)), bounds, c_int(len(groups))))

    def format_value(self, value: float) -> str:
        """Format an ISO value the way the lists show it (trailing zeros trimmed)."""
        return decode(lib.aoe_iso_format(self._handle, c_float(float(value))))


VERSION = _core.VERSION
WORKER_COUNT = _core.WORKER_COUNT
