# Author: Paulo R. A. Gilz
#
# Reading result files stored as HDF5.
#
# HDF5 is a container, not a layout. Two exporters writing the same simulation
# produce trees that share nothing but the file signature, so there is no such
# thing as "the HDF5 reader" -- there is only a reader for a convention. The
# ones seen so far differ along one axis: where the identity of the *run* is
# written, given that the identity of the *channel* is written somewhere else.
#
#     /RUN_1/ALTITUDE                a group per run, a dataset per channel
#     /ALTITUDE/ALTITUDE_1_2_0       a group per channel, a dataset per run,
#                                    the run spelled into the dataset name
#     /CASE_2/aero/ALTITUDE          the run further up the tree
#     /ALTITUDE                      a single run, nothing to disambiguate
#
# Rather than guess, this module enumerates those conventions, keeps the ones
# that partition the file consistently, and ranks what survives on evidence
# from the tree itself (see `_score`). The chosen layout is logged, because a
# reader that silently picks the wrong axis produces a plausible-looking file
# with the runs and channels transposed.
#
# What is read is exactly what the .aoe path reads: channel names, run names,
# and one float array per (run, channel). Attributes carrying anything else --
# masses, inertias, configuration -- are deliberately ignored; they are not
# channels, and inventing a home for them here would be a different feature.
#
# h5py is an optional dependency. It is not in the base install, so its absence
# is reported as a missing dependency rather than a broken file.

from __future__ import annotations

import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# Set before h5py -- and therefore before the HDF5 library -- is imported
# anywhere, because the library reads it once at initialisation.
#
# HDF5 1.10 and later take a lock on a file it opens, even read-only, and that
# lock stops the program that produces these files from writing to it. This
# tool is a viewer: it must never be the reason a solver cannot write its own
# output. Nothing is read twice without checking, and the sidecar writer
# fingerprints the source before and after and throws away a sidecar whose
# source moved underneath it, so a file being rewritten while we read it is
# detected rather than relied upon not to happen.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np

from .logging_setup import get_logger

_log = get_logger("hdf5")

SUFFIXES = (".h5", ".hdf5", ".he5", ".hdf")

# Attribute names an exporter might use for a channel's unit, in the order they
# are tried. Units are a nicety here: the axis label falls back to the bare
# channel name, which is what the packed .aoe dialect gives too.
_UNIT_KEYS = ("units", "unit", "UNITS", "UNIT", "Units", "Unit")

# Ranking weights. Kept here rather than inline so the ordering they produce can
# be read in one place.
_W_CHANNEL_MATCHES_PARENT = 3.0
_W_RUNS_ORTHOGONAL = 2.0
_W_SEVERAL_RUNS = 0.5

# How many dataset paths the ranking looks at. A file of 50 runs x 1000
# channels holds 50,000 of them, and an exporter's naming convention is uniform
# across a file: a few thousand paths spanning the whole tree settle which
# convention is in force exactly as conclusively as all of them, and ranking
# nine candidate readings over 50,000 paths each was seconds of work to reach
# the same answer.
#
# The ranking is a heuristic and this only makes it a cheaper one. The reading
# it picks is then applied to -- and checked against -- every path in the file,
# so nothing is dropped and no run goes missing because it fell outside a
# sample.
MAX_RANKING_PATHS = 4000


def available() -> bool:
    """True when h5py can be imported."""
    try:
        import h5py  # noqa: F401
    except ImportError:
        return False
    return True


def looks_like_hdf5(path: str) -> bool:
    """True when the file is HDF5, by signature first and suffix second.

    The signature is what decides: these files arrive with whatever extension
    the exporting tool felt like, and a .dat that is really HDF5 should not be
    handed to the text parser.
    """
    try:
        with open(path, "rb") as handle:
            if handle.read(8) == b"\x89HDF\r\n\x1a\n":
                return True
    except OSError:
        pass
    return os.path.splitext(path)[1].lower() in SUFFIXES


# --------------------------------------------------------------------------
# conventions
# --------------------------------------------------------------------------

def _rule_parent_group(path: str) -> Optional[Tuple[str, str]]:
    parent, _, leaf = path.rpartition("/")
    # Without the leading slash, so that this and `_rule_ancestor(1)` produce
    # the same keys for a file whose runs sit at the top level -- they are then
    # the same reading, recognised as such by the deduplication below, instead
    # of two identical candidates competing on the name of the rule.
    return (parent.strip("/") or "/"), leaf


def _rule_leaf_tail(count: int):
    def rule(path: str) -> Optional[Tuple[str, str]]:
        _, _, leaf = path.rpartition("/")
        tokens = leaf.split("_")
        if len(tokens) <= count:
            return None
        return "_".join(tokens[-count:]), "_".join(tokens[:-count])
    return rule


def _rule_ancestor(depth: int):
    def rule(path: str) -> Optional[Tuple[str, str]]:
        parts = path.strip("/").split("/")
        if len(parts) <= depth:
            return None
        return "/".join(parts[:depth]), "/".join(parts[depth:])
    return rule


def _rule_single_run(path: str) -> Optional[Tuple[str, str]]:
    _, _, leaf = path.rpartition("/")
    return "1", leaf


# Ordered only for reproducibility; the score decides.
_RULES = (
    [("parent_group", _rule_parent_group)]
    + [("leaf_tail_%d" % k, _rule_leaf_tail(k)) for k in (1, 2, 3, 4)]
    + [("ancestor_%d" % k, _rule_ancestor(k)) for k in (1, 2, 3)]
    + [("single_run", _rule_single_run)]
)


class Layout(object):
    """One consistent reading of the tree: which datasets form which run."""

    def __init__(self, rule: str, runs: Dict[str, Dict[str, str]],
                 channels: List[str], samples: int):
        self.rule = rule
        self.runs = runs            # run key -> {channel key: dataset path}
        self.channels = channels    # channel keys, in a stable order
        self.samples = samples

    @property
    def run_keys(self) -> List[str]:
        return sorted(self.runs, key=_natural_key)

    def __repr__(self) -> str:
        return "<Layout {0}: {1} runs x {2} channels x {3} samples>".format(
            self.rule, len(self.runs), len(self.channels), self.samples)


def _natural_key(text: str):
    """Sort 'CASE_2' before 'CASE_10', which plain string order does not."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(text))]


def _partition(rule, paths: Sequence[str]):
    """Group paths into {run key: {channel key: path}}, or None if the rule
    cannot read them at all.

    None means the reading is impossible rather than merely poor: a dataset
    name with too few tokens for the rule, or two datasets claiming to be the
    same channel of the same run.
    """
    grouped: Dict[str, Dict[str, str]] = {}
    for path in paths:
        keyed = rule(path)
        if keyed is None:
            return None
        run_key, channel_key = keyed
        if not channel_key:
            return None
        if channel_key in grouped.setdefault(run_key, {}):
            return None
        grouped[run_key][channel_key] = path
    return grouped or None


def _ranking_sample(members: Sequence[str]) -> Sequence[str]:
    """Paths to rank the readings on: a stride across the whole tree.

    A stride rather than the first few thousand. The second signal in `_score`
    is whether the run keys are the same under every parent group, and the
    first few thousand sorted paths all sit under the first few groups -- a
    sample that cannot see that signal at all.
    """
    if len(members) <= MAX_RANKING_PATHS:
        return members
    step = len(members) // MAX_RANKING_PATHS + 1
    return members[::step]


def _score_grouped(grouped, parents) -> float:
    """How much the tree itself supports one reading of it.

    Three signals, in descending order of how much they settle:

    1. The channel key equals the name of the group the dataset sits in.
       `/ALFA/ALFA_1_2_0` under the run-in-the-leaf reading gives the channel
       ALFA inside the group ALFA; under the single-run reading it gives a
       channel called ALFA_1_2_0 inside a group called ALFA, which no exporter
       would write on purpose.
    2. The run keys are the same under every parent group. A run identity is
       orthogonal to the tree: every channel has one dataset per run. A key
       that only appears beneath one group is part of that group's name, not a
       run.
    3. Having found more than one run at all, as a tie-break only -- a file
       really can hold one.

    Deliberately tolerant: this is given a sample of the paths, where a run
    legitimately appears to be missing channels that the rest of the file
    carries. Strictness belongs in `_finalise`, which sees everything.
    """
    total = 0
    matching = 0
    by_parent: Dict[str, set] = {}
    for run_key, channels in grouped.items():
        for channel_key, dataset_path in channels.items():
            total += 1
            parent = parents.get(dataset_path) or \
                (dataset_path.rpartition("/")[0] or "/")
            if channel_key and parent.rpartition("/")[2] == channel_key:
                matching += 1
            by_parent.setdefault(parent, set()).add(run_key)

    score = _W_CHANNEL_MATCHES_PARENT * (matching / float(total or 1))
    if len(by_parent) > 1 and len(set(map(frozenset, by_parent.values()))) == 1:
        score += _W_RUNS_ORTHOGONAL
    if len(grouped) > 1:
        score += _W_SEVERAL_RUNS
    return score


def _finalise(rule_name: str, grouped, length: int) -> Optional[Layout]:
    """Turn a grouping of every path into a Layout, or None if it is ragged.

    Only the channels every run carries can be given a stable column index, and
    a stable index is what lets a tile ask for "channel 7" of any run.
    """
    shared = None
    for channels in grouped.values():
        keys = set(channels)
        shared = keys if shared is None else (shared & keys)
    if not shared:
        return None

    ordered = sorted(shared, key=_natural_key)
    trimmed = {run_key: {key: channels[key] for key in ordered}
               for run_key, channels in grouped.items()}
    return Layout(rule_name, trimmed, ordered, length)


def _layouts(vectors: Dict[str, int]) -> List[Tuple[float, Layout]]:
    """Every consistent reading of a set of 1-D datasets, best first.

    Two stages, because ranking is a heuristic and correctness is not. The
    readings are ranked on a sample of the paths -- cheap, and as conclusive as
    the whole file for a convention that does not vary within it -- and then the
    winner is built over every path and checked. A reading that cannot describe
    the whole file is dropped and the next one tried, so the sample can only
    cost a worse choice, never a wrong one.
    """
    parents = {path: path.rpartition("/")[0] or "/" for path in vectors}

    by_length: Dict[int, List[str]] = {}
    for path, length in vectors.items():
        by_length.setdefault(length, []).append(path)

    # Longest family first: a file usually carries one set of channels plus a
    # short vector or two of metadata, and the channels are the long ones.
    for length, members in sorted(by_length.items(),
                                  key=lambda kv: (-len(kv[1]), -kv[0])):
        if len(members) < 2:
            continue
        members = sorted(members)
        sample = _ranking_sample(members)

        ranked = []
        for rule_name, rule in _RULES:
            grouped = _partition(rule, sample)
            if grouped is None:
                continue
            ranked.append((_score_grouped(grouped, parents), rule_name, rule))
        if not ranked:
            continue
        # A stable sort, so that among readings the evidence cannot separate,
        # the one declared first in _RULES wins. That ordering is deliberate:
        # the plainest convention should be preferred when nothing argues
        # against it.
        ranked.sort(key=lambda item: -item[0])

        found = []
        seen = set()
        for score, rule_name, rule in ranked:
            grouped = _partition(rule, members)
            if grouped is None:
                continue
            layout = _finalise(rule_name, grouped, length)
            if layout is None:
                continue
            signature = (tuple(sorted(layout.runs)), tuple(layout.channels))
            if signature in seen:
                continue
            seen.add(signature)
            found.append((score, layout))

        if found:
            # A shorter family cannot describe the file better than the one
            # holding most of its datasets.
            return found

    return []


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

class Hdf5File(object):
    """The channels and runs of one HDF5 file, ready to be pushed to the core."""

    def __init__(self, path: str, layout: Layout, units: Sequence[str],
                 alternatives: Sequence[str]):
        self.path = path
        self.layout = layout
        self.units = list(units)
        self.alternatives = list(alternatives)

    @property
    def channels(self) -> List[str]:
        return list(self.layout.channels)

    @property
    def runs(self) -> List[str]:
        return self.layout.run_keys


# --------------------------------------------------------------------------
# open handles
# --------------------------------------------------------------------------
#
# Opening an HDF5 file is cheap on a local disk and is not on a network share,
# where it is a round trip or several before a single byte of data is read.
# Every read here used to open the file, read, and close it, so drawing six
# parameters opened the file six times and an ISO scan over fifty opened it
# fifty times.
#
# Handles are therefore kept and reused. What bounds how long they are held is
# `release()`, which the application calls when a background task finishes:
# reads only happen inside a task, so a task is exactly the span over which
# reuse pays, and between tasks the viewer holds nothing open. That matters
# for more than tidiness -- this program must never be the reason a solver
# cannot write its own output, and a handle held while nobody is reading is a
# handle held for no reason at all.

_OPEN: Dict[str, object] = {}


def _require_h5py():
    try:
        import h5py
    except ImportError:
        raise ValueError(
            "reading HDF5 files needs the h5py package, which is not "
            "installed")
    return h5py


def _open(path: str):
    """An open, read-only handle for `path`, reused until release().

    The handle is only reused while it is still a handle onto the same file.
    A reused handle reads the file it was opened on, which is not necessarily
    the file at that path any more -- a solver rewriting its output leaves the
    old one behind the handle, and every read would quietly return yesterday's
    numbers. So the source is stat-ed on the way past: one call against the
    whole point of keeping the handle, and it is what makes keeping it safe.
    """
    stamp = _source_stamp(path)
    cached = _OPEN.get(path)
    if cached is not None:
        handle, opened_at = cached
        try:
            if handle.id.valid and opened_at == stamp:
                return handle
        except Exception:
            pass
        release(path)

    h5py = _require_h5py()
    handle = h5py.File(path, "r")
    _OPEN[path] = (handle, stamp)
    return handle


def release(path: Optional[str] = None) -> None:
    """Close the handle for `path`, or every handle when given nothing.

    Never raises. A handle that cannot be closed is dropped from the table
    regardless: the alternative is to keep handing out a handle that is
    already broken.
    """
    paths = [path] if path is not None else list(_OPEN)
    for key in paths:
        cached = _OPEN.pop(key, None)
        if cached is None:
            continue
        try:
            cached[0].close()
        except Exception:
            pass


def _unit_of(dataset) -> str:
    for key in _UNIT_KEYS:
        if key not in dataset.attrs:
            continue
        value = dataset.attrs[key]
        if isinstance(value, bytes):
            return value.decode("utf-8", "replace").strip()
        if isinstance(value, str):
            return value.strip()
    return ""


# --------------------------------------------------------------------------
# the layout cache
# --------------------------------------------------------------------------
#
# Working out which datasets belong to which run costs one metadata read per
# dataset, and a file of 50 runs by 1000 channels holds 50,000 of them. On a
# local disk that is a few seconds; on a network share it is minutes, because
# the cost is round trips rather than bytes.
#
# It is also the same answer every time. The layout of a file does not change
# unless the file does, so it is worked out once and written down. What is
# stored is small -- the rule, the channel names, their units, and the dataset
# path of every (run, channel) -- and gzipped, which takes a 50,000-entry map
# to a couple of hundred kilobytes read in one sequential pass.
#
# This matters most for the case the sidecar was supposed to have solved. Once
# a sidecar exists, not one sample is read from the HDF5 file -- and yet the
# tree was still walked on every open, because the layout had to be rebuilt
# before the core could be told what the file contains. That walk was the
# entire cost of opening an already-cached file, and all of it was waste.

LAYOUT_SUFFIX = ".aoelay"
LAYOUT_FORMAT = 1


def _layout_cache_path(source: str) -> Optional[str]:
    """Where this file's layout is written, or None if nowhere is writable.

    Beside the source first, so that the layout travels with the file and a
    colleague opening the same share pays nothing either. A read-only share is
    the ordinary case though -- solver output usually is -- so it falls back to
    the same per-user directory the core uses for sidecars, under the same
    AOE_CACHE_DIR override.
    """
    beside = source + LAYOUT_SUFFIX
    directory = os.path.dirname(os.path.abspath(source)) or "."
    if os.access(directory, os.W_OK):
        return beside

    override = os.environ.get("AOE_CACHE_DIR")
    if override:
        fallback = override
    elif os.name == "nt":
        fallback = os.path.join(
            os.environ.get("LOCALAPPDATA")
            or os.path.expanduser("~\\AppData\\Local"), "visimu")
    elif sys.platform == "darwin":
        fallback = os.path.expanduser("~/Library/Caches/visimu")
    else:
        fallback = os.path.join(
            os.environ.get("XDG_CACHE_HOME")
            or os.path.expanduser("~/.cache"), "visimu")

    try:
        if not os.path.isdir(fallback):
            os.makedirs(fallback)
    except OSError:
        return None
    # The full path is hashed into the name, so two files called results.hdf5
    # in different directories do not share one entry.
    import hashlib

    digest = hashlib.sha1(os.path.abspath(source).encode(
        "utf-8", "replace")).hexdigest()[:16]
    return os.path.join(fallback, "%s-%s%s" % (
        os.path.basename(source), digest, LAYOUT_SUFFIX))


def _source_stamp(source: str):
    """(size, mtime) of the source, or None when it cannot be read."""
    try:
        stat = os.stat(source)
    except OSError:
        return None
    return [int(stat.st_size), int(stat.st_mtime)]


def _save_layout(source: str, described: "Hdf5File") -> None:
    """Write the layout down. Silent on failure: it is only ever a shortcut."""
    target = _layout_cache_path(source)
    stamp = _source_stamp(source)
    if target is None or stamp is None:
        return

    document = {
        "format": LAYOUT_FORMAT,
        "source": stamp,
        "rule": described.layout.rule,
        "samples": int(described.layout.samples),
        "channels": list(described.layout.channels),
        "units": list(described.units),
        "alternatives": list(described.alternatives),
        "runs": described.layout.runs,
    }
    # Written beside and renamed, so that a reader never sees half a file and
    # an interrupted write leaves the old one in place.
    temporary = target + ".tmp"
    try:
        import gzip
        import json

        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(document, handle, separators=(",", ":"))
        if os.path.exists(target):
            os.remove(target)
        os.rename(temporary, target)
        _log.info("%s: layout written to %s",
                  os.path.basename(source), os.path.basename(target))
    except (OSError, ValueError, TypeError) as exc:
        _log.info("%s: could not write the layout (%s)",
                  os.path.basename(source), exc)
        try:
            os.remove(temporary)
        except OSError:
            pass


def _load_layout(source: str) -> Optional["Hdf5File"]:
    """The layout written last time, or None to work it out again.

    Three things have to hold: the file has to be the same size and age it was
    when the layout was written, the document has to be one this version
    understands, and -- because neither of those proves the tree is unchanged
    -- one dataset it names has to still be there with the length it claims.
    That last check is a single read, against fifty thousand for the walk it
    replaces, and it is what turns "probably still right" into "checked".
    """
    target = _layout_cache_path(source)
    if target is None or not os.path.exists(target):
        return None
    stamp = _source_stamp(source)
    if stamp is None:
        return None

    try:
        import gzip
        import json

        with gzip.open(target, "rt", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError, EOFError) as exc:
        _log.info("%s: layout cache unreadable (%s); working it out again",
                  os.path.basename(source), exc)
        return None

    if not isinstance(document, dict):
        return None
    if document.get("format") != LAYOUT_FORMAT:
        return None
    if document.get("source") != stamp:
        _log.info("%s: the file has changed since its layout was written",
                  os.path.basename(source))
        return None

    try:
        runs = {str(key): {str(c): str(p) for c, p in value.items()}
                for key, value in document["runs"].items()}
        channels = [str(name) for name in document["channels"]]
        layout = Layout(str(document["rule"]), runs, channels,
                        int(document["samples"]))
        units = [str(unit) for unit in document["units"]]
        alternatives = [str(name) for name in document.get("alternatives", [])]
    except (KeyError, TypeError, ValueError, AttributeError):
        return None

    if not runs or not channels or len(units) != len(channels):
        return None

    if not _layout_still_fits(source, layout):
        return None

    _log.info("%s: layout read from cache -- '%s', %d runs x %d channels "
              "(no tree walk)", os.path.basename(source), layout.rule,
              len(layout.runs), len(layout.channels))
    return Hdf5File(source, layout, units, alternatives)


def _layout_still_fits(source: str, layout: "Layout") -> bool:
    """Open one dataset the layout names and check it is as described."""
    try:
        handle = _open(source)
        first_run = layout.runs[layout.run_keys[0]]
        dataset_path = first_run[layout.channels[0]]
        dataset = handle[dataset_path]
        return (dataset.ndim == 1
                and int(dataset.shape[0]) == int(layout.samples))
    except Exception as exc:
        _log.info("%s: the cached layout no longer fits (%s)",
                  os.path.basename(source), exc)
        return False


# --------------------------------------------------------------------------
# working the layout out
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# files that describe themselves
# --------------------------------------------------------------------------
#
# Everything above infers the layout, because most exporters write a bare tree
# of arrays and leave you to work out which axis is which. Some do better than
# that, and when a file states its own structure there is no reason to guess
# at it -- guessing is strictly worse, and on this schema it was also wrong.
#
#     /DOMAIN/OSMA_1/PARAMETERS/ALFA/ALFA_1_2_0      the samples
#     /DOMAIN/OSMA_1/PARAMETERS/ALFA   @unit=deg     the channel, named
#     /DOMAIN/OSMA_1/SIMULATIONS/SIMULATION_1_2_0    the run, named
#
# PARAMETERS lists the channels, SIMULATIONS lists the runs, and a dataset is
# the two put together. Nothing has to be inferred and nothing has to be
# ranked, so the answer cannot be the wrong way round.
#
# A file holding more than one session -- OSMA_1 beside OSMA_2 -- is the case
# that made inference fail outright rather than merely waste time. The same
# run name appears under every session, so a reading keyed on the dataset name
# alone sees two datasets claiming to be the same channel of the same run,
# declares itself inconsistent, and stands down in favour of a reading that is
# consistent and useless: one "run" called DOMAIN carrying every dataset in
# the file as a channel of its own. Here the session is part of the run's
# identity, which is what it actually is.

SCHEMA_CHANNELS = "PARAMETERS"
SCHEMA_RUNS = "SIMULATIONS"
SCHEMA_RUN_PREFIX = "SIMULATION_"
# How far down to look for a session. Two levels: the root itself, its
# children, and theirs -- which is where /DOMAIN/OSMA_1 sits. Going deeper
# costs an open per group at every level, and on a file that does not use this
# schema at all those opens buy nothing: a tree of a thousand channel groups
# would pay a thousand opens to discover it is not what we are looking for.
_SCHEMA_MAX_DEPTH = 2

# And a hard ceiling on the opens regardless, for a file with a very wide top
# level. Reaching it means giving up on the schema and inferring instead,
# which is the correct outcome: a session container is narrow.
_SCHEMA_MAX_OPENS = 128


def _sessions(handle) -> List[Tuple[str, object]]:
    """(name, group) for every group holding both PARAMETERS and SIMULATIONS.

    Found by listing group members, which is one read per group visited and
    stops as soon as a session is found, rather than by walking the tree.
    """
    found: List[Tuple[str, object]] = []
    budget = [_SCHEMA_MAX_OPENS]

    def look(name, group, depth):
        try:
            members = set(group.keys())
        except Exception:
            return
        if SCHEMA_CHANNELS in members and SCHEMA_RUNS in members:
            found.append((name, group))
            return
        if depth >= _SCHEMA_MAX_DEPTH:
            return
        for key in sorted(members):
            if budget[0] <= 0:
                return
            budget[0] -= 1
            try:
                child = group[key]
            except Exception:
                continue
            if hasattr(child, "keys"):
                look("%s/%s" % (name, key) if name else key, child, depth + 1)

    look("", handle, 0)
    return found


def _schema_unit(group) -> str:
    for key in _UNIT_KEYS:
        try:
            if key not in group.attrs:
                continue
            value = group.attrs[key]
        except Exception:
            continue
        if isinstance(value, bytes):
            return value.decode("utf-8", "replace").strip()
        return str(value).strip()
    return ""


def _inspect_by_schema(handle, path: str) -> Optional["Hdf5File"]:
    """Read a file that names its own channels and runs. None if it is not one.

    The reads are: one listing per session for its channels and its runs, one
    attribute per channel for the unit, and one dataset opened to learn the
    sample count. Nothing else is touched, and no dataset name is enumerated
    at all -- a dataset's path is the channel's and the run's put together.
    """
    sessions = _sessions(handle)
    if not sessions:
        return None

    runs: Dict[str, Dict[str, str]] = {}
    units: Dict[str, str] = {}
    channel_order: List[str] = []
    samples = 0
    many = len(sessions) > 1

    for session_name, session in sessions:
        try:
            parameters = session[SCHEMA_CHANNELS]
            simulations = session[SCHEMA_RUNS]
            channels = sorted(parameters.keys(), key=_natural_key)
            simulation_names = sorted(simulations.keys(), key=_natural_key)
        except Exception:
            return None
        if not channels or not simulation_names:
            return None

        base = session.name if getattr(session, "name", None) else \
            "/" + session_name
        base = base.rstrip("/")

        for channel in channels:
            if channel not in units:
                try:
                    units[channel] = _schema_unit(parameters[channel])
                except Exception:
                    units[channel] = ""
                channel_order.append(channel)

        for simulation in simulation_names:
            tail = simulation[len(SCHEMA_RUN_PREFIX):] \
                if simulation.startswith(SCHEMA_RUN_PREFIX) else simulation
            # The session belongs in the run's identity when there is more
            # than one, or two sessions' run 1_2_0 would be the same run.
            run_key = "%s/%s" % (session_name.rpartition("/")[2], tail) \
                if many else tail
            entry = {}
            for channel in channels:
                entry[channel] = "%s/%s/%s/%s_%s" % (
                    base, SCHEMA_CHANNELS, channel, channel, tail)
            if run_key in runs:
                return None
            runs[run_key] = entry

            if not samples:
                probe = _measure(handle, entry[channels[0]])
                if probe is None:
                    # The names do not line up the way this schema says they
                    # do, so it is not this schema after all.
                    return None
                samples = probe[0]

    if not runs or not samples:
        return None

    # Only the channels every run carries can be given a stable column index.
    shared = set(channel_order)
    for entry in runs.values():
        shared &= set(entry)
    channels = [name for name in channel_order if name in shared]
    if not channels:
        return None
    trimmed = {key: {name: entry[name] for name in channels}
               for key, entry in runs.items()}

    layout = Layout("declared", trimmed, channels, samples)
    _log.info("%s: read as declared -- %d session(s), %d runs x %d channels "
              "x %d samples (no guessing)", os.path.basename(path),
              len(sessions), len(layout.runs), len(channels), samples)
    return Hdf5File(path, layout, [units.get(name, "") for name in channels],
                    [])


def _link_names(handle) -> Optional[List[str]]:
    """Every path in the file, by walking links rather than objects.

    The expensive part of the old walk was not the traversal, it was that
    `visititems` opens every object it passes and this code then asked each
    one for its dtype and shape -- three metadata reads per dataset, fifty
    thousand datasets.

    None of the layout rules look at a dataset's contents. Every one of them
    works on the path alone, so the names are enough to decide the convention,
    and shapes are needed only for the handful of datasets that are actually
    checked afterwards.

    Returns None when the low-level traversal is unavailable, which sends the
    caller back to the object walk.
    """
    names: List[str] = []

    def collect(name):
        names.append(name.decode("utf-8", "replace")
                     if isinstance(name, bytes) else str(name))
        return None

    try:
        handle.id.links.visit(collect)
    except Exception:
        return None
    return names or None


def _leaf_names(names: Sequence[str]) -> List[str]:
    """The paths nothing hangs beneath: the datasets.

    A dataset cannot contain links, so a path that is not a prefix of another
    is a dataset or an empty group. An empty group that slips through is
    harmless -- it is dropped when the datasets are checked.
    """
    prefixes = set()
    for name in names:
        cut = name.rfind("/")
        while cut > 0:
            prefixes.add(name[:cut])
            cut = name.rfind("/", 0, cut)
    return ["/" + name for name in names if name not in prefixes]


# How much of the tree a convention is allowed not to understand. A real file
# carries odd things among its channels -- a short AUX vector, an empty
# METADATA group -- and the strict partition used by the object walk rejects a
# convention outright when it cannot key even one path. That is the right
# answer there, because that walk has already thrown away everything of the
# wrong length before it starts. Working from names there is no length to
# filter on yet, so a convention is allowed to skip what it cannot key, and is
# rejected only when it fails on so much of the file that it is plainly the
# wrong reading.
MAX_UNKEYED = 0.25


def _partition_loosely(rule, paths: Sequence[str]):
    """Group paths by run and channel, skipping the ones the rule cannot key.

    None when the reading is impossible rather than merely incomplete: two
    datasets claiming to be the same channel of the same run, or more than
    MAX_UNKEYED of the paths unkeyable.
    """
    grouped: Dict[str, Dict[str, str]] = {}
    unkeyed = 0
    for path in paths:
        keyed = rule(path)
        if keyed is None or not keyed[1]:
            unkeyed += 1
            continue
        run_key, channel_key = keyed
        if channel_key in grouped.setdefault(run_key, {}):
            return None
        grouped[run_key][channel_key] = path
    if not grouped:
        return None
    if unkeyed > MAX_UNKEYED * max(1, len(paths)):
        return None
    return grouped


def _rank_on_names(paths: Sequence[str], parents: Dict[str, str]):
    """Score every convention against these paths, best first."""
    sample = _ranking_sample(sorted(paths))
    ranked = []
    for rule_name, rule in _RULES:
        grouped = _partition_loosely(rule, sample)
        if grouped is None:
            continue
        ranked.append((_score_grouped(grouped, parents), rule_name, rule))
    ranked.sort(key=lambda item: -item[0])
    return ranked


def _measure(handle, dataset_path: str):
    """(length, unit) of one dataset, or None if it is not a channel."""
    try:
        dataset = handle[dataset_path]
        if dataset.dtype.kind not in "fiu" or dataset.ndim != 1:
            return None
        length = int(dataset.shape[0])
        if length < 2:
            return None
        return length, _unit_of(dataset)
    except Exception:
        return None


def _verify(handle, rule_name: str, grouped, path: str):
    """Turn a grouping worked out from names into a checked Layout.

    This is where the reads happen, and there are three kinds. Every dataset
    of the *first* run is opened, which settles the common length, drops
    anything that is not a 1-D numeric channel, and collects the units in the
    same pass. One dataset of every *other* run is opened to confirm it agrees.
    Nothing else is touched.

    So the cost is one read per channel plus one per run -- on a file of 50
    runs by 1000 channels, 1,050 instead of 50,000.
    """
    run_keys = sorted(grouped, key=_natural_key)
    first = grouped[run_keys[0]]

    lengths: Dict[int, int] = {}
    measured = {}
    for channel, dataset_path in first.items():
        found = _measure(handle, dataset_path)
        if found is None:
            continue
        measured[channel] = found
        lengths[found[0]] = lengths.get(found[0], 0) + 1
    if not measured:
        return None

    # The length most of the first run's channels share. A short metadata
    # vector sitting beside the channels is dropped by this rather than
    # becoming a channel two samples long.
    samples = max(lengths, key=lambda value: (lengths[value], value))
    channels = sorted((name for name, (length, _unit) in measured.items()
                       if length == samples), key=_natural_key)
    if not channels:
        return None

    # Only the channels every run carries can be given a stable column index,
    # and a stable index is what lets a tile ask for "channel 7" of any run.
    shared = set(channels)
    for run_key in run_keys[1:]:
        shared &= set(grouped[run_key])
    channels = [name for name in channels if name in shared]
    if not channels:
        return None

    # Every other run has to agree on the length, and one dataset each says
    # whether it does.
    for run_key in run_keys[1:]:
        found = _measure(handle, grouped[run_key][channels[0]])
        if found is None or found[0] != samples:
            # A file whose runs are of different lengths is not one this
            # reading describes; the object walk groups by length and will
            # make a different, and correct, choice.
            return None

    runs = {run_key: {name: grouped[run_key][name] for name in channels}
            for run_key in run_keys}

    layout = Layout(rule_name, runs, channels, samples)
    units = [measured[name][1] for name in channels]
    return layout, units


def _inspect_by_name(handle, path: str) -> Optional["Hdf5File"]:
    """Work the layout out from paths, checking only what has to be checked.

    Returns None when this cannot be done -- no cheap traversal, nothing that
    looks like a channel, or no convention that survives being checked -- and
    the caller falls back to opening every dataset.
    """
    names = _link_names(handle)
    if not names:
        return None
    leaves = _leaf_names(names)
    if len(leaves) < 2:
        return None

    parents = {leaf: leaf.rpartition("/")[0] or "/" for leaf in leaves}
    ranked = _rank_on_names(leaves, parents)
    if not ranked:
        return None

    for score, rule_name, rule in ranked:
        grouped = _partition_loosely(rule, sorted(leaves))
        if grouped is None:
            continue
        checked = _verify(handle, rule_name, grouped, path)
        if checked is None:
            continue
        layout, units = checked
        alternatives = ["%s (%.2f)" % (other[1], other[0])
                        for other in ranked[1:4]]
        _log.info("%s: read as '%s' -- %d runs x %d channels x %d samples "
                  "(score %.2f, from %d names; also considered: %s)",
                  os.path.basename(path), layout.rule, len(layout.runs),
                  len(layout.channels), layout.samples, score, len(leaves),
                  ", ".join(alternatives) or "nothing else fitted")
        return Hdf5File(path, layout, units,
                        [other[1] for other in ranked[1:]])
    return None


def _inspect_by_object(handle, path: str) -> "Hdf5File":
    """The thorough walk: open every dataset and ask it what it is.

    Correct whatever the file looks like, and expensive in exactly the way
    that hurts over a network. Kept as the fallback for the cases the cheap
    path declines: an h5py without the low-level link traversal, runs of
    differing lengths, or a tree whose datasets cannot be told from its groups
    by shape alone.
    """
    h5py = _require_h5py()
    vectors: Dict[str, int] = {}

    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return None
        if obj.dtype.kind not in "fiu" or obj.ndim != 1:
            return None
        if obj.shape[0] < 2:
            return None
        vectors["/" + name] = int(obj.shape[0])
        return None

    # Units are deliberately NOT read here. Probing six attribute names on
    # every dataset is 300,000 HDF5 metadata calls on a file of 50 runs x
    # 1000 channels, to build a list of 1000 units of which 980 are the same
    # answer repeated once per run. They are read from one run below, once the
    # layout says which datasets that run owns.
    handle.visititems(visit)

    if not vectors:
        raise ValueError(
            "no 1-D numeric datasets found: this file may store its "
            "channels as tables rather than one dataset per channel")

    ranked = _layouts(vectors)
    if not ranked:
        raise ValueError(
            "could not work out which datasets belong to which run")

    score, layout = ranked[0]
    alternatives = ["%s (%.2f)" % (other.rule, other_score)
                    for other_score, other in ranked[1:4]]
    _log.info("%s: read as '%s' -- %d runs x %d channels x %d samples "
              "(score %.2f, by opening every dataset; also considered: %s)",
              os.path.basename(path), layout.rule, len(layout.runs),
              len(layout.channels), layout.samples, score,
              ", ".join(alternatives) or "nothing else fitted")

    first = layout.runs[layout.run_keys[0]]
    channel_units = []
    for key in layout.channels:
        dataset_path = first.get(key)
        if dataset_path is None:
            channel_units.append("")
            continue
        try:
            channel_units.append(_unit_of(handle[dataset_path]))
        except (KeyError, OSError):
            channel_units.append("")

    return Hdf5File(path, layout, channel_units,
                    [other.rule for _, other in ranked[1:]])


def inspect(path: str, use_cache: bool = True) -> Hdf5File:
    """Work out how this file is laid out. Raises ValueError if it cannot.

    Three routes, cheapest first: the layout written down last time, the paths
    in the file, and finally every dataset in it.
    """
    _require_h5py()

    if use_cache:
        cached = _load_layout(path)
        if cached is not None:
            return cached

    handle = _open(path)
    # A file that names its own channels and runs is read as it says; only a
    # file that does not is guessed at.
    described = _inspect_by_schema(handle, path)
    if described is None:
        described = _inspect_by_name(handle, path)
    if described is None:
        described = _inspect_by_object(handle, path)

    if use_cache:
        _save_layout(path, described)
    return described


def read_first_values(path: str, layout: Layout, channel_index: int):
    """The first sample of one channel in every run, without reading a column.

    What an ISO scan needs. The core answers this for a parsed file by reading
    the first record of each block through its index, which is the difference
    between an ISO scan taking microseconds and taking seconds; a pushed file
    has no index here, but HDF5 can read element zero of a dataset directly,
    which costs one chunk instead of the column.

    NaN where a run does not carry the channel, which is what the core returns
    for the same case.
    """
    run_keys = layout.run_keys
    out = np.full(len(run_keys), np.nan, dtype=np.float32)
    if not (0 <= channel_index < len(layout.channels)):
        return out
    channel = layout.channels[channel_index]

    handle = _open(path)
    for index, run_key in enumerate(run_keys):
        dataset_path = layout.runs[run_key].get(channel)
        if dataset_path is None:
            continue
        dataset = handle[dataset_path]
        if dataset.shape[0] == 0:
            continue
        out[index] = np.float32(dataset[0])
    return out


def read_selected(path: str, layout: Layout,
                  wanted: Sequence[Tuple[int, int]]):
    """Yield (run index, channel index, samples) for just the pairs asked for.

    The lazy half of the reader. A page of four tiles wants four channels out
    of forty, and reading the other thirty-six is most of the time a large
    file takes to open -- so this reads the datasets named and no others.

    The file is opened once for the whole batch and the pairs are walked in
    (run, channel) order, which is the order the datasets of one run sit in the
    tree. The handle is shared and outlives the call, so a second batch in the
    same task does not reopen the file at all.
    """
    run_keys = layout.run_keys
    ordered = sorted(set((int(run), int(column)) for run, column in wanted))

    handle = _open(path)
    for run_index, channel_index in ordered:
        if not (0 <= run_index < len(run_keys)):
            continue
        if not (0 <= channel_index < len(layout.channels)):
            continue
        channels = layout.runs[run_keys[run_index]]
        dataset_path = channels.get(layout.channels[channel_index])
        if dataset_path is None:
            continue
        values = np.asarray(handle[dataset_path][:], dtype=np.float32)
        yield run_index, channel_index, np.ascontiguousarray(values)


def read_columns(path: str, layout: Layout):
    """Yield (run index, channel index, samples) for every dataset in turn.

    One dataset is open at a time and converted to float32 immediately, which
    is what the core stores anyway: reading the file into a dict first would
    hold the whole of it twice at the moment of the push.
    """
    handle = _open(path)
    for run_index, run_key in enumerate(layout.run_keys):
        channels = layout.runs[run_key]
        for channel_index, channel_key in enumerate(layout.channels):
            dataset_path = channels.get(channel_key)
            if dataset_path is None:
                continue
            values = np.asarray(handle[dataset_path][:], dtype=np.float32)
            yield run_index, channel_index, np.ascontiguousarray(values)
