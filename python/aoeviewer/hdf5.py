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


def inspect(path: str) -> Hdf5File:
    """Work out how this file is laid out. Raises ValueError if it cannot."""
    try:
        import h5py
    except ImportError:
        raise ValueError(
            "reading HDF5 files needs the h5py package, which is not "
            "installed")

    vectors: Dict[str, int] = {}

    with h5py.File(path, "r") as handle:
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
        # 1000 channels, to build a list of 1000 units of which 980 are the
        # same answer repeated once per run. They are read from one run below,
        # once the layout says which datasets that run owns.
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
                  "(score %.2f; also considered: %s)",
                  os.path.basename(path), layout.rule, len(layout.runs),
                  len(layout.channels), layout.samples, score,
                  ", ".join(alternatives) or "nothing else fitted")

        # One run's datasets, which is where a unit is declared once per
        # channel rather than once per channel per run.
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
    import h5py

    run_keys = layout.run_keys
    out = np.full(len(run_keys), np.nan, dtype=np.float32)
    if not (0 <= channel_index < len(layout.channels)):
        return out
    channel = layout.channels[channel_index]

    with h5py.File(path, "r") as handle:
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
    tree: the alternative, one open per column, pays the HDF5 open and its
    metadata walk per channel.
    """
    import h5py

    run_keys = layout.run_keys
    ordered = sorted(set((int(run), int(column)) for run, column in wanted))

    with h5py.File(path, "r") as handle:
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
    import h5py

    with h5py.File(path, "r") as handle:
        for run_index, run_key in enumerate(layout.run_keys):
            channels = layout.runs[run_key]
            for channel_index, channel_key in enumerate(layout.channels):
                dataset_path = channels.get(channel_key)
                if dataset_path is None:
                    continue
                values = np.asarray(handle[dataset_path][:], dtype=np.float32)
                yield run_index, channel_index, np.ascontiguousarray(values)
