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


def _score(layout: Layout, parents: Dict[str, str]) -> float:
    """How much the tree itself supports this reading of it.

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
    """
    total = 0
    matching = 0
    by_parent: Dict[str, set] = {}
    for run_key, channels in layout.runs.items():
        for channel_key, dataset_path in channels.items():
            total += 1
            parent = parents[dataset_path]
            if channel_key and parent.rpartition("/")[2] == channel_key:
                matching += 1
            by_parent.setdefault(parent, set()).add(run_key)

    score = _W_CHANNEL_MATCHES_PARENT * (matching / float(total or 1))
    if len(by_parent) > 1 and len(set(map(frozenset, by_parent.values()))) == 1:
        score += _W_RUNS_ORTHOGONAL
    if len(layout.runs) > 1:
        score += _W_SEVERAL_RUNS
    return score


def _layouts(vectors: Dict[str, int]) -> List[Tuple[float, Layout]]:
    """Every consistent reading of a set of 1-D datasets, best first."""
    parents = {path: path.rpartition("/")[0] or "/" for path in vectors}

    by_length: Dict[int, List[str]] = {}
    for path, length in vectors.items():
        by_length.setdefault(length, []).append(path)

    found = []
    seen = set()
    # Longest family first: a file usually carries one set of channels plus a
    # short vector or two of metadata, and the channels are the long ones.
    for length, members in sorted(by_length.items(),
                                  key=lambda kv: (-len(kv[1]), -kv[0])):
        if len(members) < 2:
            continue
        for rule_name, rule in _RULES:
            grouped: Dict[str, Dict[str, str]] = {}
            ok = True
            for path in members:
                keyed = rule(path)
                if keyed is None:
                    ok = False
                    break
                run_key, channel_key = keyed
                if not channel_key:
                    ok = False
                    break
                if channel_key in grouped.setdefault(run_key, {}):
                    # Two datasets claiming to be the same channel of the same
                    # run: this reading cannot be the right one.
                    ok = False
                    break
                grouped[run_key][channel_key] = path
            if not ok or not grouped:
                continue

            shared = None
            for channels in grouped.values():
                keys = set(channels)
                shared = keys if shared is None else (shared & keys)
            if not shared:
                continue

            ordered = sorted(shared, key=_natural_key)
            trimmed = {run_key: {key: channels[key] for key in ordered}
                       for run_key, channels in grouped.items()}
            signature = (tuple(sorted(trimmed)), tuple(ordered))
            if signature in seen:
                continue
            seen.add(signature)

            layout = Layout(rule_name, trimmed, ordered, length)
            found.append((_score(layout, parents), layout))
        if found:
            # A shorter family cannot describe the file better than the one
            # holding most of its datasets.
            break

    # A stable sort, so that among readings the evidence cannot separate, the
    # one declared first in _RULES wins. That ordering is deliberate: the
    # plainest convention should be preferred when nothing argues against it.
    found.sort(key=lambda item: -item[0])
    return found


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
    units: Dict[str, str] = {}

    with h5py.File(path, "r") as handle:
        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return None
            if obj.dtype.kind not in "fiu" or obj.ndim != 1:
                return None
            if obj.shape[0] < 2:
                return None
            full = "/" + name
            vectors[full] = int(obj.shape[0])
            unit = _unit_of(obj)
            if unit:
                units[full] = unit
            return None

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

        first = layout.runs[layout.run_keys[0]]
        channel_units = [units.get(first[key], "") for key in layout.channels]

    return Hdf5File(path, layout, channel_units,
                    [other.rule for _, other in ranked[1:]])


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
