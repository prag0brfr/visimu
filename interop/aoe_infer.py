#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Infer how to read an unknown format, by verifying against its .aoe twin.

The other two scripts assume a human in the loop: profile the file, send the
profile, get a reader back, grade it.  When both files are in hand that loop is
unnecessary, because the twin is ground truth.  A parse is either right or
wrong, and the twin says which -- so the machine can search the space of
plausible parses itself and keep the one that reproduces the twin.

What it does, in one run:

1.  Profiles the unknown file structurally (the same analysis `aoe_probe.py`
    does).
2.  Enumerates every plausible way to read it -- each delimiter, fixed-width
    columns, each HDF5 layout convention, each binary word size, endianness
    and row/column order.
3.  Parses the file each way and matches the resulting columns against the
    twin's channels **by their values**, not by their names.  This is the part
    that removes the guesswork: a column is identified as ALTITUDE because it
    *is* ALTITUDE, even when the file has no header at all, or spells it
    differently, or stores it in feet -- an affine relationship is detected and
    reported as a unit conversion.
4.  Keeps the strategy that explains the most of the twin, writes a working
    reader for it, runs that reader, and grades it against the twin.

The result is one report and one generated reader. The report carries the parse
strategy, the column-to-channel map, and the agreement statistics; it carries no
sample of anyone's data.

    python aoe_infer.py --aoe RUN.aoe --other RUN.DAT
    python aoe_infer.py --aoe RUN.aoe --other RUN.h5 -o inferred.json
    python aoe_infer.py --aoe RUN.aoe --other RUN.BIN --no-constants

The generated reader lands in `readers/generated_<name>_reader.py` and is a
plain file you can read, edit and keep.
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import math
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in (HERE, os.path.join(os.path.dirname(HERE), "python")):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import numpy as np                                      # noqa: E402

import aoe_probe                                        # noqa: E402
import aoe_compare                                      # noqa: E402

FINGERPRINT_SAMPLES = 96
DEFAULT_TOLERANCE = 1e-5
MIN_CORRELATION = 1.0 - 1e-9

# Conversions worth naming when a column turns out to be a scaled twin of a
# channel.  Recognising the factor is what turns "column 11 is ALTITUDE times
# 3.28084" into "column 11 is ALTITUDE in feet".
KNOWN_FACTORS = [
    (3.280839895013123, "metres -> feet"),
    (0.3048, "feet -> metres"),
    (57.29577951308232, "radians -> degrees"),
    (0.017453292519943295, "degrees -> radians"),
    (1.9438444924406046, "m/s -> knots"),
    (0.5144444444444445, "knots -> m/s"),
    (3.6, "m/s -> km/h"),
    (0.2777777777777778, "km/h -> m/s"),
    (2.204622621848776, "kg -> lb"),
    (0.45359237, "lb -> kg"),
    (0.001, "unit -> milli"),
    (1000.0, "milli -> unit"),
    (-1.0, "sign reversed"),
]


def name_factor(scale):
    """Name a gain when it is a unit conversion, within float32 slack.

    The gain comes out of a least-squares fit on values the .aoe stored as
    float32, so it is never exactly 0.3048; matching has to be as loose as the
    data is, and 1e-6 relative is still far tighter than the gap between any
    two conversions in the table.
    """
    for value, label in KNOWN_FACTORS:
        if abs(scale - value) <= 1e-6 * max(1.0, abs(value)):
            return label
    return None


# --------------------------------------------------------------------------
# parsed shapes
# --------------------------------------------------------------------------

class ParsedRun(object):
    """One run as some candidate strategy sees it: labelled, unnamed columns."""

    def __init__(self, label, columns, names=None):
        self.label = label
        self.columns = columns
        self.names = names

    @property
    def width(self):
        return len(self.columns)


class Strategy(object):
    """A way of reading the file, and the description that explains it."""

    def __init__(self, key, description, runs):
        self.key = key
        self.description = description
        self.runs = runs

    @property
    def viable(self):
        return bool(self.runs) and all(run.width for run in self.runs)


# --------------------------------------------------------------------------
# candidate strategies: text
# --------------------------------------------------------------------------

def _to_float(token):
    token = token.strip()
    if not token:
        return float("nan")
    try:
        return float(token)
    except ValueError:
        pass
    try:
        return float(token.replace("D", "E").replace("d", "e")
                     .replace("Q", "E").replace("q", "e"))
    except ValueError:
        pass
    try:
        return float(re.sub(r"(\d)([+-]\d{1,3})$", r"\1E\2", token))
    except ValueError:
        return float("nan")


def _numeric_share(tokens):
    if not tokens:
        return 0.0
    good = sum(1 for token in tokens
               if aoe_probe.classify_token(aoe_probe.unquote(token))
               in ("int", "float", "special"))
    return good / float(len(tokens))


def _group_runs(rows, labels, names, width):
    """Split consecutive data rows into runs, one per contiguous block."""
    runs = []
    block = []
    block_label = None
    for index, (row, label) in enumerate(zip(rows, labels)):
        if row is None:                     # a separator line ended the block
            if block:
                runs.append((block_label, block))
                block = []
                block_label = None
            if label:
                block_label = label
            continue
        block.append(row)
    if block:
        runs.append((block_label, block))

    parsed = []
    for order, (label, block) in enumerate(runs):
        if not block:
            continue
        columns = [np.asarray([row[j] for row in block], dtype=np.float64)
                   for j in range(width)]
        parsed.append(ParsedRun(label or str(order + 1), columns, names))
    return parsed


def _label_of(line):
    """The run label carried by a separator line, if it carries one.

    ``*TABLEAU : 3`` and ``RUN 12`` both name a run; the convention varies, so
    the last token that looks like an identifier or a number is taken.
    """
    tokens = [aoe_probe.unquote(t) for t in line.replace(":", " ").split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return None
    tail = tokens[-1]
    if re.match(r"^[A-Za-z0-9_\-.]+$", tail):
        return tail
    return None


def text_strategies(path, max_bytes):
    raw, _, truncated = aoe_probe.read_windows(path, max_bytes)
    if truncated:
        # An inferred reader has to be graded against the whole twin, so a
        # partial read would only produce a strategy that then fails.
        with open(path, "rb") as handle:
            raw = handle.read()
    text, codec, _ = aoe_probe.decode_best(raw)
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    out = []
    for delimiter, label in aoe_probe.CANDIDATE_DELIMITERS:
        tokenised = [aoe_probe.split_fields(line, delimiter)
                     if line.strip() else [] for line in lines]
        widths = {}
        for tokens in tokenised:
            if len(tokens) >= 2 and _numeric_share(tokens) >= 0.6:
                widths[len(tokens)] = widths.get(len(tokens), 0) + 1
        if not widths:
            continue
        width = max(widths, key=lambda k: widths[k])
        if widths[width] < 3:
            continue

        rows = []
        labels = []
        names = None
        for line, tokens in zip(lines, tokenised):
            if len(tokens) == width and _numeric_share(tokens) >= 0.6:
                rows.append([_to_float(token) for token in tokens])
                labels.append(None)
            else:
                rows.append(None)
                labels.append(_label_of(line) if line.strip() else None)
                if (names is None and len(tokens) == width
                        and _numeric_share(tokens) < 0.3):
                    names = [aoe_probe.unquote(token) for token in tokens]

        runs = _group_runs(rows, labels, names, width)
        if runs:
            out.append(Strategy(
                "text:%s" % label,
                {
                    "kind": "delimited_text",
                    "encoding": codec,
                    "delimiter": label,
                    "field_count": width,
                    "header_present": names is not None,
                    "runs_found": len(runs),
                },
                runs))

    fixed = aoe_probe.fixed_width_columns(lines)
    if fixed:
        slices = [(f["start"], f["width"]) for f in fixed["fields"]]
        rows = []
        labels = []
        names = None
        for line in lines:
            if not line.strip():
                rows.append(None)
                labels.append(None)
                continue
            cells = [line[start:start + length] for start, length in slices]
            if _numeric_share([c for c in cells if c.strip()]) >= 0.6 and \
                    any(c.strip() for c in cells):
                rows.append([_to_float(cell) for cell in cells])
                labels.append(None)
            else:
                rows.append(None)
                labels.append(_label_of(line))
                if names is None and len(line.split()) == len(slices):
                    names = [aoe_probe.unquote(t) for t in line.split()]
        runs = _group_runs(rows, labels, names, len(slices))
        if runs:
            out.append(Strategy(
                "text:fixed",
                {
                    "kind": "fixed_width_text",
                    "encoding": codec,
                    "field_count": len(slices),
                    "fields": [{"start": s, "width": w} for s, w in slices],
                    "header_present": names is not None,
                    "runs_found": len(runs),
                },
                runs))
    return out


# --------------------------------------------------------------------------
# candidate strategies: HDF5
# --------------------------------------------------------------------------

# How a dataset's location encodes which run and which channel it holds.  HDF5
# is a container, not a layout, so every exporter invents its own convention;
# these are the ones seen in the wild, and the twin decides which is in force.

def _rule_parent_group(path):
    """/RUN_1/ALTITUDE -- the group is the run, the dataset is the channel."""
    parent, _, leaf = path.rpartition("/")
    return (parent or "/"), leaf


def _rule_leaf_tail(count):
    """/ALTITUDE/ALTITUDE_1_2_0 -- the run is a suffix of the dataset name.

    The transpose of the case above, and just as common: a group per channel,
    a dataset per run, with the run identity spelled into the leaf name.
    """
    def rule(path):
        _, _, leaf = path.rpartition("/")
        tokens = leaf.split("_")
        if len(tokens) <= count:
            return None
        return "_".join(tokens[-count:]), "_".join(tokens[:-count])
    return rule


def _rule_ancestor(depth):
    """/CASE_2/aero/ALTITUDE -- the run is an ancestor further up the tree."""
    def rule(path):
        parts = path.strip("/").split("/")
        if len(parts) <= depth:
            return None
        return "/".join(parts[:depth]), "/".join(parts[depth:])
    return rule


def _rule_single_run(path):
    """Everything belongs to one run; only the channel varies."""
    _, _, leaf = path.rpartition("/")
    return "1", leaf


HDF5_RULES = (
    [("parent_group", 0, _rule_parent_group)]
    + [("leaf_tail_%d" % k, k, _rule_leaf_tail(k)) for k in (1, 2, 3, 4)]
    + [("ancestor_%d" % k, k, _rule_ancestor(k)) for k in (1, 2, 3)]
    + [("single_run", 0, _rule_single_run)]
)


def hdf5_strategies(path):
    try:
        import h5py
    except ImportError:
        return []

    out = []
    with h5py.File(path, "r") as handle:
        vectors = []
        tables = []
        cubes = []

        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset) or obj.dtype.kind not in "fiu":
                return None
            if obj.ndim == 1 and obj.shape[0] >= 2:
                vectors.append("/" + name)
            elif obj.ndim == 2 and min(obj.shape) >= 2:
                tables.append(("/" + name, obj))
            elif obj.ndim == 3:
                cubes.append(("/" + name, obj))
            return None

        handle.visititems(visit)

        # 1-D datasets: the common case, and the one where the convention is
        # invisible without a twin.  Group them by length first -- a file often
        # carries a short metadata vector beside the long channels.
        cache = {}

        def values(dataset_path):
            if dataset_path not in cache:
                cache[dataset_path] = np.asarray(
                    handle[dataset_path][:], dtype=np.float64)
            return cache[dataset_path]

        by_length = {}
        for dataset_path in vectors:
            length = int(handle[dataset_path].shape[0])
            by_length.setdefault(length, []).append(dataset_path)

        seen_partitions = set()
        for length, members in sorted(by_length.items(),
                                      key=lambda kv: -len(kv[1])):
            if len(members) < 2:
                continue
            for label, count, rule in HDF5_RULES:
                grouped = {}
                ok = True
                for dataset_path in members:
                    keyed = rule(dataset_path)
                    if keyed is None:
                        ok = False
                        break
                    run_key, channel_key = keyed
                    if channel_key in grouped.setdefault(run_key, {}):
                        ok = False          # ambiguous: two channels collide
                        break
                    grouped[run_key][channel_key] = dataset_path
                if not ok or not grouped:
                    continue

                # Only the channels every run carries can be given a stable
                # column index, and a stable index is what lets the votes from
                # different runs be compared at all.
                shared = None
                for channels in grouped.values():
                    keys = set(channels)
                    shared = keys if shared is None else (shared & keys)
                if not shared or len(shared) < 2:
                    continue
                ordered = sorted(shared)

                signature = (label, length, tuple(sorted(grouped)),
                             tuple(ordered))
                if signature in seen_partitions:
                    continue
                seen_partitions.add(signature)

                runs = [
                    ParsedRun(run_key,
                              [values(grouped[run_key][key]) for key in ordered],
                              list(ordered))
                    for run_key in sorted(grouped)
                ]
                out.append(Strategy(
                    "hdf5:%s" % label,
                    {"kind": "hdf5",
                     "layout": "1-D datasets, run identified by %s"
                               % label.replace("_", " "),
                     "rule": label,
                     "rule_tokens": count,
                     "samples": length,
                     "channels_per_run": len(ordered),
                     "runs_found": len(runs)},
                    runs))

        # 2-D tables, one per run, channels along either axis.
        for axis in (1, 0):
            runs = []
            for name, obj in sorted(tables):
                block = np.asarray(obj[:], dtype=np.float64)
                if axis == 1:
                    columns = [block[:, j] for j in range(block.shape[1])]
                else:
                    columns = [block[j, :] for j in range(block.shape[0])]
                runs.append(ParsedRun(name.split("/")[-1], columns))
            if runs:
                out.append(Strategy(
                    "hdf5:table_axis%d" % axis,
                    {"kind": "hdf5",
                     "layout": "one 2-D dataset per run, channels along axis %d"
                               % axis,
                     "rule": "table", "axis": axis,
                     "runs_found": len(runs)},
                    runs))

        # A single 3-D block, runs along the first axis.
        for axis in (2, 1):
            runs = []
            for name, obj in sorted(cubes):
                block = np.asarray(obj[:], dtype=np.float64)
                for index in range(block.shape[0]):
                    plane = block[index]
                    if axis == 2:
                        columns = [plane[:, j] for j in range(plane.shape[1])]
                    else:
                        columns = [plane[j, :] for j in range(plane.shape[0])]
                    runs.append(ParsedRun(str(index + 1), columns))
            if runs:
                out.append(Strategy(
                    "hdf5:cube_axis%d" % axis,
                    {"kind": "hdf5",
                     "layout": "one 3-D dataset, runs along axis 0, "
                               "channels along axis %d" % axis,
                     "rule": "cube", "axis": axis,
                     "runs_found": len(runs)},
                    runs))
    return out


# --------------------------------------------------------------------------
# candidate strategies: binary
# --------------------------------------------------------------------------

def binary_strategies(path, expected_channels, expected_runs):
    with open(path, "rb") as handle:
        raw = handle.read()
    out = []

    for endian, endian_label in (("<", "little"), (">", "big")):
        chain = aoe_probe.fortran_record_chain(raw, endian + "i")
        if not chain["consumed_all"] or chain["records"] < 2:
            continue
        lengths = chain["distinct_record_lengths"]
        if len(lengths) != 1:
            continue
        length = lengths[0]["length"]
        records = []
        offset = 0
        while offset < len(raw):
            offset += 4
            records.append(raw[offset:offset + length])
            offset += length + 4

        for itemsize, code in ((4, "f"), (8, "d")):
            if length % itemsize:
                continue
            per_record = length // itemsize
            with np.errstate(over="ignore", invalid="ignore"):
                decoded = [np.frombuffer(chunk, dtype=np.dtype(endian + code))
                           .astype(np.float64) for chunk in records]

            # One record per sample: the record is a row of channels.
            block = np.vstack(decoded)
            out.append(Strategy(
                "binary:fortran_row_%s_%d" % (endian_label, itemsize),
                {"kind": "fortran_unformatted",
                 "endianness": endian_label, "word_bytes": itemsize,
                 "record_layout": "one record per sample (row of channels)",
                 "field_count": per_record, "records": len(records),
                 "runs_found": 1},
                [ParsedRun("1", [block[:, j] for j in range(block.shape[1])])]))

            # One record per channel: the record is a whole column.
            if expected_channels and len(records) % expected_channels == 0:
                per_run = len(records) // max(1, expected_runs or 1)
                if per_run == expected_channels:
                    runs = []
                    for index in range(expected_runs or 1):
                        chunk = decoded[index * per_run:(index + 1) * per_run]
                        runs.append(ParsedRun(str(index + 1), chunk))
                    out.append(Strategy(
                        "binary:fortran_col_%s_%d" % (endian_label, itemsize),
                        {"kind": "fortran_unformatted",
                         "endianness": endian_label, "word_bytes": itemsize,
                         "record_layout": "one record per channel "
                                          "(column-major)",
                         "field_count": expected_channels,
                         "records": len(records),
                         "runs_found": len(runs)},
                        runs))

    # Headerless raw arrays: the twin's channel count is the only hint there is.
    if expected_channels:
        for endian, endian_label in (("<", "little"), (">", "big")):
            for itemsize, code in ((4, "f"), (8, "d")):
                if len(raw) % itemsize:
                    continue
                total = len(raw) // itemsize
                if total % expected_channels:
                    continue
                rows = total // expected_channels
                if rows < 2:
                    continue
                with np.errstate(over="ignore", invalid="ignore"):
                    flat = np.frombuffer(
                        raw, dtype=np.dtype(endian + code)).astype(np.float64)
                for order in ("row_major", "column_major"):
                    if order == "row_major":
                        block = flat.reshape(rows, expected_channels)
                        columns = [block[:, j]
                                   for j in range(expected_channels)]
                    else:
                        block = flat.reshape(expected_channels, rows)
                        columns = [block[j] for j in range(expected_channels)]
                    out.append(Strategy(
                        "binary:raw_%s_%d_%s" % (endian_label, itemsize, order),
                        {"kind": "raw_binary", "endianness": endian_label,
                         "word_bytes": itemsize, "order": order,
                         "field_count": expected_channels,
                         "samples": rows, "runs_found": 1},
                        [ParsedRun("1", columns)]))
    return out


def candidate_strategies(path, twin_runs, max_bytes):
    with open(path, "rb") as handle:
        head = handle.read(512)
    magic = aoe_probe.sniff_magic(head)
    channels = max((len(v) for v in twin_runs.values()), default=0)
    runs = len(twin_runs)

    if magic == "hdf5":
        return hdf5_strategies(path)
    if aoe_probe.looks_textual(head):
        return text_strategies(path, max_bytes)
    return binary_strategies(path, channels, runs)


# --------------------------------------------------------------------------
# matching columns to channels by value
# --------------------------------------------------------------------------

def fingerprint(values, count=FINGERPRINT_SAMPLES):
    """A fixed-length sample of an array, taken at fractional positions.

    Two arrays of the same length holding the same quantity produce the same
    fingerprint; the comparison then costs a dot product instead of a full
    sweep, which matters when 500 columns meet 500 channels.
    """
    size = values.size
    if size == 0:
        return np.zeros(count)
    index = (np.linspace(0.0, 1.0, count) * (size - 1)).astype(np.int64)
    sample = values[index].astype(np.float64)
    return np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0)


def _normalise_rows(matrix):
    matrix = np.nan_to_num(np.asarray(matrix, dtype=np.float64),
                           nan=0.0, posinf=0.0, neginf=0.0)
    centred = matrix - matrix.mean(axis=1, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.linalg.norm(centred, axis=1, keepdims=True)
    norms = np.nan_to_num(norms, nan=0.0, posinf=0.0, neginf=0.0)
    norms[norms == 0.0] = 1.0
    return centred / norms


def relate(column, channel, tolerance):
    """How a column relates to a channel: identical, scaled, affine, or not.

    Reporting the relationship rather than just a verdict is what lets a unit
    difference be seen for what it is instead of being read as a wrong column.
    """
    count = min(column.size, channel.size)
    if count == 0:
        return None
    a = column[:count]
    b = channel[:count]
    finite = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(finite) < max(2, count // 100):
        return None
    x = a[finite]
    y = b[finite]

    # A wrong binary interpretation decodes to enormous or denormal numbers;
    # overflow here is a verdict, not an error.
    errstate = np.errstate(all="ignore")
    errstate.__enter__()
    try:
        return _relate_inner(x, y, column, channel, tolerance)
    finally:
        errstate.__exit__(None, None, None)


def _relate_inner(x, y, column, channel, tolerance):
    scale = np.maximum(np.abs(x), np.abs(y))
    scale[scale == 0.0] = 1.0
    worst = float((np.abs(x - y) / scale).max())
    if not np.isfinite(worst):
        return None
    if worst <= tolerance:
        return {"relation": "identical", "max_rel_diff": worst,
                "length_equal": bool(column.size == channel.size)}

    if x.std() == 0.0 or y.std() == 0.0:
        return None
    # y = gain * x + offset, solved in closed form on the finite pairs.
    # polyfit would do the same thing and warn about conditioning on the wild
    # inputs a wrong binary interpretation produces.
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    variance = float(((x - x_mean) ** 2).mean())
    if variance == 0.0 or not np.isfinite(variance):
        return None
    gain = float(((x - x_mean) * (y - y_mean)).mean() / variance)
    offset = y_mean - gain * x_mean
    if not (np.isfinite(gain) and np.isfinite(offset)) or gain == 0.0:
        return None
    residual = np.abs(y - (gain * x + offset))
    magnitude = np.maximum(np.abs(y), 1e-300)
    worst_affine = float((residual / np.maximum(magnitude, 1e-12)).max())
    if worst_affine > tolerance:
        return None
    # An offset that is negligible against the channel's own magnitude is the
    # fit's noise, not a real shift: calling it affine would invent a constant.
    negligible = abs(offset) <= 1e-6 * float(np.abs(y).max() or 1.0)
    relation = {
        "relation": "scaled" if negligible else "affine",
        "gain": float(gain),
        "offset": float(offset),
        "max_rel_diff": worst_affine,
        "length_equal": bool(column.size == channel.size),
    }
    label = name_factor(float(gain))
    if label:
        relation["looks_like"] = label
    return relation


def _constant_value(values):
    """The single value an array holds, or None if it holds more than one.

    Channels that are constant through a run -- a mass setting, a
    configuration flag, a reference point -- are invisible to correlation,
    because a flat line correlates with nothing.  They need their own pass.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    first = float(finite[0])
    if float(np.abs(finite - first).max()) > 0.0:
        return None
    return first


def match_run(twin_channels, columns, tolerance):
    """Assign each column the twin channel it reproduces, one to one."""
    names = list(twin_channels)
    if not names or not columns:
        return {}, list(names), list(range(len(columns)))

    # Garbage interpretations of binary produce infinities; they should score
    # zero, not raise.
    channel_fp = _normalise_rows(
        np.vstack([fingerprint(twin_channels[name]) for name in names]))
    column_fp = _normalise_rows(
        np.vstack([fingerprint(column) for column in columns]))
    correlation = np.abs(column_fp.dot(channel_fp.T))

    order = np.argsort(-correlation, axis=1)
    taken = set()
    matches = {}

    # Two passes, because channels that are affine functions of each other are
    # perfectly correlated and would otherwise be assigned arbitrarily: a run's
    # ALTITUDE and MACH can both be linear in TIME.  Every column that
    # reproduces a channel exactly is settled first; only what is left over is
    # allowed to match through a scale or an offset.
    for exact_only in (True, False):
        for index in range(len(columns)):
            if index in matches:
                continue
            for rank in range(min(8, len(names))):
                candidate = int(order[index, rank])
                if names[candidate] in taken:
                    continue
                if correlation[index, candidate] < 0.99:
                    break
                relation = relate(columns[index],
                                  twin_channels[names[candidate]], tolerance)
                if relation is None:
                    continue
                if exact_only and relation["relation"] != "identical":
                    continue
                matches[index] = (names[candidate], relation)
                taken.add(names[candidate])
                break

    # Third pass: constants, matched on the value they hold rather than on a
    # shape they do not have.  Two channels holding the same constant in this
    # run are genuinely indistinguishable here; the vote across runs in
    # canonical_map settles most of those, and what survives is flagged.
    constant_channels = []
    for name in names:
        if name in taken:
            continue
        value = _constant_value(twin_channels[name])
        if value is not None:
            constant_channels.append((name, value))

    for index in range(len(columns)):
        if index in matches or not constant_channels:
            continue
        value = _constant_value(columns[index])
        if value is None:
            continue
        scale = max(1.0, abs(value))
        hits = [(name, other) for name, other in constant_channels
                if name not in taken and abs(other - value) <= tolerance * scale]
        if not hits:
            continue
        name, _ = hits[0]
        relation = {"relation": "identical", "max_rel_diff": 0.0,
                    "constant": True,
                    "length_equal": bool(
                        columns[index].size == twin_channels[name].size)}
        if len(hits) > 1:
            relation["ambiguous_with"] = len(hits) - 1
        matches[index] = (name, relation)
        taken.add(name)

    unmatched_channels = [name for name in names if name not in taken]
    unmatched_columns = [i for i in range(len(columns)) if i not in matches]
    return matches, unmatched_channels, unmatched_columns


def score_strategy(strategy, twin_runs, tolerance):
    """Grade a strategy by how much of the twin it reproduces."""
    twin_labels = list(twin_runs)
    used = set()
    per_run = []
    total = 0.0

    for parsed in strategy.runs:
        best = None
        for label in twin_labels:
            if label in used:
                continue
            matches, missing, spare = match_run(
                twin_runs[label], parsed.columns, tolerance)
            exact = sum(1 for _, (_, rel) in matches.items()
                        if rel["relation"] == "identical")
            converted = len(matches) - exact
            value = exact + 0.75 * converted
            if best is None or value > best[0]:
                best = (value, label, matches, missing, spare)
        if best is None:
            continue
        value, label, matches, missing, spare = best
        if value <= 0:
            continue
        used.add(label)
        total += value
        per_run.append({
            "parsed_run": parsed.label,
            "twin_run": label,
            "matched": len(matches),
            "identical": sum(1 for _, (_, rel) in matches.items()
                             if rel["relation"] == "identical"),
            "unmatched_channels": missing,
            "unmatched_columns": spare,
            "matches": matches,
            "names": parsed.names,
        })
    return total, per_run


# --------------------------------------------------------------------------
# generating a reader
# --------------------------------------------------------------------------

READER_PREAMBLE = '''# -*- coding: utf-8 -*-
"""Reader generated by aoe_infer.py on {when}.

The strategy below was not guessed: it was chosen because it reproduces the
.aoe twin, and the column-to-channel map was recovered by matching values,
not names.  {summary}

Edit it freely -- it is a plain file.  Re-grade it at any time with

    python aoe_compare.py --aoe TWIN.aoe --other FILE --reader {module}:read
"""

from __future__ import annotations

import re

import numpy as np

# what each column (or dataset key) holds, as verified against the twin
COLUMNS = {columns}

# the same keys -> (gain, offset), applied as channel = gain * column + offset
CONVERSIONS = {conversions}
'''

TEXT_BODY = '''

def _to_float(token):
    token = token.strip()
    if not token:
        return float("nan")
    try:
        return float(token)
    except ValueError:
        pass
    try:
        return float(token.replace("D", "E").replace("d", "e"))
    except ValueError:
        pass
    try:
        return float(re.sub(r"(\\d)([+-]\\d{{1,3}})$", r"\\1E\\2", token))
    except ValueError:
        return float("nan")


def _split(line):
    {split_expression}


def _is_data(cells):
    if len(cells) != {width}:
        return False
    good = 0
    for cell in cells:
        text = cell.strip().strip("'\\"")
        if not text:
            continue
        try:
            float(text.replace("D", "E").replace("d", "e"))
            good += 1
        except ValueError:
            return False
    return good >= max(2, int(0.6 * {width}))


def _label(line):
    tokens = [t for t in line.replace(":", " ").split() if t]
    if tokens and re.match(r"^[A-Za-z0-9_\\-.]+$", tokens[-1]):
        return tokens[-1]
    return None


def read(path):
    """Return {{run_name: {{channel_name: np.ndarray}}}}."""
    runs = {{}}
    block = []
    label = None
    order = [0]

    def flush():
        if not block:
            return
        name = label or str(len(runs) + 1)
        channels = {{}}
        for index, channel in COLUMNS.items():
            values = np.asarray([row[index] for row in block],
                                dtype=np.float64)
            gain, offset = CONVERSIONS.get(index, (1.0, 0.0))
            if gain != 1.0 or offset != 0.0:
                values = gain * values + offset
            channels[channel] = values
        runs[name] = channels
        order[0] += 1

    with open(path, "r", encoding="{encoding}", errors="replace") as handle:
        for line in handle:
            stripped = line.rstrip("\\n")
            if not stripped.strip():
                continue
            cells = _split(stripped)
            if _is_data(cells):
                block.append([_to_float(cell) for cell in cells])
                continue
            flush()
            block = []
            label = _label(stripped)

    flush()
    return runs
'''

BINARY_BODY = '''

def read(path):
    """Return {{run_name: {{channel_name: np.ndarray}}}}."""
    with open(path, "rb") as handle:
        raw = handle.read()

    dtype = np.dtype("{endian}{code}")
    {decode}

    channels = {{}}
    for index, channel in COLUMNS.items():
        values = columns[index].astype(np.float64)
        gain, offset = CONVERSIONS.get(index, (1.0, 0.0))
        if gain != 1.0 or offset != 0.0:
            values = gain * values + offset
        channels[channel] = values
    return {{"1": channels}}
'''

HDF5_RULE_BODY = '''

# How this exporter encodes run and channel in a dataset's location.
RULE = "{rule}"
RULE_TOKENS = {tokens}


def _key(path):
    """(run key, channel key) for one dataset path, or None if it is neither."""
    parent, _, leaf = path.rpartition("/")
    if RULE == "parent_group":
        return (parent or "/"), leaf
    if RULE.startswith("leaf_tail"):
        tokens = leaf.split("_")
        if len(tokens) <= RULE_TOKENS:
            return None
        return "_".join(tokens[-RULE_TOKENS:]), "_".join(tokens[:-RULE_TOKENS])
    if RULE.startswith("ancestor"):
        parts = path.strip("/").split("/")
        if len(parts) <= RULE_TOKENS:
            return None
        return "/".join(parts[:RULE_TOKENS]), "/".join(parts[RULE_TOKENS:])
    return "1", leaf


def read(path):
    """Return {{run_name: {{channel_name: np.ndarray}}}}."""
    import h5py

    runs = {{}}
    with h5py.File(path, "r") as handle:
        found = []

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 1 \
                    and obj.dtype.kind in "fiu":
                found.append("/" + name)
            return None

        handle.visititems(visit)

        for dataset_path in sorted(found):
            keyed = _key(dataset_path)
            if keyed is None:
                continue
            run_key, channel_key = keyed
            channel = COLUMNS.get(channel_key)
            if channel is None:
                # Not one of the identified channels: a metadata vector, or a
                # channel this reader was never shown.  Skipping it is right;
                # the conformance report is where a missing channel shows up.
                continue
            values = np.asarray(handle[dataset_path][:], dtype=np.float64)
            gain, offset = CONVERSIONS.get(channel_key, (1.0, 0.0))
            if gain != 1.0 or offset != 0.0:
                values = gain * values + offset
            runs.setdefault(run_key, {{}})[channel] = values
    return runs
'''

HDF5_TABLE_BODY = '''

def read(path):
    """Return {{run_name: {{channel_name: np.ndarray}}}}."""
    import h5py

    runs = {{}}
    with h5py.File(path, "r") as handle:
        tables = []

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                tables.append(("/" + name, obj))
            return None

        handle.visititems(visit)
        for name, obj in sorted(tables):
            block = np.asarray(obj[:], dtype=np.float64)
            if {axis} == 1:
                columns = [block[:, j] for j in range(block.shape[1])]
            else:
                columns = [block[j] for j in range(block.shape[0])]
            channels = {{}}
            for index, channel in COLUMNS.items():
                values = columns[index]
                gain, offset = CONVERSIONS.get(index, (1.0, 0.0))
                if gain != 1.0 or offset != 0.0:
                    values = gain * values + offset
                channels[channel] = values
            runs[name.split("/")[-1]] = channels
    return runs
'''


def _split_expression(description):
    if description["kind"] == "fixed_width_text":
        slices = [(f["start"], f["width"]) for f in description["fields"]]
        return ("return [line[start:start + width] for start, width in %r]"
                % (slices,))
    delimiter = description["delimiter"]
    if delimiter == "whitespace":
        return "return line.split()"
    literal = {"comma": ",", "semicolon": ";", "tab": "\\t",
               "pipe": "|", "colon": ":"}[delimiter]
    return 'return line.split("%s")' % literal


def generate_reader(strategy, column_map, conversions, module_name, summary):
    description = strategy.description
    header = READER_PREAMBLE.format(
        when=datetime.datetime.utcnow().isoformat(timespec="seconds"),
        summary=summary,
        module="readers.%s" % module_name,
        columns=repr(column_map),
        conversions=repr(conversions))

    if description["kind"] in ("delimited_text", "fixed_width_text"):
        width = description["field_count"]
        return header + TEXT_BODY.format(
            split_expression=_split_expression(description),
            width=width,
            encoding=description.get("encoding", "utf-8"))

    if description["kind"] in ("fortran_unformatted", "raw_binary"):
        endian = "<" if description["endianness"] == "little" else ">"
        code = "f4" if description["word_bytes"] == 4 else "f8"
        if description["kind"] == "raw_binary":
            count = description["field_count"]
            if description["order"] == "row_major":
                decode = ("flat = np.frombuffer(raw, dtype=dtype)\n"
                          "    block = flat.reshape(-1, %d)\n"
                          "    columns = [block[:, j] for j in range(%d)]"
                          % (count, count))
            else:
                decode = ("flat = np.frombuffer(raw, dtype=dtype)\n"
                          "    block = flat.reshape(%d, -1)\n"
                          "    columns = [block[j] for j in range(%d)]"
                          % (count, count))
        else:
            decode = (
                "records = []\n"
                "    offset = 0\n"
                "    marker = np.dtype('%si4')\n"
                "    while offset < len(raw):\n"
                "        length = int(np.frombuffer(raw, dtype=marker,\n"
                "                                   count=1, offset=offset)[0])\n"
                "        offset += 4\n"
                "        records.append(np.frombuffer(raw, dtype=dtype,\n"
                "                                     count=length // dtype.itemsize,\n"
                "                                     offset=offset))\n"
                "        offset += length + 4\n"
                "    block = np.vstack(records)\n"
                "    columns = [block[:, j] for j in range(block.shape[1])]"
                % endian)
        return header + BINARY_BODY.format(
            endian=endian, code=code, decode=decode)

    if description["kind"] == "hdf5":
        if description.get("rule") in ("table", "cube"):
            return header + HDF5_TABLE_BODY.format(axis=description["axis"])
        return header + HDF5_RULE_BODY.format(
            rule=description["rule"], tokens=description.get("rule_tokens", 0))

    raise ValueError("no template for %s" % description["kind"])


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def infer(twin_runs, path, tolerance=DEFAULT_TOLERANCE,
          max_bytes=aoe_probe.DEFAULT_MAX_BYTES):
    """Try every plausible parse and return the strategies, best first."""
    strategies = [s for s in candidate_strategies(path, twin_runs, max_bytes)
                  if s.viable]
    graded = []
    for strategy in strategies:
        score, per_run = score_strategy(strategy, twin_runs, tolerance)
        graded.append((score, strategy, per_run))
    graded.sort(key=lambda item: -item[0])
    return graded


def canonical_map(per_run, use_constants):
    """The column-to-channel map, and the conversions, agreed across runs."""
    votes = {}
    for entry in per_run:
        for index, (channel, relation) in entry["matches"].items():
            votes.setdefault(index, {}).setdefault(channel, []).append(relation)

    columns = {}
    conversions = {}
    details = []
    for index in sorted(votes):
        channel, relations = max(votes[index].items(),
                                 key=lambda kv: len(kv[1]))
        columns[index] = channel
        relation = relations[0]
        entry = {
            "column": index,
            "channel": channel,
            "relation": relation["relation"],
            "runs_agreeing": len(relations),
            "max_rel_diff_exponent": aoe_compare.magnitude_bucket(
                max(r["max_rel_diff"] for r in relations)),
        }
        if relation.get("constant"):
            entry["constant"] = True
        ambiguous = sum(1 for r in relations if r.get("ambiguous_with"))
        if ambiguous:
            entry["ambiguous_in_runs"] = ambiguous
        if len(votes[index]) > 1:
            entry["disputed_by_runs"] = {
                candidate: len(seen)
                for candidate, seen in votes[index].items()}
        if relation["relation"] != "identical":
            if use_constants:
                conversions[index] = (relation["gain"], relation["offset"])
                entry["gain"] = relation["gain"]
                entry["offset"] = relation["offset"]
            if "looks_like" in relation:
                entry["looks_like"] = relation["looks_like"]
        details.append(entry)
    return columns, conversions, details


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Work out how to read an unknown format by verifying "
                    "candidate parses against an .aoe export of the same "
                    "simulation.")
    parser.add_argument("--aoe", required=True, help="the .aoe twin")
    parser.add_argument("--other", required=True,
                        help="the same data in the unknown format")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="relative agreement treated as a match "
                             "(default 1e-5; .aoe stores float32)")
    parser.add_argument("--tier",
                        choices=[aoe_probe.TIER_STRUCTURE,
                                 aoe_probe.TIER_IDENTIFIERS],
                        default=aoe_probe.TIER_IDENTIFIERS,
                        help="whether the result names channels, runs and "
                             "dataset keys (default) or masks them as "
                             "aoe_probe.py does; the structural profile is "
                             "masked either way")
    parser.add_argument("--no-constants", action="store_true",
                        help="omit unit-conversion gains and offsets from the "
                             "report (they are the only file-derived numbers "
                             "in it)")
    parser.add_argument("--reader-name",
                        help="name for the generated reader module "
                             "(default: generated_<file stem>_reader)")
    parser.add_argument("-o", "--output", help="write the JSON report here")
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args(argv)

    for path in (args.aoe, args.other):
        if not os.path.isfile(path):
            parser.error("not a file: %s" % path)

    twin_runs = aoe_compare.load_aoe(args.aoe)
    if not twin_runs:
        parser.error("the .aoe twin holds no runs")

    graded = infer(twin_runs, args.other, args.tolerance)
    report = {
        "generated_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "tolerance": args.tolerance,
        "twin": {
            "runs": len(twin_runs),
            "channels": len(next(iter(twin_runs.values()))),
        },
        "structure": aoe_probe.profile_path(
            args.other, aoe_probe.TIER_STRUCTURE, False,
            aoe_probe.DEFAULT_MAX_BYTES),
        "strategies_tried": [
            {"key": strategy.key, "score": round(score, 2),
             "description": strategy.description}
            for score, strategy, _ in graded
        ],
    }

    if not graded or graded[0][0] <= 0:
        report["verdict"] = "no strategy reproduced the twin"
        _emit(report, args)
        return 1

    score, strategy, per_run = graded[0]
    columns, conversions, details = canonical_map(
        per_run, not args.no_constants)

    stem = re.sub(r"[^A-Za-z0-9]+", "_",
                  os.path.splitext(os.path.basename(args.other))[0]).strip("_")
    module_name = args.reader_name or ("generated_%s_reader" % stem.lower())
    summary = ("%d of the twin's %d channels were identified by value."
               % (len(columns), report["twin"]["channels"]))
    keys = per_run[0].get("names") if per_run else None
    if strategy.description.get("kind") == "hdf5" and \
            strategy.description.get("rule") not in ("table", "cube") and keys:
        # The 1-D layouts address a channel by a key in the dataset's name,
        # not by a column number, so the generated reader is given that key.
        columns = {keys[index]: channel for index, channel in columns.items()
                   if index < len(keys)}
        conversions = {keys[index]: value
                       for index, value in conversions.items()
                       if index < len(keys)}
    source = generate_reader(strategy, columns, conversions, module_name,
                             summary)
    readers_dir = os.path.join(HERE, "readers")
    if not os.path.isdir(readers_dir):
        os.makedirs(readers_dir)
    reader_path = os.path.join(readers_dir, module_name + ".py")
    with open(reader_path, "w", encoding="utf-8") as handle:
        handle.write(source)

    report["winner"] = {
        "key": strategy.key,
        "description": strategy.description,
        "score": round(score, 2),
        "runs": [{k: v for k, v in entry.items() if k != "matches"}
                 for entry in per_run],
        "column_map": details,
        "generated_reader": "readers/%s.py" % module_name,
    }

    # Grade the generated reader the same way aoe_compare would, by running it.
    conformance = None
    try:
        sys.path.insert(0, HERE)
        module = importlib.import_module("readers.%s" % module_name)
        importlib.reload(module)
        produced = aoe_compare.normalise_reader_output(module.read(args.other))
        # The unknown format labels its runs its own way -- "case_2" against
        # the twin's "2" -- so grade against the pairing inference established
        # rather than hoping the two vocabularies coincide.
        run_map = {entry["twin_run"]: entry["parsed_run"]
                   for entry in per_run}
        align = "name" if set(run_map.values()) & set(produced) else "order"
        # Grade at the tolerance the matching used, not at a stricter one.
        # The twin is an .aoe file, which stores samples as float32 and, where
        # it is a text export, with a fixed number of significant digits: a
        # reader that is exactly right still lands a few parts in a million
        # away from it, and calling that a failure would be a bug in the
        # grader rather than a finding about the reader.
        conformance = aoe_compare.compare(
            twin_runs, produced, {}, run_map, align, args.tolerance,
            aoe_compare.TIER_IDENTIFIERS, False)
        conformance.pop("comparisons", None)
    except Exception as exc:
        conformance = {"error": "%s: %s" % (type(exc).__name__, exc)}
    if args.tier == aoe_probe.TIER_STRUCTURE:
        _mask_names(report)
    report["conformance"] = conformance
    report["verdict"] = (conformance.get("verdict", "unknown")
                         if isinstance(conformance, dict) else "unknown")

    exponents = [entry["max_rel_diff_exponent"] for entry in details
                 if entry["max_rel_diff_exponent"] is not None]
    if exponents:
        worst = max(exponents)
        report["agreement"] = {
            "worst_exponent": worst,
            "float32_exponent": int(math.floor(math.log10(
                aoe_compare.FLOAT32_EPS))),
            "reading": _agreement_reading(worst),
        }

    _emit(report, args)
    return 0 if report["verdict"] == "equivalent" else 1


def _agreement_reading(worst):
    """What the residual between the two readings means.

    Three regimes, and telling them apart is most of the interpretation: at
    float32 epsilon the two sides hold the same number; a few parts per million
    is the .aoe twin's own decimal precision, which no reader can beat; beyond
    that the reader is reading something else.
    """
    if worst <= -7:
        return ("agreement at float32 epsilon: the two sides hold the same "
                "numbers, and the residual is the core's float32 storage")
    if worst <= -5:
        return ("agreement limited by the .aoe twin's decimal precision, not "
                "by the reader: a text .aoe keeps a fixed number of "
                "significant digits, so this is as close as any correct "
                "reader can come")
    return ("the two sides disagree beyond rounding; a column is likely "
            "mismatched, scaled, or read from the wrong place")


def _mask_names(report):
    """Reduce every name in the result to its shape, as the profiler does.

    The structural profile is masked whatever the tier; this covers the parts
    that come from matching -- channel names, run labels, dataset keys -- for
    when the report itself has to leave the site.
    """
    mask = aoe_probe.masked_name
    tier = aoe_probe.TIER_STRUCTURE
    winner = report.get("winner")
    if winner:
        for entry in winner.get("runs", []):
            entry["parsed_run"] = mask(str(entry["parsed_run"]), tier)
            entry["twin_run"] = mask(str(entry["twin_run"]), tier)
            entry["unmatched_channels"] = [
                mask(name, tier) for name in entry.get("unmatched_channels", [])]
            entry.pop("names", None)
        for entry in winner.get("column_map", []):
            entry["channel"] = mask(entry["channel"], tier)
            entry.pop("disputed_by_runs", None)
    conformance = report.get("conformance")
    if isinstance(conformance, dict) and "runs" in conformance:
        for key in ("only_in_aoe", "only_in_other"):
            conformance["runs"][key] = [
                mask(str(name), tier) for name in conformance["runs"][key]]
            conformance["channels"][key] = [
                mask(str(name), tier) for name in conformance["channels"][key]]


def _emit(report, args):
    payload = json.dumps(report, indent=2, default=str)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    if args.json_only:
        print(payload)
        return
    print("\n".join(render(report)))
    if args.output:
        print("\nJSON report written to %s" % args.output)


def render(report):
    lines = []
    twin = report["twin"]
    lines.append("twin      %d runs, %d channels"
                 % (twin["runs"], twin["channels"]))
    lines.append("tried     %d parse strategies"
                 % len(report["strategies_tried"]))
    for entry in report["strategies_tried"][:6]:
        lines.append("  %-34s score %s" % (entry["key"], entry["score"]))

    winner = report.get("winner")
    if not winner:
        lines.append("")
        lines.append("verdict   %s" % report.get("verdict"))
        return lines

    lines.append("")
    lines.append("winner    %s" % winner["key"])
    for key, value in sorted(winner["description"].items()):
        if key == "fields":
            value = "%d slices" % len(value)
        lines.append("  %-16s %s" % (key, value))
    lines.append("")
    lines.append("runs")
    for entry in winner["runs"]:
        lines.append("  %-10s -> twin %-10s %d matched (%d identical)"
                     % (entry["parsed_run"], entry["twin_run"],
                        entry["matched"], entry["identical"]))
        if entry["unmatched_channels"]:
            lines.append("      twin channels not found: %s"
                         % ", ".join(entry["unmatched_channels"][:10]))
        if entry["unmatched_columns"]:
            lines.append("      columns not identified:  %s"
                         % ", ".join(str(c)
                                     for c in entry["unmatched_columns"][:10]))

    lines.append("")
    lines.append("%-6s %-34s %-10s %s" % ("column", "channel", "relation",
                                          "agreement"))
    for entry in winner["column_map"]:
        relation = entry["relation"]
        if relation != "identical":
            relation = "%s x%.6g" % (relation, entry.get("gain", 1.0))
            if entry.get("looks_like"):
                relation += " (%s)" % entry["looks_like"]
        exponent = entry["max_rel_diff_exponent"]
        lines.append("%-6d %-34s %-10s %s"
                     % (entry["column"], entry["channel"][:34], relation,
                        "exact" if exponent is None else "1e%d" % exponent))

    lines.append("")
    lines.append("reader    %s" % winner["generated_reader"])
    conformance = report.get("conformance") or {}
    if "error" in conformance:
        lines.append("graded    could not run: %s" % conformance["error"])
    else:
        lines.append("graded    %d comparisons, %d failing"
                     % (conformance.get("channels", {}).get("compared", 0),
                        conformance.get("failures", 0)))
    agreement = report.get("agreement")
    if agreement:
        lines.append("residual  worst 1e%d across all channels"
                     % agreement["worst_exponent"])
        lines.append("          %s" % agreement["reading"])
    lines.append("")
    lines.append("verdict   %s" % str(report.get("verdict", "")).upper())
    return lines


if __name__ == "__main__":
    sys.exit(main())
