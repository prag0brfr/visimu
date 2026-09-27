#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conformance harness: does a candidate reader agree with the .aoe twin?

The premise is the one thing that makes this whole exercise tractable: for a
given simulation, the same data can be exported twice -- once in the ad-hoc
format, once as .aoe.  The .aoe file is then ground truth, and a candidate
reader for the ad-hoc format can be graded against it *locally*, with neither
file leaving the site.

What travels afterwards is this script's report, which describes agreement and
disagreement in the abstract: which channels one side has and the other does
not, whether the arrays are the same length, and for each channel a single
scalar saying how far apart the two readings are.  No sample is printed.

Usage
-----
    # Grade the built-in delimited-text reader against the twin:
    python aoe_compare.py --aoe RUN.aoe --other RUN.DAT --reader csv

    # Grade a reader you (or I) wrote, given as module:function
    python aoe_compare.py --aoe RUN.aoe --other RUN.DAT \
        --reader readers.example_reader:read

    # Rename channels that the two formats spell differently
    python aoe_compare.py --aoe RUN.aoe --other RUN.DAT --reader csv \
        --map channel_map.json

    # Send nothing but yes/no verdicts, no magnitudes at all
    python aoe_compare.py ... --tier structure --no-magnitudes

The reader protocol
-------------------
A reader is any callable taking a path and returning either

    {"RUN NAME": {"CHANNEL": sequence_of_floats, ...}, ...}

or an iterable of ``{"name": ..., "channels": {...}}`` mappings.  Anything
array-like works; it is converted with numpy.asarray.  ``readers/`` holds a
worked example to copy.

Exit code is 0 when the two sides agree within tolerance and 1 when they do
not, so this can be wired into a test run.
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The app's Python package lives beside this folder.  Importing it is the only
# coupling this tool has to the application, and it is read-only.
_APP_PYTHON = os.path.join(os.path.dirname(HERE), "python")
for _candidate in (HERE, _APP_PYTHON):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

try:
    import numpy as np
except ImportError:                                     # pragma: no cover
    sys.stderr.write("numpy is required: pip install numpy\n")
    raise

FLOAT32_EPS = float(np.finfo(np.float32).eps)

TIER_STRUCTURE = "structure"
TIER_IDENTIFIERS = "identifiers"


# --------------------------------------------------------------------------
# name handling
# --------------------------------------------------------------------------

_NORMALISE = re.compile(r"[\s_\-.]+")


def normalise(name):
    """Fold the spellings that mean the same channel.

    ``Total Force``, ``TOTAL_FORCE`` and ``total-force`` are one channel; the
    two exporters rarely agree on which of them to write.
    """
    return _NORMALISE.sub("", str(name)).upper()


def shape_name(name):
    """The masked form used when the report must carry no vocabulary."""
    out = []
    for ch in str(name):
        if ch.isdigit():
            out.append("9")
        elif ch.isupper():
            out.append("A")
        elif ch.islower():
            out.append("a")
        else:
            out.append(ch)
    return "".join(out)


def emit(name, tier):
    return name if tier == TIER_IDENTIFIERS else shape_name(name)


# --------------------------------------------------------------------------
# loading the .aoe twin
# --------------------------------------------------------------------------

def load_aoe(path):
    """Read every run and channel out of an .aoe file via the app's session.

    This deliberately goes through ``aoeviewer.Session`` rather than parsing
    the file here: the point of the comparison is agreement with what the
    application will actually show, not with a second implementation of it.
    """
    try:
        from aoeviewer.session import Session
    except ImportError as exc:                          # pragma: no cover
        raise SystemExit(
            "Could not import aoeviewer (%s).\n"
            "Run this from a checkout where ../python/aoeviewer exists and "
            "the core library has been built, or set PYTHONPATH." % exc)

    runs = {}
    with Session() as session:
        handle = session.add_file(path)
        names = list(session.parameters)
        for sim_index, simulation in enumerate(handle.simulations):
            channels = {}
            for name in names:
                try:
                    values = session.channel(handle.index, sim_index, name)
                except Exception:
                    continue
                if values is None:
                    continue
                # The session hands out a view into the core's memory; the
                # copy is what makes it safe to hold past the session's life.
                channels[name] = np.array(values, dtype=np.float64, copy=True)
            runs[simulation.name] = channels
    return runs


# --------------------------------------------------------------------------
# candidate readers
# --------------------------------------------------------------------------

def builtin_csv_reader(path):
    """A delimited-text reader good enough to bootstrap the comparison.

    It sniffs the delimiter, takes the last non-numeric line before the data
    as the header, and treats the whole file as one run.  Most ad-hoc text
    formats need more than this -- which is precisely what the comparison is
    for -- but it makes the harness runnable on day one.
    """
    import csv as _csv

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        sample = fh.read(65536)
        fh.seek(0)
        try:
            dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t| ")
            delimiter = dialect.delimiter
        except Exception:
            delimiter = None
        rows = [line.split(delimiter) if delimiter else line.split()
                for line in fh.read().splitlines() if line.strip()]

    header = None
    data_start = 0
    for index, row in enumerate(rows[:200]):
        numeric = sum(1 for cell in row if _is_number(cell))
        if numeric >= max(2, 0.6 * len(row)):
            data_start = index
            break
        header = row
    if header is None:
        header = ["c%d" % i for i in range(len(rows[data_start]))]
    header = [cell.strip() for cell in header]

    width = len(header)
    columns = [[] for _ in range(width)]
    for row in rows[data_start:]:
        if len(row) < width:
            continue
        for i in range(width):
            columns[i].append(_to_float(row[i]))
    name = os.path.splitext(os.path.basename(path))[0]
    return {name: {header[i]: np.asarray(columns[i], dtype=np.float64)
                   for i in range(width)}}


_NUMBER = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdDqQ][+-]?\d+|[+-]\d{1,3})?$")


def _is_number(cell):
    return bool(_NUMBER.match(cell.strip()))


def _to_float(cell):
    cell = cell.strip()
    if not cell:
        return float("nan")
    try:
        return float(cell)
    except ValueError:
        pass
    # Fortran writes 1.0D+03, and sometimes drops the exponent letter entirely.
    fixed = cell.replace("D", "E").replace("d", "e").replace("Q", "E")
    try:
        return float(fixed)
    except ValueError:
        pass
    fixed = re.sub(r"(\d)([+-]\d{1,3})$", r"\1E\2", cell)
    try:
        return float(fixed)
    except ValueError:
        return float("nan")


def resolve_reader(spec):
    if spec == "csv":
        return builtin_csv_reader
    if ":" not in spec:
        raise SystemExit("--reader must be 'csv' or 'module:function', got %r"
                         % spec)
    module_name, attribute = spec.split(":", 1)
    module = importlib.import_module(module_name)
    reader = getattr(module, attribute, None)
    if not callable(reader):
        raise SystemExit("%s has no callable %r" % (module_name, attribute))
    return reader


def normalise_reader_output(result):
    """Accept either of the two shapes the reader protocol allows."""
    if isinstance(result, dict):
        return {str(key): {str(k): np.asarray(v, dtype=np.float64)
                           for k, v in value.items()}
                for key, value in result.items()}
    runs = {}
    for index, item in enumerate(result):
        name = str(item.get("name", "run%d" % index))
        runs[name] = {str(k): np.asarray(v, dtype=np.float64)
                      for k, v in item["channels"].items()}
    return runs


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def magnitude_bucket(value):
    """The decimal exponent of a difference, instead of the difference.

    ``1e-7`` becomes -7.  Enough to tell a rounding artefact from a wrong
    column, and not enough to reconstruct anything.
    """
    if value is None or not math.isfinite(value) or value == 0.0:
        return None
    return int(math.floor(math.log10(abs(value))))


def compare_arrays(left, right, tolerance, magnitudes):
    """Grade one channel: same length, same finiteness pattern, same numbers."""
    result = {
        "aoe_length": int(left.size),
        "other_length": int(right.size),
        "length_equal": bool(left.size == right.size),
    }
    count = min(left.size, right.size)
    if count == 0:
        result["comparable"] = False
        return result
    a = left[:count]
    b = right[:count]
    a_nan = ~np.isfinite(a)
    b_nan = ~np.isfinite(b)
    result["nan_mismatch"] = int(np.count_nonzero(a_nan != b_nan))
    both = ~(a_nan | b_nan)
    result["compared_elements"] = int(np.count_nonzero(both))
    if not np.any(both):
        result["comparable"] = False
        return result
    diff = np.abs(a[both] - b[both])
    scale = np.maximum(np.abs(a[both]), np.abs(b[both]))
    scale[scale == 0.0] = 1.0
    relative = diff / scale
    max_abs = float(diff.max())
    max_rel = float(relative.max())
    result["comparable"] = True
    result["within_tolerance"] = bool(max_rel <= tolerance)
    result["within_float32_eps"] = bool(max_rel <= FLOAT32_EPS)
    result["worst_index_fraction"] = round(
        float(int(np.argmax(relative)) / float(max(1, result["compared_elements"]))), 4)
    if magnitudes:
        result["max_abs_diff"] = max_abs
        result["max_rel_diff"] = max_rel
    else:
        result["max_abs_diff_exponent"] = magnitude_bucket(max_abs)
        result["max_rel_diff_exponent"] = magnitude_bucket(max_rel)
    return result


def align(left_names, right_names, mapping):
    """Pair the two sides' names, via the map first and then by normal form."""
    pairs = []
    right_by_norm = {}
    for name in right_names:
        right_by_norm.setdefault(normalise(name), name)
    used = set()
    for name in left_names:
        target = mapping.get(name)
        if target is None:
            target = right_by_norm.get(normalise(name))
        if target is not None and target in right_names and target not in used:
            pairs.append((name, target))
            used.add(target)
    only_left = [n for n in left_names if n not in dict(pairs)]
    only_right = [n for n in right_names if n not in used]
    return pairs, only_left, only_right


def compare(aoe_runs, other_runs, mapping, run_mapping, align_runs,
            tolerance, tier, magnitudes):
    aoe_names = list(aoe_runs)
    other_names = list(other_runs)

    if align_runs == "order" and len(aoe_names) == len(other_names):
        run_pairs = list(zip(aoe_names, other_names))
        runs_only_aoe = []
        runs_only_other = []
    else:
        run_pairs, runs_only_aoe, runs_only_other = align(
            aoe_names, other_names, run_mapping)

    channel_results = {}
    channels_only_aoe = set()
    channels_only_other = set()

    for aoe_run, other_run in run_pairs:
        left = aoe_runs[aoe_run]
        right = other_runs[other_run]
        pairs, only_left, only_right = align(
            list(left), list(right), mapping)
        channels_only_aoe.update(only_left)
        channels_only_other.update(only_right)
        for aoe_channel, other_channel in pairs:
            entry = compare_arrays(left[aoe_channel], right[other_channel],
                                   tolerance, magnitudes)
            entry["run"] = emit(aoe_run, tier)
            entry["channel"] = emit(aoe_channel, tier)
            entry["other_channel"] = emit(other_channel, tier)
            entry["renamed"] = bool(normalise(aoe_channel) !=
                                    normalise(other_channel))
            channel_results.setdefault(
                normalise(aoe_channel), []).append(entry)

    flat = [entry for entries in channel_results.values() for entry in entries]
    failures = [e for e in flat
                if not e.get("length_equal")
                or not e.get("comparable")
                or not e.get("within_tolerance")
                or e.get("nan_mismatch")]

    return {
        "runs": {
            "aoe_count": len(aoe_names),
            "other_count": len(other_names),
            "matched": len(run_pairs),
            "only_in_aoe": [emit(n, tier) for n in runs_only_aoe],
            "only_in_other": [emit(n, tier) for n in runs_only_other],
            "aligned_by": align_runs,
        },
        "channels": {
            "compared": len(flat),
            "distinct": len(channel_results),
            "only_in_aoe": sorted(emit(n, tier) for n in channels_only_aoe),
            "only_in_other": sorted(emit(n, tier) for n in channels_only_other),
        },
        "tolerance": tolerance,
        "float32_eps": FLOAT32_EPS,
        "comparisons": flat,
        "failures": len(failures),
        "verdict": "equivalent" if not failures and flat else "differs",
    }


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def render(report):
    lines = []
    runs = report["runs"]
    lines.append("runs      %d in .aoe, %d in the other file, %d matched (%s)"
                 % (runs["aoe_count"], runs["other_count"], runs["matched"],
                    runs["aligned_by"]))
    if runs["only_in_aoe"]:
        lines.append("          only in .aoe:   %s"
                     % ", ".join(runs["only_in_aoe"][:12]))
    if runs["only_in_other"]:
        lines.append("          only in other:  %s"
                     % ", ".join(runs["only_in_other"][:12]))

    channels = report["channels"]
    lines.append("channels  %d comparisons over %d distinct channels"
                 % (channels["compared"], channels["distinct"]))
    if channels["only_in_aoe"]:
        lines.append("          missing from the other file: %s"
                     % ", ".join(channels["only_in_aoe"][:20]))
    if channels["only_in_other"]:
        lines.append("          not present in the .aoe:     %s"
                     % ", ".join(channels["only_in_other"][:20]))

    lines.append("")
    lines.append("%-28s %-10s %8s %8s  %s"
                 % ("channel", "run", "n(aoe)", "n(other)", "agreement"))
    for entry in report["comparisons"]:
        if "max_rel_diff" in entry:
            agreement = "rel %.3e" % entry["max_rel_diff"]
        elif "max_rel_diff_exponent" in entry:
            exponent = entry["max_rel_diff_exponent"]
            agreement = "rel 1e%s" % exponent if exponent is not None else "exact"
        else:
            agreement = "not comparable"
        flags = []
        if not entry.get("length_equal"):
            flags.append("LENGTH")
        if entry.get("nan_mismatch"):
            flags.append("NAN x%d" % entry["nan_mismatch"])
        if entry.get("comparable") and not entry.get("within_tolerance"):
            flags.append("OUT OF TOLERANCE")
        if entry.get("renamed"):
            flags.append("renamed")
        lines.append("%-28s %-10s %8d %8d  %-14s %s"
                     % (str(entry["channel"])[:28], str(entry["run"])[:10],
                        entry["aoe_length"], entry["other_length"],
                        agreement, " ".join(flags)))
    lines.append("")
    lines.append("verdict   %s (%d failing comparisons)"
                 % (report["verdict"].upper(), report["failures"]))
    return lines


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def load_mapping(path):
    if not path:
        return {}, {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("channels", {}), payload.get("runs", {})


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Grade a candidate reader for an ad-hoc format against "
                    "an .aoe export of the same simulation.")
    parser.add_argument("--aoe", required=True, help="the .aoe twin")
    parser.add_argument("--other", required=True,
                        help="the same data in the ad-hoc format")
    parser.add_argument("--reader", default="csv",
                        help="'csv' for the built-in delimited reader, or "
                             "module:function (default: csv)")
    parser.add_argument("--map", dest="mapping",
                        help="JSON with {\"channels\": {...}, \"runs\": {...}} "
                             "renames, .aoe name -> other-file name")
    parser.add_argument("--align-runs", choices=["name", "order"],
                        default="name",
                        help="pair runs by name (default) or by position")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="maximum relative difference treated as "
                             "agreement (default 1e-6, ~float32 round-trip)")
    parser.add_argument("--tier", choices=[TIER_STRUCTURE, TIER_IDENTIFIERS],
                        default=TIER_IDENTIFIERS,
                        help="whether the report names channels (default) or "
                             "masks them, as aoe_probe.py does")
    parser.add_argument("--no-magnitudes", action="store_true",
                        help="report the exponent of each difference instead "
                             "of the difference itself")
    parser.add_argument("-o", "--output", help="write the JSON report here")
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args(argv)

    for path in (args.aoe, args.other):
        if not os.path.isfile(path):
            parser.error("not a file: %s" % path)

    channel_map, run_map = load_mapping(args.mapping)
    reader = resolve_reader(args.reader)

    aoe_runs = load_aoe(args.aoe)
    other_runs = normalise_reader_output(reader(args.other))

    report = {
        "generated_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "tier": args.tier,
        "reader": args.reader,
    }
    report.update(compare(aoe_runs, other_runs, channel_map, run_map,
                          args.align_runs, args.tolerance, args.tier,
                          not args.no_magnitudes))

    payload = json.dumps(report, indent=2, default=str)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    if args.json_only:
        print(payload)
    else:
        print("\n".join(render(report)))
        if args.output:
            print("\nJSON report written to %s" % args.output)
    return 0 if report["verdict"] == "equivalent" else 1


if __name__ == "__main__":
    sys.exit(main())
