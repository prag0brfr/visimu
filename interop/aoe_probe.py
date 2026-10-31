#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structural profiler for unknown result-file formats.

The problem this solves: visimu needs to read a proprietary format, but no
sample of that format may leave the site where it lives.  This script reads
such a file locally and writes a report that describes its *structure* without
ever reproducing its *content*.  The report is what travels.

What the report contains
------------------------
Text files      the distinct line shapes ("masks") with their counts, the
                delimiter statistics, the field geometry (count, width, fixed
                or free), the numeric-token grammar in use, and the repeating
                block period.
HDF5 files      the group/dataset tree with shape, dtype, chunking,
                compression, attribute names and dimension-scale links.
Binary files    the magic bytes, the Fortran unformatted record chain if one
                is present, the plausible floating-point layout, and the
                record period.
Containers      zip / tar / gz / bz2 / xz members, profiled recursively.

What the report never contains
------------------------------
No field value, ever, in any tier.  Numbers are reported as shapes ("999.99")
and as counts, never as numbers read from the file.  The two exceptions are
deliberate and structural: byte counts (file size, record length, array
shape), and derived flags that carry no magnitude (is a column monotonic, how
many NaNs there are).

Redaction tiers (--tier)
------------------------
structure     the default, and the safe one.  Every name in the file --
              column headers, HDF5 dataset paths, attribute names, member
              filenames, even the profiled file's own name -- is masked the
              same way a data line is.  The report describes the skeleton and
              nothing else.
identifiers   the same report, plus the real names: headers, HDF5 paths,
              attribute names, units strings.  Names are usually engineering
              vocabulary rather than data, and they are what makes a reader
              writable, but they are the one thing that could be considered
              disclosure, so they are opt-in and they are printed in their own
              clearly delimited section of the report so they can be reviewed
              or deleted before the report is sent anywhere.

There is no third tier.  Values are never emitted at any setting.

Usage
-----
    python aoe_probe.py RESULTS.DAT
    python aoe_probe.py RESULTS.DAT --tier identifiers -o profile.json
    python aoe_probe.py ARCHIVE.zip --recurse
    python aoe_probe.py RUN.h5 --tier identifiers --json-only

Requires Python 3.8+ and the standard library.  h5py is used when present and
its absence only limits HDF5 profiling to the container level.  numpy is used
when present and only makes the binary scan faster.
"""

from __future__ import annotations

import argparse
import bz2
import collections
import datetime
import gzip
import hashlib
import io
import json
import lzma
import os
import re
import struct
import sys
import tarfile
import tempfile
import zipfile

PROBE_VERSION = 1

TIER_STRUCTURE = "structure"
TIER_IDENTIFIERS = "identifiers"

# Reading caps.  A results file can be gigabytes; the structure is visible in
# far less than that.  The head is read in full, then a middle and a tail
# window, so a file whose layout changes part-way through still shows both.
DEFAULT_MAX_BYTES = 8 * 1024 * 1024
MAX_MASKS_REPORTED = 60
MAX_IDENTIFIERS = 400
MAX_HDF5_OBJECTS = 2000
SCAN_ELEMENTS = 200000

# --------------------------------------------------------------------------
# masking
# --------------------------------------------------------------------------

_TRANSLATION = {}
for _c in range(0x20, 0x7F):
    _ch = chr(_c)
    if _ch.isdigit():
        _TRANSLATION[_c] = "9"
    elif "A" <= _ch <= "Z":
        _TRANSLATION[_c] = "A"
    elif "a" <= _ch <= "z":
        _TRANSLATION[_c] = "a"
    else:
        _TRANSLATION[_c] = _ch


def mask(text):
    """Return the shape of ``text``: digits become 9, letters A or a.

    Punctuation, spaces and layout survive, because they are the grammar.
    Non-ASCII letters become ``u`` and other control bytes become ``.``, so a
    mask is always printable and always the same length as its input.

        *TABLEAU : SIM_001    ->    *AAAAAAA : AAA_999
          0.0000   1.2340E+02 ->      9.9999   9.9999A+99
    """
    out = []
    for ch in text:
        code = ord(ch)
        mapped = _TRANSLATION.get(code)
        if mapped is not None:
            out.append(mapped)
        elif ch.isdigit():
            out.append("9")
        elif ch.isalpha():
            out.append("u")
        elif ch in "\t":
            out.append("\t")
        else:
            out.append(".")
    return "".join(out)


_RUN = re.compile(r"(.)\1{2,}")


def collapse(masked):
    """Collapse runs of three or more identical mask characters.

    ``9.9999A+99`` stays as it is; ``AAAAAAAAAAAA`` becomes ``A{12}``.  The
    collapsed form is what makes a report of wide fixed-width records readable.
    """
    return _RUN.sub(lambda m: "%s{%d}" % (m.group(1), len(m.group(0))), masked)


def family(masked):
    """A coarser grouping of a mask: every sign character becomes ``s``.

    ``9.9{6}A+99`` and ``9.9{6}A-99`` are one field layout written twice; the
    exponent's sign is data, not structure.  Folding it keeps a report of a
    real file from fragmenting into two near-identical halves.
    """
    return masked.replace("+", "s").replace("-", "s")


def masked_name(name, tier):
    """A name as the current tier allows it: real, or shaped."""
    if tier == TIER_IDENTIFIERS:
        return name
    return collapse(mask(name))


# --------------------------------------------------------------------------
# token grammar
# --------------------------------------------------------------------------

# Covers C-style (1.0E+03), Fortran D exponents (1.0D+03), Fortran exponents
# with the letter dropped (1.0+03), bare integers and bare decimals.
_NUMBER = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdDqQ][+-]?\d+|[+-]\d{1,3})?$")
_SPECIAL = re.compile(r"^[+-]?(?:nan|inf(?:inity)?|\*+|-+)$", re.IGNORECASE)
_INTEGER = re.compile(r"^[+-]?\d+$")
_IDENTIFIERISH = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-/\[\]()%:]*$")


_QUOTES = ("'", '"', "`")


def unquote(token):
    """Strip a matching pair of surrounding quotes.

    Quoted headers -- ``'TIME' 'ALPHA'`` -- are extremely common in Fortran
    output, and a quote is punctuation around a name, not part of it.
    """
    token = token.strip()
    if len(token) >= 2 and token[0] in _QUOTES and token[-1] == token[0]:
        return token[1:-1].strip()
    return token


_KIND_LETTER = {
    "int": "I", "float": "F", "special": "S", "name": "N",
    "other": "?", "empty": "_",
}


def field_grammar(fields):
    """A line's signature by field *class* rather than field shape.

    The character mask keeps digit-run lengths, which is what fixed-width
    formats are made of -- but in free-format text every line then has its own
    unique shape and nothing groups.  Replacing each field with one letter for
    its kind gives the view that does group: ``IFFFF`` is the same record
    whether a value needed three digits or nine.
    """
    letters = [_KIND_LETTER[classify_token(unquote(field))] for field in fields]
    return collapse("".join(letters))


def classify_token(token):
    """One of ``int``, ``float``, ``special``, ``name``, ``other``, ``empty``."""
    token = token.strip()
    if not token:
        return "empty"
    if _INTEGER.match(token):
        return "int"
    if _NUMBER.match(token):
        return "float"
    if _SPECIAL.match(token):
        return "special"
    if _IDENTIFIERISH.match(token):
        return "name"
    return "other"


def number_grammar(token):
    """Which numeric dialect a token is written in, or None."""
    token = token.strip()
    if not _NUMBER.match(token):
        return None
    if re.search(r"[dD]", token):
        return "fortran_d_exponent"
    if re.search(r"[eE]", token):
        return "c_e_exponent"
    if re.search(r"\d[+-]\d", token):
        return "fortran_bare_exponent"
    if "." in token:
        return "plain_decimal"
    return "integer"


# --------------------------------------------------------------------------
# byte-level sniffing
# --------------------------------------------------------------------------

MAGIC = [
    (b"\x89HDF\r\n\x1a\n", "hdf5"),
    (b"PK\x03\x04", "zip"),
    (b"\x1f\x8b", "gzip"),
    (b"BZh", "bzip2"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"CDF\x01", "netcdf3"),
    (b"CDF\x02", "netcdf3_64bit"),
    (b"\x93NUMPY", "npy"),
    (b"SQLite format 3\x00", "sqlite"),
    (b"MATLAB", "matlab"),
]


def sniff_magic(head):
    for signature, name in MAGIC:
        if head.startswith(signature):
            return name
    if len(head) > 257 + 5 and head[257:262] == b"ustar":
        return "tar"
    return None


def looks_textual(sample):
    """True when a byte sample reads as text rather than as packed binary."""
    if not sample:
        return True
    if b"\x00" in sample:
        return False
    printable = sum(
        1 for b in sample if 0x20 <= b < 0x7F or b in (0x09, 0x0A, 0x0D))
    return printable / float(len(sample)) > 0.90


def decode_best(raw):
    """Decode bytes to text, reporting which codec and BOM were involved."""
    for bom, codec in ((b"\xef\xbb\xbf", "utf-8-sig"),
                       (b"\xff\xfe\x00\x00", "utf-32-le"),
                       (b"\x00\x00\xfe\xff", "utf-32-be"),
                       (b"\xff\xfe", "utf-16-le"),
                       (b"\xfe\xff", "utf-16-be")):
        if raw.startswith(bom):
            return raw.decode(codec, "replace"), codec, True
    for codec in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(codec), codec, False
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", "replace"), "latin-1", False


# --------------------------------------------------------------------------
# text profiling
# --------------------------------------------------------------------------

CANDIDATE_DELIMITERS = [
    (",", "comma"), (";", "semicolon"), ("\t", "tab"), ("|", "pipe"),
    (":", "colon"), (" ", "whitespace"),
]


def split_fields(line, delimiter):
    if delimiter == " ":
        return line.split()
    return line.split(delimiter)


def delimiter_report(lines):
    """Score each candidate delimiter by how consistent its field count is."""
    scored = []
    for delimiter, label in CANDIDATE_DELIMITERS:
        counts = collections.Counter()
        for line in lines:
            if not line.strip():
                continue
            fields = split_fields(line, delimiter)
            counts[len(fields)] += 1
        total = sum(counts.values())
        if not total:
            continue
        modal_count, modal_hits = counts.most_common(1)[0]
        if modal_count < 2:
            continue
        scored.append({
            "delimiter": label,
            "modal_field_count": modal_count,
            "consistency": round(modal_hits / float(total), 4),
            "distinct_field_counts": len(counts),
        })
    scored.sort(key=lambda d: (-d["consistency"], -d["modal_field_count"]))
    return scored


def fixed_width_columns(lines, limit=400):
    """Infer column boundaries from positions that are blank on every line.

    Fortran fixed-format output has no delimiter at all; the field boundaries
    are byte offsets.  A column position that is a space in every sampled line
    is a boundary.  The result is the list of (start, width) slices, which is
    exactly what a reader needs and exactly what a masked sample cannot show.
    """
    sample = [ln for ln in lines[:limit] if ln.strip()]
    if len(sample) < 3:
        return None
    width = max(len(ln) for ln in sample)
    blank = [True] * width
    for line in sample:
        padded = line.ljust(width)
        for i, ch in enumerate(padded):
            if ch != " ":
                blank[i] = False
    slices = []
    start = None
    for i in range(width):
        if not blank[i] and start is None:
            start = i
        elif blank[i] and start is not None:
            slices.append((start, i - start))
            start = None
    if start is not None:
        slices.append((start, width - start))
    if not slices or len(slices) < 2:
        return None
    return {
        "line_width": width,
        "field_count": len(slices),
        "fields": [{"start": s, "width": w} for s, w in slices],
        "uniform_width": len(set(w for _, w in slices)) == 1,
    }


def block_period(mask_ids, max_period=512):
    """Smallest period that explains the sequence of line shapes.

    A per-run header followed by N data lines repeats; finding the period tells
    a reader where one record ends and the next begins.
    """
    n = len(mask_ids)
    if n < 6:
        return None
    best = None
    for period in range(1, min(max_period, n // 2) + 1):
        hits = 0
        for i in range(n - period):
            if mask_ids[i] == mask_ids[i + period]:
                hits += 1
        score = hits / float(n - period)
        if score > 0.95:
            best = {"period": period, "agreement": round(score, 4)}
            break
        if best is None and score > 0.80:
            best = {"period": period, "agreement": round(score, 4)}
    return best


def profile_text(raw, tier, truncated):
    text, codec, had_bom = decode_best(raw)
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    cr = text.count("\r") - crlf
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if truncated and lines:
        # The last line of a truncated read is a fragment; its shape is a lie.
        lines = lines[:-1]

    mask_counts = collections.Counter()
    family_counts = collections.Counter()
    mask_first = {}
    mask_ids = []
    blank = 0
    for index, line in enumerate(lines):
        if not line.strip():
            blank += 1
            mask_ids.append("")
            continue
        shaped = collapse(mask(line.rstrip()))
        mask_counts[shaped] += 1
        family_counts[family(shaped)] += 1
        mask_ids.append(shaped)
        if shaped not in mask_first:
            mask_first[shaped] = index + 1

    delimiters = delimiter_report(lines)
    chosen = delimiters[0]["delimiter"] if delimiters else None
    delimiter_char = None
    for char, label in CANDIDATE_DELIMITERS:
        if label == chosen:
            delimiter_char = char
            break

    # Token grammar and per-line field composition, over the chosen delimiter.
    grammars = collections.Counter()
    kinds = collections.Counter()
    field_grammars = collections.Counter()
    field_grammar_first = {}
    numeric_lines = 0
    header_candidates = []
    if delimiter_char is not None:
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            fields = split_fields(line, delimiter_char)
            signature = field_grammar(fields)
            field_grammars[signature] += 1
            if signature not in field_grammar_first:
                field_grammar_first[signature] = index + 1
            local = collections.Counter()
            for token in fields:
                kind = classify_token(unquote(token))
                local[kind] += 1
                kinds[kind] += 1
                grammar = number_grammar(token)
                if grammar:
                    grammars[grammar] += 1
            numeric = local["int"] + local["float"] + local["special"]
            if numeric >= max(2, 0.6 * len(fields)):
                numeric_lines += 1
            elif local["name"]:
                header_candidates.append((index + 1, fields))

    report = {
        "encoding": codec,
        "byte_order_mark": had_bom,
        "line_endings": {"crlf": crlf, "lf": lf, "cr": cr},
        "line_count_sampled": len(lines),
        "blank_lines": blank,
        "distinct_line_shapes": len(mask_counts),
        "numeric_lines": numeric_lines,
        "line_shapes": [
            {
                "shape": shaped,
                "count": count,
                "first_line": mask_first[shaped],
                "share": round(count / float(max(1, len(lines) - blank)), 4),
            }
            for shaped, count in mask_counts.most_common(MAX_MASKS_REPORTED)
        ],
        "line_shapes_omitted": max(0, len(mask_counts) - MAX_MASKS_REPORTED),
        "distinct_line_families": len(family_counts),
        "line_families": [
            {
                "shape": shaped,
                "count": count,
                "share": round(count / float(max(1, len(lines) - blank)), 4),
            }
            for shaped, count in family_counts.most_common(MAX_MASKS_REPORTED)
        ],
        "delimiters": delimiters,
        "chosen_delimiter": chosen,
        "field_grammars": [
            {
                "grammar": signature,
                "count": count,
                "first_line": field_grammar_first[signature],
                "share": round(count / float(max(1, len(lines) - blank)), 4),
            }
            for signature, count in field_grammars.most_common(
                MAX_MASKS_REPORTED)
        ],
        "distinct_field_grammars": len(field_grammars),
        "token_kinds": dict(kinds),
        "number_grammars": dict(grammars),
        "fixed_width": fixed_width_columns(lines),
        "fixed_width_is_likely": bool(
            not delimiters or delimiters[0]["consistency"] < 0.98),
        "block_period": block_period([m for m in mask_ids if m]),
        "header_candidate_lines": [ln for ln, _ in header_candidates[:20]],
    }

    identifiers = []
    if tier == TIER_IDENTIFIERS:
        seen = set()
        for _, fields in header_candidates:
            for token in fields:
                token = unquote(token)
                if not token or classify_token(token) != "name":
                    continue
                if token in seen:
                    continue
                seen.add(token)
                identifiers.append(token)
                if len(identifiers) >= MAX_IDENTIFIERS:
                    break
            if len(identifiers) >= MAX_IDENTIFIERS:
                break
    report["identifier_candidates"] = identifiers
    return report


# --------------------------------------------------------------------------
# binary profiling
# --------------------------------------------------------------------------

def fortran_record_chain(raw, marker="<i", limit=20000):
    """Walk a Fortran unformatted sequential file's record-marker chain.

    Each record is ``[int32 n][n bytes][int32 n]``.  If the chain walks the
    whole buffer, the layout is settled and the record lengths give the field
    count directly.
    """
    size = struct.calcsize(marker)
    offset = 0
    lengths = collections.Counter()
    records = 0
    total = len(raw)
    while offset + 2 * size <= total and records < limit:
        (length,) = struct.unpack_from(marker, raw, offset)
        if length < 0 or offset + 2 * size + length > total:
            break
        (trailer,) = struct.unpack_from(marker, raw, offset + size + length)
        if trailer != length:
            break
        lengths[length] += 1
        records += 1
        offset += 2 * size + length
    return {
        "records": records,
        "bytes_consumed": offset,
        "consumed_all": offset == total,
        "distinct_record_lengths": [
            {"length": length, "count": count}
            for length, count in lengths.most_common(20)
        ],
    }


def float_plausibility(raw, fmt, count=4096):
    """How many decoded floats are finite and in an engineering range.

    This asks a yes/no question of each word -- is this plausibly a real
    number -- and reports only the tally.  No decoded value is kept.
    """
    size = struct.calcsize(fmt)
    words = min(count, len(raw) // size)
    if not words:
        return None
    finite = 0
    plausible = 0
    zero = 0
    for i in range(words):
        (value,) = struct.unpack_from(fmt, raw, i * size)
        if value != value or value in (float("inf"), float("-inf")):
            continue
        finite += 1
        magnitude = abs(value)
        if magnitude == 0.0:
            zero += 1
            plausible += 1
        elif 1e-30 < magnitude < 1e30:
            plausible += 1
    return {
        "words_examined": words,
        "finite_fraction": round(finite / float(words), 4),
        "plausible_fraction": round(plausible / float(words), 4),
        "zero_fraction": round(zero / float(words), 4),
    }


def profile_binary(raw):
    head = raw[:32]
    tail = raw[-32:] if len(raw) >= 32 else b""
    nulls = raw.count(0)
    report = {
        "head_hex": head.hex(),
        "tail_hex": tail.hex(),
        "null_byte_fraction": round(nulls / float(max(1, len(raw))), 4),
        "fortran_records_little_endian": fortran_record_chain(raw, "<i"),
        "fortran_records_big_endian": fortran_record_chain(raw, ">i"),
        "float_layouts": {},
    }
    for label, fmt in (("float32_le", "<f"), ("float32_be", ">f"),
                       ("float64_le", "<d"), ("float64_be", ">d")):
        plausible = float_plausibility(raw, fmt)
        if plausible:
            report["float_layouts"][label] = plausible
    best = None
    for label, stats in report["float_layouts"].items():
        if best is None or stats["plausible_fraction"] > best[1]:
            best = (label, stats["plausible_fraction"])
    report["most_plausible_layout"] = best[0] if best else None
    return report


# --------------------------------------------------------------------------
# HDF5 profiling
# --------------------------------------------------------------------------

def profile_hdf5(path, tier):
    try:
        import h5py
    except ImportError:
        return {
            "available": False,
            "note": "h5py is not installed; install it to profile the tree "
                    "(pip install h5py).",
        }
    try:
        import numpy as np
    except ImportError:
        np = None

    objects = []
    truncated = [False]

    def scan_values(dataset):
        """Shape-only facts about a dataset's contents.

        Whether a column rises monotonically, how many NaNs it holds and
        whether every entry is a whole number are the three things a reader
        needs to identify an axis and to pick a dtype.  None of them is a
        value.
        """
        if np is None or dataset.dtype.kind not in "fiu":
            return None
        try:
            if dataset.ndim == 0:
                return None
            flat_count = 1
            for dim in dataset.shape:
                flat_count *= int(dim)
            if flat_count == 0:
                return {"empty": True}
            take = min(SCAN_ELEMENTS, int(dataset.shape[0]))
            block = np.asarray(dataset[:take])
            column = block.reshape(take, -1)[:, 0].astype("float64", copy=False)
        except Exception as exc:                        # pragma: no cover
            return {"error": type(exc).__name__}
        finite = np.isfinite(column)
        diffs = np.diff(column[finite]) if finite.sum() > 1 else None
        return {
            "elements_examined": int(column.size),
            "nan_count": int(np.isnan(column).sum()),
            "inf_count": int(np.isinf(column).sum()),
            "all_integral": bool(
                finite.all() and np.all(column[finite] == np.floor(column[finite]))),
            "monotonic_increasing": bool(
                diffs is not None and diffs.size and np.all(diffs > 0)),
            "monotonic_nondecreasing": bool(
                diffs is not None and diffs.size and np.all(diffs >= 0)),
            "constant": bool(finite.any() and
                             column[finite].min() == column[finite].max()),
        }

    def attributes(obj):
        out = []
        for key in obj.attrs:
            entry = {"name": masked_name(str(key), tier)}
            try:
                value = obj.attrs[key]
                entry["dtype"] = str(getattr(value, "dtype", type(value).__name__))
                entry["shape"] = list(getattr(value, "shape", ()) or ())
                if tier == TIER_IDENTIFIERS and isinstance(value, bytes):
                    entry["text"] = value.decode("utf-8", "replace")[:120]
                elif tier == TIER_IDENTIFIERS and isinstance(value, str):
                    entry["text"] = value[:120]
            except Exception as exc:                    # pragma: no cover
                entry["error"] = type(exc).__name__
            out.append(entry)
        return out

    with h5py.File(path, "r") as handle:

        def visit(name, obj):
            if len(objects) >= MAX_HDF5_OBJECTS:
                truncated[0] = True
                return True
            entry = {
                "path": masked_name("/" + name, tier),
                "depth": name.count("/") + 1,
                "kind": "group" if isinstance(obj, h5py.Group) else "dataset",
                "attributes": attributes(obj),
            }
            if isinstance(obj, h5py.Dataset):
                entry.update({
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "itemsize": int(obj.dtype.itemsize),
                    "chunks": list(obj.chunks) if obj.chunks else None,
                    "compression": obj.compression,
                    "compression_opts": obj.compression_opts,
                    "shuffle": bool(obj.shuffle),
                    "fletcher32": bool(obj.fletcher32),
                    "is_dimension_scale": bool(
                        obj.attrs.get("CLASS", b"") in (b"DIMENSION_SCALE",
                                                        "DIMENSION_SCALE")),
                })
                if obj.dtype.names:
                    entry["compound_fields"] = [
                        {"name": masked_name(field, tier),
                         "dtype": str(obj.dtype[field])}
                        for field in obj.dtype.names
                    ]
                entry["content_shape"] = scan_values(obj)
            objects.append(entry)
            return None

        handle.visititems(visit)
        root_attrs = attributes(handle)

    datasets = [o for o in objects if o["kind"] == "dataset"]
    shape_families = collections.Counter(
        tuple(o.get("shape") or ()) for o in datasets)
    return {
        "available": True,
        "object_count": len(objects),
        "group_count": len(objects) - len(datasets),
        "dataset_count": len(datasets),
        "truncated": truncated[0],
        "root_attributes": root_attrs,
        "shape_families": [
            {"shape": list(shape), "datasets": count}
            for shape, count in shape_families.most_common(30)
        ],
        "objects": objects,
    }


# --------------------------------------------------------------------------
# containers
# --------------------------------------------------------------------------

def profile_container(path, kind, tier, recurse, max_bytes, depth):
    entry = {"kind": kind, "members": []}
    try:
        if kind == "zip":
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist()[:500]:
                    entry["members"].append({
                        "name": masked_name(info.filename, tier),
                        "suffix": os.path.splitext(info.filename)[1].lower(),
                        "size_bytes": info.file_size,
                        "compressed_bytes": info.compress_size,
                        "compression": info.compress_type,
                        "is_directory": info.is_dir(),
                    })
        elif kind == "tar":
            with tarfile.open(path) as archive:
                for info in archive.getmembers()[:500]:
                    entry["members"].append({
                        "name": masked_name(info.name, tier),
                        "suffix": os.path.splitext(info.name)[1].lower(),
                        "size_bytes": info.size,
                        "is_directory": info.isdir(),
                    })
    except Exception as exc:
        entry["error"] = "%s: %s" % (type(exc).__name__, exc)
        return entry

    if recurse and depth < 2:
        entry["member_profiles"] = []
        with tempfile.TemporaryDirectory(prefix="aoe_probe_") as scratch:
            names = _extract_some(path, kind, scratch, limit=6)
            for extracted, original in names:
                entry["member_profiles"].append({
                    "name": masked_name(original, tier),
                    "profile": profile_path(extracted, tier, recurse,
                                            max_bytes, depth + 1),
                })
    return entry


def _extract_some(path, kind, scratch, limit):
    out = []
    if kind == "zip":
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir() or len(out) >= limit:
                    continue
                target = os.path.join(scratch, "m%d" % len(out))
                with archive.open(info) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                out.append((target, info.filename))
    elif kind == "tar":
        with tarfile.open(path) as archive:
            for info in archive.getmembers():
                if not info.isfile() or len(out) >= limit:
                    continue
                target = os.path.join(scratch, "m%d" % len(out))
                extracted = archive.extractfile(info)
                if extracted is None:
                    continue
                with open(target, "wb") as dst:
                    dst.write(extracted.read())
                out.append((target, info.name))
    return out


_STREAM_OPENERS = {
    "gzip": gzip.open,
    "bzip2": bz2.open,
    "xz": lzma.open,
}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def read_windows(path, max_bytes):
    """Read the head, a middle slice and the tail of a file.

    Sampling only the head would miss a trailer or a layout that changes after
    the first block, which is common in files that concatenate runs.
    """
    size = os.path.getsize(path)
    if size <= max_bytes:
        with open(path, "rb") as handle:
            return handle.read(), size, False
    chunk = max_bytes // 3
    with open(path, "rb") as handle:
        head = handle.read(chunk)
        handle.seek(size // 2)
        middle = handle.read(chunk)
        handle.seek(max(0, size - chunk))
        tail = handle.read(chunk)
    return head + b"\n" + middle + b"\n" + tail, size, True


def fingerprint(path, cap=4 * 1024 * 1024):
    """A digest of the first megabytes.

    Its only purpose is to let two reports be recognised as describing the
    same file, or an .aoe twin be tied to the run it came from.  A digest is
    not reversible and carries no content.
    """
    digest = hashlib.sha256()
    read = 0
    with open(path, "rb") as handle:
        while read < cap:
            block = handle.read(min(1 << 20, cap - read))
            if not block:
                break
            digest.update(block)
            read += len(block)
    return digest.hexdigest()[:16]


def profile_path(path, tier, recurse, max_bytes, depth=0):
    size = os.path.getsize(path)
    with open(path, "rb") as handle:
        head = handle.read(512)
    magic = sniff_magic(head)

    report = {
        "suffix": os.path.splitext(path)[1].lower(),
        "size_bytes": size,
        "detected": magic,
        "fingerprint_sha256_16": fingerprint(path),
    }
    if tier == TIER_IDENTIFIERS:
        report["name"] = os.path.basename(path)
    else:
        report["name_shape"] = collapse(mask(os.path.basename(path)))

    if magic == "hdf5":
        report["kind"] = "hdf5"
        report["hdf5"] = profile_hdf5(path, tier)
        return report

    if magic in ("zip", "tar"):
        report["kind"] = "container"
        report["container"] = profile_container(
            path, magic, tier, recurse, max_bytes, depth)
        return report

    if magic in _STREAM_OPENERS:
        report["kind"] = "compressed_stream"
        opener = _STREAM_OPENERS[magic]
        try:
            with opener(path, "rb") as stream:
                inner = stream.read(max_bytes)
        except Exception as exc:
            report["error"] = "%s: %s" % (type(exc).__name__, exc)
            return report
        report["inner"] = profile_bytes(inner, tier, truncated=True)
        report["inner"]["decompressed_bytes_read"] = len(inner)
        return report

    raw, size, truncated = read_windows(path, max_bytes)
    report.update(profile_bytes(raw, tier, truncated))
    report["bytes_examined"] = len(raw)
    report["sampled_windows"] = truncated
    return report


def profile_bytes(raw, tier, truncated):
    if looks_textual(raw[:65536]):
        return {"kind": "text", "text": profile_text(raw, tier, truncated)}
    inner_magic = sniff_magic(raw[:512])
    return {
        "kind": "binary",
        "detected_inner": inner_magic,
        "binary": profile_binary(raw),
    }


# --------------------------------------------------------------------------
# human-readable rendering
# --------------------------------------------------------------------------

def render(report, indent=0):
    pad = "  " * indent
    lines = []
    kind = report.get("kind")
    lines.append("%sfile      %s  (%s, %d bytes)" % (
        pad, report.get("name") or report.get("name_shape"),
        kind, report.get("size_bytes", 0)))
    if report.get("detected"):
        lines.append("%sdetected  %s" % (pad, report["detected"]))

    if kind == "text" or "text" in report:
        text = report["text"]
        lines.append("%sencoding  %s%s, endings crlf=%d lf=%d cr=%d" % (
            pad, text["encoding"], " +BOM" if text["byte_order_mark"] else "",
            text["line_endings"]["crlf"], text["line_endings"]["lf"],
            text["line_endings"]["cr"]))
        lines.append("%slines     %d sampled, %d blank, %d numeric, "
                     "%d distinct shapes" % (
                         pad, text["line_count_sampled"], text["blank_lines"],
                         text["numeric_lines"], text["distinct_line_shapes"]))
        if text["delimiters"]:
            best = text["delimiters"][0]
            lines.append("%sdelimiter %s -> %d fields, consistency %.2f" % (
                pad, best["delimiter"], best["modal_field_count"],
                best["consistency"]))
        if text["fixed_width"]:
            fixed = text["fixed_width"]
            lines.append("%sfixed     %d fields over %d columns%s%s" % (
                pad, fixed["field_count"], fixed["line_width"],
                ", uniform width" if fixed["uniform_width"] else "",
                "" if text.get("fixed_width_is_likely", True)
                else " (incidental: a delimiter already explains the file)"))
        if text["number_grammars"]:
            lines.append("%snumbers   %s" % (
                pad, ", ".join("%s=%d" % kv
                               for kv in sorted(text["number_grammars"].items()))))
        if text["block_period"]:
            lines.append("%speriod    %d lines (agreement %.2f)" % (
                pad, text["block_period"]["period"],
                text["block_period"]["agreement"]))
        if text.get("field_grammars"):
            lines.append("%sgrammars  %d distinct field-class signatures "
                         "(I=int F=float S=nan/inf N=name)"
                         % (pad, text["distinct_field_grammars"]))
            for entry in text["field_grammars"][:6]:
                lines.append("%s  %6d x  line %-6d %s" % (
                    pad, entry["count"], entry["first_line"],
                    entry["grammar"][:140]))
        lines.append("%sfamilies  %d distinct once exponent signs are folded"
                     % (pad, text.get("distinct_line_families", 0)))
        lines.append("%sshapes    (most frequent first)" % pad)
        for shape in text["line_shapes"][:12]:
            lines.append("%s  %6d x  %s" % (
                pad, shape["count"], shape["shape"][:150]))
        if text["line_shapes_omitted"]:
            lines.append("%s  ... %d more shapes" % (
                pad, text["line_shapes_omitted"]))
        if text.get("identifier_candidates"):
            lines.append("%sIDENTIFIERS (review before sending):" % pad)
            lines.append("%s  %s" % (
                pad, ", ".join(text["identifier_candidates"][:60])))

    if kind == "binary" or "binary" in report:
        binary = report["binary"]
        lines.append("%smagic     %s" % (pad, binary["head_hex"]))
        lines.append("%snulls     %.3f of bytes" % (
            pad, binary["null_byte_fraction"]))
        for label in ("fortran_records_little_endian",
                      "fortran_records_big_endian"):
            chain = binary[label]
            if chain["records"]:
                lines.append("%s%s: %d records, consumed all = %s" % (
                    pad, label.replace("fortran_records_", "fortran "),
                    chain["records"], chain["consumed_all"]))
        if binary["most_plausible_layout"]:
            best = binary["float_layouts"][binary["most_plausible_layout"]]
            lines.append("%sfloats    %s plausible for %.1f%% of words" % (
                pad, binary["most_plausible_layout"],
                100.0 * best["plausible_fraction"]))

    if "hdf5" in report:
        h5 = report["hdf5"]
        if not h5.get("available"):
            lines.append("%shdf5      %s" % (pad, h5["note"]))
        else:
            lines.append("%shdf5      %d groups, %d datasets" % (
                pad, h5["group_count"], h5["dataset_count"]))
            for family in h5["shape_families"][:8]:
                lines.append("%s  shape %-22s x %d datasets" % (
                    pad, str(family["shape"]), family["datasets"]))
            for obj in h5["objects"][:40]:
                if obj["kind"] == "dataset":
                    lines.append("%s  %-44s %-12s %s%s" % (
                        pad, obj["path"][:44], str(obj.get("shape")),
                        obj.get("dtype"),
                        "  chunked" if obj.get("chunks") else ""))
                else:
                    lines.append("%s  %-44s group" % (pad, obj["path"][:44]))
            if len(h5["objects"]) > 40:
                lines.append("%s  ... %d more objects (see the JSON)" % (
                    pad, len(h5["objects"]) - 40))

    if "container" in report:
        container = report["container"]
        lines.append("%scontainer %d members" % (
            pad, len(container["members"])))
        for member in container["members"][:20]:
            lines.append("%s  %-40s %10d bytes  %s" % (
                pad, str(member["name"])[:40], member["size_bytes"],
                member["suffix"]))
        for nested in container.get("member_profiles", []):
            lines.append("%s  --- member %s ---" % (pad, nested["name"]))
            lines.extend(render(nested["profile"], indent + 2))

    if "inner" in report:
        lines.append("%sinner stream:" % pad)
        inner = dict(report["inner"])
        inner.setdefault("name_shape", "(decompressed stream)")
        inner.setdefault("size_bytes", inner.get("decompressed_bytes_read", 0))
        lines.extend(render(inner, indent + 1))

    if report.get("error"):
        lines.append("%sERROR     %s" % (pad, report["error"]))
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Describe an unknown result file's structure without "
                    "reproducing its contents.")
    parser.add_argument("path", help="file to profile")
    parser.add_argument("--tier", choices=[TIER_STRUCTURE, TIER_IDENTIFIERS],
                        default=TIER_STRUCTURE,
                        help="how much naming to include (default: structure, "
                             "which masks every name in the file)")
    parser.add_argument("-o", "--output", help="write the JSON report here")
    parser.add_argument("--json-only", action="store_true",
                        help="print JSON to stdout instead of the text report")
    parser.add_argument("--recurse", action="store_true",
                        help="profile members inside zip/tar archives")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES,
                        help="cap on bytes read from one file "
                             "(default %d)" % DEFAULT_MAX_BYTES)
    args = parser.parse_args(argv)

    if not os.path.isfile(args.path):
        parser.error("not a file: %s" % args.path)

    report = {
        "probe_version": PROBE_VERSION,
        "generated_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "tier": args.tier,
        "platform": sys.platform,
        "python": sys.version.split()[0],
    }
    report.update(profile_path(args.path, args.tier, args.recurse,
                               args.max_bytes))

    payload = json.dumps(report, indent=2, sort_keys=False, default=str)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    if args.json_only:
        print(payload)
    else:
        print("\n".join(render(report)))
        if args.output:
            print("\nJSON report written to %s" % args.output)
        elif not args.json_only:
            print("\n(pass -o profile.json to save the machine-readable form)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
