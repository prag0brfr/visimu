# -*- coding: utf-8 -*-
"""A worked example of the reader protocol, to copy when writing a real one.

A reader is one function.  It takes a path and returns

    {run_name: {channel_name: array_of_floats}}

That is all aoe_compare.py needs, and it is also the shape a future push-based
entry point into the C++ core would want: one contiguous array per channel,
named, grouped by run.  Getting a reader correct here is therefore most of the
work of getting the format into the application.

This example parses a fixed-width Fortran-style listing of the sort that shows
up in solver output:

    *RUN  SIM_001
    *CHANNELS   TIME        ALTITUDE    MACH
      0.000E+00  1.000E+04  8.500E-01
      1.000E-01  1.001E+04  8.502E-01
    *END

Replace the three parsing rules with the real ones once aoe_probe.py has said
what the real ones are; the surrounding shape stays.
"""

from __future__ import annotations

import re

import numpy as np

_RUN = re.compile(r"^\*RUN\s+(\S+)", re.IGNORECASE)
_CHANNELS = re.compile(r"^\*CHANNELS\s+(.*)$", re.IGNORECASE)
_END = re.compile(r"^\*END", re.IGNORECASE)


def _to_float(token):
    """Parse the numeric dialects a Fortran writer emits.

    ``1.0D+03`` and the exponent-letter-less ``1.0+03`` both appear in the
    wild, and float() rejects both.
    """
    token = token.strip()
    if not token or set(token) <= {"*"}:
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
        return float(re.sub(r"(\d)([+-]\d{1,3})$", r"\1E\2", token))
    except ValueError:
        return float("nan")


def read(path):
    """Return {run: {channel: np.ndarray}} for one ad-hoc listing file."""
    runs = {}
    run_name = None
    channels = None
    columns = None

    def flush():
        if run_name is None or not channels:
            return
        runs[run_name] = {
            name: np.asarray(columns[index], dtype=np.float64)
            for index, name in enumerate(channels)
        }

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            match = _RUN.match(line)
            if match:
                flush()
                run_name = match.group(1)
                channels = None
                columns = None
                continue

            match = _CHANNELS.match(line)
            if match:
                channels = match.group(1).split()
                columns = [[] for _ in channels]
                continue

            if _END.match(line):
                flush()
                run_name = None
                channels = None
                columns = None
                continue

            if channels is None:
                continue
            fields = line.split()
            if len(fields) < len(channels):
                continue
            for index in range(len(channels)):
                columns[index].append(_to_float(fields[index]))

    flush()
    return runs
