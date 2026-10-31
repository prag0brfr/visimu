# -*- coding: utf-8 -*-
"""A reader for the .aoe text format itself, written against this protocol.

Its purpose is not to replace the core's parser -- the core reads .aoe faster
than this ever will.  Its purpose is to be the reference implementation of the
reader protocol, and to prove the harness: grading this reader against the same
file read through ``visimu.Session`` must come back "equivalent", because
both sides are reading the same bytes.  If it does not, the harness is wrong,
and that is worth knowing before an unknown format is graded with it.

So this doubles as the end-to-end test of the whole arrangement:

    python aoe_compare.py --aoe sample/quick.aoe --other sample/quick.aoe \\
        --reader readers.aoe_text_reader:read --align-runs order

The format:

    *PARAMETRES
    'TIME' 'ALPHA' 'BETA' ...        <- quoted channel names, one line
    *TABLEAU : 1                     <- a run begins, named by what follows ':'
    0 1.53544 0 7.54693 ...          <- one whitespace-separated row per sample
    *TABLEAU : 2
    ...

Note the two things that make a real format annoying and that this format
already has: the names are quoted, and the numbers are free-format, so no two
lines have the same character shape.  Both are what ``aoe_probe.py``'s quote
stripping and field-class grammar exist to handle.
"""

from __future__ import annotations

import numpy as np

PARAMETERS_KEYWORD = "*PARAMETRES"
TABLE_KEYWORD = "*TABLEAU"


def _unquote(token):
    token = token.strip()
    if len(token) >= 2 and token[0] in ("'", '"') and token[-1] == token[0]:
        return token[1:-1].strip()
    return token


def read(path):
    """Return {run_name: {channel_name: np.ndarray}}."""
    runs = {}
    names = []
    columns = None
    run_name = None
    expect_names = False

    def flush():
        if run_name is None or columns is None:
            return
        runs[run_name] = {
            name: np.asarray(columns[index], dtype=np.float64)
            for index, name in enumerate(names)
        }

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.upper().startswith(PARAMETERS_KEYWORD):
                expect_names = True
                continue

            if expect_names:
                names = [_unquote(token) for token in stripped.split()]
                expect_names = False
                continue

            if stripped.upper().startswith(TABLE_KEYWORD):
                flush()
                _, _, label = stripped.partition(":")
                run_name = label.strip() or str(len(runs) + 1)
                columns = [[] for _ in names]
                continue

            if columns is None:
                continue
            fields = line.split()
            if len(fields) < len(names):
                # A short row is a truncated write, not a row of zeros; dropping
                # it is the only honest choice and the harness will show it as a
                # length difference if the core disagrees.
                continue
            for index in range(len(names)):
                columns[index].append(float(fields[index]))

    flush()
    return runs
