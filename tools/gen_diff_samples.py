#!/usr/bin/env python3
# Author: Paulo R. A. Gilz
"""Generate the pair of files the non-regression comparison is tried on.

    python tools/gen_diff_samples.py sample/

Three files: two result files and the thresholds to compare them against.
They exist because the comparison is the one feature nobody can try out with
one file, and because "two files that are the same except where they are not"
is a thing nobody has lying around -- a real pair differs in a thousand
places and says nothing about whether the tool works.

What is engineered into them, and why each piece is there:

* The same runs, by identifier, in a *different order* in the two files. The
  files are matched by name, and a pair that happened to agree positionally
  would never show it.
* RUN_002 is identical in both files, in every channel, so it must not be
  flagged. A comparison that flags everything looks the same as one that
  works, until you have a run that should come out clean.
* RUN_001 is flagged on eighteen channels, which is more than one page of
  sixteen tiles holds: plotting it has to spill onto a second page.
* RUN_003 is flagged on five, and RUN_004 on one, so the report has runs of
  three different sizes in it.
* Three shapes of deviation, which are the three the shading has to tell
  apart (see DEVIATIONS): one group alone, one group inside another, and a
  deviation large enough that the higher group owns the whole window and the
  lower keeps only the slivers at its edges, where the coarser file's samples
  ramp through the lower band. That last shape is the priority rule at its
  edge, and the slivers are not a flaw in it: they are where the two files
  really do differ by that much.
* The second file samples its x axis more coarsely, so the arithmetic has to
  interpolate rather than subtract element by element.

Small on purpose: a megabyte, written in a second. The flight-data generator
makes the files that are big enough to be slow; these are the files that are
*wrong* in known places.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Sequence, Tuple

# TIME first; the rest are the channels thresholds are written against.
CHANNELS = ("TIME", "ALFA", "BETA", "Q1", "TETA", "PHI", "PSI", "NZ", "NY",
            "MACH", "ALT", "VCAS", "GAMMA", "DELTA_E", "DELTA_A", "DELTA_R",
            "CL", "CD", "CM", "THRUST", "MASS")
UNITS = ("s", "deg", "deg", "deg/s", "deg", "deg", "deg", "g", "g",
         "", "ft", "kt", "deg", "deg", "deg", "deg",
         "", "", "", "N", "kg")

SIGNALS = CHANNELS[1:]

# The run identifiers, and the order each file holds them in.
RUNS = ("RUN_001", "RUN_002", "RUN_003", "RUN_004")
ORDER_A = ("RUN_001", "RUN_002", "RUN_003", "RUN_004")
ORDER_B = ("RUN_003", "RUN_001", "RUN_004", "RUN_002")

DURATION = 40.0          # seconds
STEP_A = 0.05
STEP_B = 0.08            # deliberately not file A's

# The thresholds the configuration declares. ALFA has its own pair, as in the
# specification's example; every other channel shares one. The deviations
# below are sized against these rather than written out twice.
BLACK = {"ALFA": 25.0}
AMBER = {"ALFA": 10.0}
for _name in SIGNALS:
    BLACK.setdefault(_name, 10.0)
    AMBER.setdefault(_name, 5.0)

# The three shapes of deviation.
#
#   "amber"  a step between the two thresholds: the lower group alone.
#   "both"   the same, with a larger step inside it: both groups, and the
#            shading reads lower / higher / lower across the window.
#   "buried" a step above the higher threshold across the whole window: the
#            higher group owns the window and the lower is left with the
#            edges, where the interpolation ramps through its band.
SHAPES = ("both", "amber", "buried")

# (run, channel) -> (shape, start, end). Built rather than written out: the
# point of RUN_001 is that it has more flagged channels than a page holds,
# and a table of eighteen rows written by hand is a table with a typo in it.
DEVIATIONS: Dict[Tuple[str, str], Tuple[str, float, float]] = {}

for _at, _name in enumerate(SIGNALS[:18]):
    # Windows staggered so the pages do not all look like one plot drawn
    # eighteen times.
    _start = 4.0 + 1.5 * _at
    DEVIATIONS[("RUN_001", _name)] = (SHAPES[_at % 3], _start, _start + 12.0)

for _at, _name in enumerate(("BETA", "TETA", "NZ", "ALT", "CM")):
    DEVIATIONS[("RUN_003", _name)] = (SHAPES[_at % 3], 6.0 + 4.0 * _at,
                                      16.0 + 4.0 * _at)

DEVIATIONS[("RUN_004", "ALFA")] = ("amber", 30.0, 34.0)

# How far inside a "both" window the higher group's step sits.
INNER_IN = 5.0
INNER_OUT = 10.0

CONFIG = '''# Thresholds for the file-to-file comparison, as visimu reads them.
#
# A list of groups, each with a list of algoconfs. Each entry names the
# variables it covers (semicolon-separated), the threshold, and optionally the
# x axis to compare along -- TIME when it does not say. Keys this program does
# not know are ignored, at any level, so a file carrying your own fields works
# as it is.
#
# Priority is the threshold *value*: the higher number wins where two groups
# are both exceeded. The group names are identifiers, not colours -- the
# colour of each one is chosen in the DIFF tab.

 - group_name: "BLACK"
   description: "deviations that fail the test"
   algoconfs:
     - variables: "ALFA"
       x: "TIME"
       threshold: 25
     - variables: "{others}"
       x: "TIME"
       threshold: 10

 - group_name: "AMBER"
   description: "deviations worth a look"
   algoconfs:
     - variables: "ALFA"
       threshold: 10
     - variables: "{others}"
       threshold: 5
'''.format(others=";".join(name for name in SIGNALS if name != "ALFA"))


def _base(run_id: str, time: Sequence[float]) -> Dict[str, List[float]]:
    """The signals both files agree on, for one run.

    Smooth, bounded and run-dependent: a channel that is the same in every run
    makes a wrong run index invisible, and channels that all have the same
    shape make a wrong column index invisible too. So each one is given its
    own frequency, offset and scale.
    """
    number = RUNS.index(run_id)
    out: Dict[str, List[float]] = {}
    for at, name in enumerate(SIGNALS):
        frequency = 0.10 + 0.07 * (at % 7)
        phase = 0.4 * number + 0.3 * at
        scale = 1.0 + 0.5 * (at % 5)
        offset = 2.0 + 1.5 * (at % 4) + number
        out[name] = [offset + scale * math.sin(frequency * t + phase)
                     for t in time]
    return out


def _deviation(run_id: str, param: str, t: float) -> float:
    """What the second file adds to the first, at one instant.

    Flat steps rather than bumps: a step crosses its threshold at an instant
    that can be written down, which is what makes the shaded regions in the
    report checkable by hand.
    """
    plan = DEVIATIONS.get((run_id, param))
    if plan is None:
        return 0.0
    shape, start, end = plan
    if not (start <= t <= end):
        return 0.0
    if shape == "buried":
        return BLACK[param] + 4.0
    if shape == "amber":
        return AMBER[param] + 2.0
    # "both": the higher group's step sits inside the lower group's window.
    if start + INNER_IN <= t <= start + INNER_OUT:
        return BLACK[param] + 4.0
    return AMBER[param] + 2.0


def flagged(run_id: str) -> List[str]:
    """Which channels of a run were made to differ, in channel order.

    Used by the tests, and by anyone wondering what the files are supposed to
    say before running the comparison on them.
    """
    return [name for name in SIGNALS if (run_id, name) in DEVIATIONS]


def _grid(step: float) -> List[float]:
    count = int(round(DURATION / step)) + 1
    return [at * step for at in range(count)]


def write_file(path: str, order: Sequence[str], step: float,
               deviate: bool) -> int:
    """Write one of the two files. Returns how many samples it holds."""
    time = _grid(step)
    written = 0
    with open(path, "w", encoding="ascii", newline="\n") as handle:
        handle.write("*PARAMETRES\t\t\t\n")
        for index, (name, unit) in enumerate(zip(CHANNELS, UNITS), start=1):
            handle.write("\t{0}\t'{1}'\t \t'{2}'\n".format(index, name, unit))
        handle.write("\n")

        for run_id in order:
            handle.write("*TABLEAU : {0}\n".format(run_id))
            signals = _base(run_id, time)
            for at, t in enumerate(time):
                row = [t]
                for param in SIGNALS:
                    value = signals[param][at]
                    if deviate:
                        value += _deviation(run_id, param, t)
                    row.append(value)
                handle.write(" " + " ".join("{0:.6f}".format(v)
                                            for v in row) + "\n")
                written += 1
            handle.write("\n")
    return written


def generate(folder: str, quiet: bool = False) -> List[str]:
    """Write the two files and the configuration into `folder`.

    Returns the paths written. Files already there are left alone, like the
    flight-data samples: someone who has edited one to try something should
    not find it replaced by a rebuild.
    """
    os.makedirs(folder, exist_ok=True)
    plan = (
        ("regression_a.aoe", ORDER_A, STEP_A, False),
        ("regression_b.aoe", ORDER_B, STEP_B, True),
    )
    written: List[str] = []
    for name, order, step, deviate in plan:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            continue
        partial = path + ".partial"
        write_file(partial, order, step, deviate)
        os.replace(partial, path)
        written.append(path)
        if not quiet:
            print("  {0}: {1} runs, {2} channels".format(
                name, len(order), len(SIGNALS)))

    config = os.path.join(folder, "regression_thresholds.yaml")
    if not os.path.exists(config):
        with open(config, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(CONFIG)
        written.append(config)
        if not quiet:
            print("  regression_thresholds.yaml: 2 groups, {0} channels"
                  .format(len(SIGNALS)))
    if written and not quiet:
        for run_id in RUNS:
            names = flagged(run_id)
            print("  {0}: {1} channel(s) made to differ{2}".format(
                run_id, len(names),
                "" if names else "  (must come out clean)"))
    return written


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", default="sample",
                        help="where to write (default: sample)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    written = generate(args.folder, quiet=args.quiet)
    if not args.quiet and not written:
        print("Already there; nothing written.")
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
