# Author: Paulo R. A. Gilz
#
# Aligning runs on an event rather than on the clock.
#
# The problem this solves: a channel recorded over several runs of the same
# manoeuvre carries the manoeuvre at a different place in each run, because the
# recording started at a different moment. Plotted against TIME they are the
# same shape smeared sideways, and comparing them means comparing the shapes,
# not the clocks. A car accelerating from rest in five runs: the interesting
# instant is "when the speed began to rise", and every run has its own.
#
# So the user names a condition -- SPEED rises above 0.5, say -- this module
# finds where it first holds in each run, and the tile draws every curve
# shifted so those instants coincide.
#
# Deliberately free of Qt and of the core: the arithmetic is the part worth
# testing, and it is testable here with nothing but numpy.
#
# Three things the implementation is careful about:
#
#  * The event is located by interpolation, not by sample index. Two runs
#    sampled on different grids would otherwise align only to the nearest
#    sample, which on a 20 Hz log is a visible step.
#  * A run where the condition never holds is reported as such and left
#    unshifted. Hiding it would lose data; shifting it by a guess would be a
#    lie; the panel says how many did not match.
#  * NaN is not a value that satisfies anything. Gaps are skipped rather than
#    treated as zero, which is the same rule the parser applies when it turns
#    an unreadable token into NaN.

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# How the event is described. Each is a question with one answer per run.
CONDITIONS = (
    ("above", "first rises above"),
    ("below", "first falls below"),
    ("crosses", "first crosses (either way)"),
    ("rises", "first starts to rise"),
    ("falls", "first starts to fall"),
    ("max", "reaches its maximum"),
    ("min", "reaches its minimum"),
)

CONDITION_KEYS = tuple(key for key, _label in CONDITIONS)

# Where the events are moved to.
TARGETS = (
    ("zero", "zero"),
    ("first", "the first run's event"),
    ("value", "a value I choose"),
)

TARGET_KEYS = tuple(key for key, _label in TARGETS)

# A level is "reached" when the channel moves by this fraction of its own span,
# for the rises/falls conditions. A fraction rather than an absolute: the same
# setting then means the same thing on a channel in metres and one in radians.
RISE_FRACTION = 0.02


class Event(object):
    """Where one run's event is, and whether it was found at all."""

    __slots__ = ("x", "found", "reason")

    def __init__(self, x: float = float("nan"), found: bool = False,
                 reason: str = ""):
        self.x = x
        self.found = found
        self.reason = reason

    def __repr__(self) -> str:                      # pragma: no cover - debug
        return "<Event {0} found={1} {2}>".format(self.x, self.found,
                                                  self.reason)


def _finite_pair(x: np.ndarray, y: np.ndarray):
    """The samples where both the axis and the tested channel are readable."""
    count = min(x.size, y.size)
    if count == 0:
        return None, None
    x = np.asarray(x[:count], dtype=np.float64)
    y = np.asarray(y[:count], dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    if not np.any(good):
        return None, None
    return x[good], y[good]


def _crossing_x(x: np.ndarray, y: np.ndarray, index: int,
                level: float) -> float:
    """Where between two samples the channel passed `level`.

    `index` is the first sample on the far side. Straight-line interpolation
    between it and its predecessor: the alternative is to report the sample's
    own x, which quantises the alignment to the sampling interval and shows up
    as a step between runs logged at different rates.
    """
    if index <= 0:
        return float(x[0])
    y0, y1 = y[index - 1], y[index]
    if y1 == y0:
        return float(x[index])
    fraction = (level - y0) / (y1 - y0)
    fraction = min(1.0, max(0.0, float(fraction)))
    return float(x[index - 1] + fraction * (x[index] - x[index - 1]))


def find_event(x: np.ndarray, y: np.ndarray, condition: str,
               level: float = 0.0) -> Event:
    """Where `condition` first holds for one run, on the tile's own x axis.

    `x` is the channel the tile is plotted against and `y` the channel being
    tested; they need not be the same length, in which case the overlap is
    used, because a formula channel can come out shorter than the axis.
    """
    x, y = _finite_pair(x, y)
    if x is None or x.size == 0:
        return Event(reason="no readable samples")

    if condition == "max":
        return Event(float(x[int(np.argmax(y))]), True)
    if condition == "min":
        return Event(float(x[int(np.argmin(y))]), True)

    if condition == "above":
        hits = np.flatnonzero(y > level)
        if hits.size == 0:
            return Event(reason="never above {0:g}".format(level))
        return Event(_crossing_x(x, y, int(hits[0]), level), True)

    if condition == "below":
        hits = np.flatnonzero(y < level)
        if hits.size == 0:
            return Event(reason="never below {0:g}".format(level))
        return Event(_crossing_x(x, y, int(hits[0]), level), True)

    if condition == "crosses":
        # Either direction, judged against where the channel started: a
        # condition the user can set without knowing the sign of the approach.
        side = np.sign(y - level)
        start = next((s for s in side if s != 0), 0.0)
        if start == 0.0:
            # It starts exactly on the level; the first move off it counts.
            hits = np.flatnonzero(side != 0)
        else:
            hits = np.flatnonzero(side == -start)
        if hits.size == 0:
            return Event(reason="never crosses {0:g}".format(level))
        return Event(_crossing_x(x, y, int(hits[0]), level), True)

    if condition in ("rises", "falls"):
        # "Starts to move", with no level to choose: the channel's own span
        # sets the threshold, so the same setting works on a speed in m/s and
        # an angle in radians. The reference is the first sample, which is what
        # "starts" means -- a run already moving at sample zero reports its
        # first sample, correctly.
        span = float(np.nanmax(y) - np.nanmin(y))
        if span <= 0.0:
            return Event(reason="the channel never changes")
        threshold = span * RISE_FRACTION
        base = float(y[0])
        if condition == "rises":
            hits = np.flatnonzero(y > base + threshold)
            crossing = base + threshold
        else:
            hits = np.flatnonzero(y < base - threshold)
            crossing = base - threshold
        if hits.size == 0:
            return Event(reason="never moves far enough from its start")
        return Event(_crossing_x(x, y, int(hits[0]), crossing), True)

    return Event(reason="unknown condition {0!r}".format(condition))


def offsets(events: Dict[Tuple[int, int], Event], target: str,
            target_value: float = 0.0,
            order: Optional[Sequence[Tuple[int, int]]] = None
            ) -> Dict[Tuple[int, int], float]:
    """How far each run has to move, keyed the same way as `events`.

    An offset is subtracted from the run's x axis, so a run whose event sits at
    x=12 and should sit at x=0 has an offset of 12.

    A run whose event was not found gets no entry at all rather than an offset
    of zero: the two are the same number and not the same statement, and the
    panel reports the difference.
    """
    found = {key: event for key, event in events.items() if event.found}
    if not found:
        return {}

    if target == "first":
        # "First" is the first run in the order the caller gives, which is the
        # order the tile draws in -- not the lowest key, which would be an
        # arbitrary run the user never pointed at.
        sequence = list(order) if order else sorted(found)
        reference = None
        for key in sequence:
            if key in found:
                reference = found[key].x
                break
        if reference is None:
            return {}
    elif target == "value":
        reference = float(target_value)
    else:
        reference = 0.0

    return {key: event.x - reference for key, event in found.items()}


def describe(condition: str, param: str, level: float, target: str,
             target_value: float) -> str:
    """One line saying what the alignment does, for the panel and the axis."""
    if not param:
        return "no channel chosen"

    labels = dict(CONDITIONS)
    phrase = labels.get(condition, condition)
    if condition in ("above", "below", "crosses"):
        event = "{0} {1} {2:g}".format(param, phrase, level)
    else:
        event = "{0} {1}".format(param, phrase)

    if target == "first":
        where = "the first run's event"
    elif target == "value":
        where = "{0:g}".format(target_value)
    else:
        where = "0"
    return "aligned on {0}, moved to {1}".format(event, where)


def shift(values: np.ndarray, offset: float) -> np.ndarray:
    """`values` moved by `offset`, as a new array.

    Never in place: the array handed in is a view into the core's own memory,
    shared with every other tile and with the cache.
    """
    return np.asarray(values, dtype=np.float32) - np.float32(offset)
