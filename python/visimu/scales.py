# Author: Paulo R. A. Gilz
"""Conversion between data units and plot-view units.

pyqtgraph's log mode does not scale the axis. It plots log10 of the samples on
an ordinary linear axis and relabels the ticks, so every number crossing into or
out of the view -- ``setYRange``, ``setLimits``, ``viewRange`` -- is in log
units, while a tile's stored ``y_min``/``y_max`` are, and must remain, in data
units. The two were mixed: ticking Log Y on a channel spanning 1 to 1000 framed
the view at log-space 1..1000, which is data 10^1 to 10^1000, while the curve
sat at log-values 0..3. The tile looked empty, double-clicking recomputed the
same answer, and those mixed numbers then propagated as a variable-wide limit
onto tiles that were not in log mode at all.

Keeping the conversion here rather than inline at the call sites has a second
purpose: this module imports no Qt, so the arithmetic can be tested without a
running application, which is where a mistake of this kind hides.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

__all__ = ["to_view", "from_view", "range_to_view", "resample_key",
           "resample_target", "MAX_EXPONENT", "MAX_TILE_POINTS",
           "POINTS_PER_PIXEL"]

# Past this the exponent overflows a float, and no plot has anything to say
# about 10^400 anyway.
MAX_EXPONENT = 300.0


def to_view(value: float, log: bool) -> Optional[float]:
    """Data units to view units. None when the value has no place on a log axis.

    Zero and negatives have no logarithm, and a caller that gets None must
    decide what to do about it rather than being handed a silently wrong
    number -- which is precisely how this bug read before it was found.
    """
    value = float(value)
    if not log:
        return value if math.isfinite(value) else None
    if not (value > 0.0) or math.isinf(value) or math.isnan(value):
        return None
    return math.log10(value)


def from_view(value: float, log: bool) -> float:
    """View units back to data units."""
    value = float(value)
    if not log:
        return value
    if value > MAX_EXPONENT:
        value = MAX_EXPONENT
    elif value < -MAX_EXPONENT:
        value = -MAX_EXPONENT
    return float(10.0 ** value)


def range_to_view(low: float, high: float, log: bool,
                  decades: float = 3.0) -> Optional[Tuple[float, float]]:
    """A (low, high) pair in view units, or None if it cannot be shown.

    On a log axis a channel that crosses zero has a drawable part and an
    undrawable one. Framing it on the drawable part -- ``decades`` below the
    peak -- shows the user their data; refusing to frame it at all shows them an
    empty tile and no explanation.
    """
    view_high = to_view(high, log)
    view_low = to_view(low, log)
    if view_high is None:
        return None
    if view_low is None:
        view_low = view_high - abs(decades)
    if view_high < view_low:
        view_low, view_high = view_high, view_low
    return (view_low, view_high)


# ---------------------------------------------------------------- draw budget

# Points handed to Qt per horizontal pixel, per curve. Two is what a min/max
# decimation needs to keep every spike: one for the high, one for the low.
POINTS_PER_PIXEL = 2

# Ceiling on the points handed to Qt for one tile, across every curve. The
# per-curve budget above is the first limit; this is the second, for the case
# where the curve *count* is what makes the total unreasonable.
MAX_TILE_POINTS = 1_500_000


def resample_target(width_target: int, curve_count: int, total_points: int) -> int:
    """How many buckets each curve gets when a tile is re-decimated.

    Two limits, and the answer is the smaller. `width_target` is the per-curve
    one -- two points per horizontal pixel, which is all a screen can show.
    MAX_TILE_POINTS is the second, on the total across every curve: a thousand
    runs of 500 samples is 600 000 points on a 600-pixel tile, and no amount of
    per-curve fidelity is visible at a thousand curves per pixel column.

    The second limit is a **ceiling, never a floor**, and making that hard to
    get wrong is why this is a named function rather than three lines inside
    the redraw. Written as a plain assignment it was neither:
    `MAX_TILE_POINTS / curves / 2` only falls below a typical width target
    somewhere past 400 curves, so at 75 -- three files of 25 runs against one
    parameter, an entirely ordinary comparison -- a budget meant to protect the
    draw path *raised* the target from 1 800 to 10 000 and pushed 1.5 M points
    into Qt on every pan instead of 270 k. Decimation itself barely noticed
    (12 ms against 19 ms); the cost was all in pyqtgraph, and it showed up as a
    window that would not keep up with the mouse.

    The figure returned is buckets, not points: min/max decimation emits two
    points per bucket, which is why the budget divides by two.
    """
    if curve_count <= 1 or total_points <= MAX_TILE_POINTS:
        return width_target
    return min(width_target, max(32, int(MAX_TILE_POINTS / curve_count / 2)))


def significant(value: float, digits: int = 12) -> float:
    """`value` rounded to `digits` significant figures.

    A view range read back from a plot wobbles in the last bits of a double as
    the mouse moves, so comparing raw floats calls every mouse move a new view.
    Twelve figures is far finer than any zoom a person performs and far coarser
    than that noise.
    """
    if value == 0.0 or not math.isfinite(value):
        return 0.0
    factor = 10.0 ** (digits - 1 - math.floor(math.log10(abs(value))))
    return round(value * factor) / factor


def resample_key(x_range: Optional[Tuple[float, float]], target: int,
                 log: bool) -> tuple:
    """What a tile's drawn points were computed for.

    Two redraws with equal keys would produce the same points, so the second
    can be skipped -- which is what makes restyling a curve, dragging a marker
    or returning to a page cost nothing.

    Both halves of the answer are needed, and getting that wrong is easy. The
    position is quantised to the bucket width, because a difference smaller than
    one bucket cannot move a drawn point. But the span has to be carried
    separately and in absolute terms: quantising *only* the endpoints hides
    every zoom whose lower edge scales with its width -- a range of 0..1 and a
    range of 0..0.25 both come out as bucket -82, because the bucket shrank by
    exactly the factor the edge did. Zooming in around the origin then kept
    whatever resolution the curve already had, and no amount of zooming got any
    more of it.
    """
    if x_range is None:
        return (None, None, int(target), bool(log))
    low, high = float(x_range[0]), float(x_range[1])
    span = high - low
    if not (span > 0.0) or not math.isfinite(span) or not math.isfinite(low):
        return (None, None, int(target), bool(log))
    bucket = span / max(1, int(target))
    return (round(low / bucket), significant(span), int(target), bool(log))


# ---------------------------------------------------------------------------
# How much room the tick numbers need
# ---------------------------------------------------------------------------
#
# pyqtgraph works the height of the bottom axis out for itself, from the text
# it drew the last time it painted. That is one paint behind: the height it
# reserves is right for the font the axis had before, so the first layout after
# the tick font changes -- which is every change of interface scale, and the
# first paint of every tile -- reserves the previous font's height. When the
# new font is taller the numbers are drawn below the space reserved for them
# and the tile clips them, which is what hides the time values along the bottom
# of a plot.
#
# It shows up on Linux and not on Windows because the default font differs:
# DejaVu Sans at the same point size is taller than Segoe UI, so the figure
# pyqtgraph starts from is too small there and close enough to right here.
#
# The fix is to stop asking it to guess. The height below is computed from the
# font's own metrics, measured on the machine that will draw it, and set as a
# fixed height -- so it is right on the first paint, on every platform, at
# every scale.

# Breathing room under the numbers, in pixels at a scale of 1.0. Without any,
# a descender on the longest tick label sits exactly on the tile's edge.
TICK_ROW_PAD = 2.0


def tick_row_height(text_height: float, tick_length: float = 0.0,
                    offset: float = 2.0, scale: float = 1.0) -> int:
    """Pixels the bottom axis needs for one row of tick numbers.

    `text_height` is the font's full line height including descenders, which
    is what QFontMetrics.height() gives; measuring the glyphs of the labels
    actually drawn would be smaller and would clip the first label containing
    a minus sign or a digit with a descender.

    `tick_length` is pyqtgraph's, and is negative when the ticks are drawn
    inside the plot -- which is the default, and costs the axis no height at
    all, so a negative value contributes nothing rather than subtracting.
    """
    scale = float(scale) if scale and scale > 0 else 1.0
    height = max(0.0, float(text_height))
    height += max(0.0, float(offset))
    height += max(0.0, float(tick_length))
    height += TICK_ROW_PAD * scale
    # One pixel is the floor: a height of zero makes pyqtgraph hide the axis
    # rather than draw it small.
    return max(1, int(math.ceil(height)))
