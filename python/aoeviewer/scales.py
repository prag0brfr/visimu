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

__all__ = ["to_view", "from_view", "range_to_view", "MAX_EXPONENT"]

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
