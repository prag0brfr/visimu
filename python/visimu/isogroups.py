# Author: Paulo R. A. Gilz
"""Editing the intervals an ISO parameter is grouped into.

A grouped ISO parameter turns a sweep of forty Mach numbers into three or four
bands you can actually read. The core proposes the even split; everything about
*changing* it afterwards is here.

The representation is the reason this module exists. A group is stored as its
own ``(name, low, high, active)`` tuple -- that is what the filter reads and
what a workspace saves -- but a set of independent intervals is the wrong thing
to let someone edit directly: nothing stops two of them overlapping, or a gap
opening between them that quietly drops every run in it. Both mistakes are
invisible until a plot comes back short.

So editing works on **edges** instead. N groups over N+1 sorted edges, group i
spanning ``[edge[i], edge[i+1]]``; moving one edge is the only operation, and it
is clamped to its neighbours. Overlaps and interior gaps are then not something
to validate, they are something that cannot be expressed. The outer two edges
are still free to move inwards, which is how you deliberately exclude an
outlier -- that is a gap the user asked for, and `values_outside` reports it.

No Qt here, so the arithmetic can be tested without a running application --
the same reason visimu.scales is its own module.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

__all__ = [
    "Group",
    "clamp_group_count",
    "edges_of",
    "groups_from_edges",
    "move_edge",
    "even_edges",
    "default_name",
    "rename_defaults",
    "counts_per_group",
    "group_of",
    "values_outside",
    "is_contiguous",
]

# (name, low, high, active) -- the shape AppState.IsoState.groups holds.
Group = Tuple[str, float, float, bool]


def clamp_group_count(count: int, value_count: int) -> int:
    """A usable number of groups for a parameter with `value_count` values.

    Never more groups than there are values to put in them: past that the extra
    intervals are empty by construction, and an empty interval in the list is a
    filter row that can never match anything.
    """
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = 1
    if value_count <= 0:
        return 1
    return max(1, min(count, int(value_count)))


def edges_of(groups: Sequence[Group]) -> List[float]:
    """The N+1 edges of N contiguous groups.

    Groups that are *not* contiguous -- a workspace saved before this, or one
    whose values have moved under it -- still produce edges here, taking each
    group's low and the last group's high. The result is sorted, so a subsequent
    move_edge() works from something coherent even if what it was given was not.
    """
    if not groups:
        return []
    # By low first, so the "last high" really is the last one. Taking the highs
    # in stored order instead gave a set of edges describing neither the old
    # grouping nor a valid new one.
    ordered = sorted(groups, key=lambda group: float(group[1]))
    edges = [float(group[1]) for group in ordered]
    edges.append(max(float(ordered[-1][2]), edges[-1]))
    return edges


def groups_from_edges(edges: Sequence[float],
                      names: Optional[Sequence[str]] = None,
                      active: Optional[Sequence[bool]] = None,
                      format_value: Optional[Callable[[float], str]] = None
                      ) -> List[Group]:
    """N groups from N+1 edges, keeping the names and flags given by position.

    Positional rather than matched by value on purpose: when the number of
    groups changes there is no correspondence to match on, and when it does not,
    position is what the user was looking at in the table.
    """
    edges = [float(e) for e in edges]
    if len(edges) < 2:
        return []
    out: List[Group] = []
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        name = None
        if names is not None and index < len(names):
            name = names[index]
        if not name:
            name = default_name(low, high, format_value)
        flag = True
        if active is not None and index < len(active):
            flag = bool(active[index])
        out.append((name, low, high, flag))
    return out


def move_edge(edges: Sequence[float], index: int, value: float) -> List[float]:
    """Put edge `index` at `value`, clamped so the edges stay ordered.

    The clamp is what makes overlaps and interior gaps unrepresentable: an edge
    may travel anywhere between the edges on either side of it, and no further.
    Dragging one past its neighbour would reorder the groups under the row the
    user is editing, which is never what dragging a boundary means.

    An out-of-range index, or a value that is not a finite number, leaves the
    edges alone -- a typo in a table cell should do nothing, not something.
    """
    result = [float(e) for e in edges]
    if not (0 <= index < len(result)):
        return result
    try:
        value = float(value)
    except (TypeError, ValueError):
        return result
    if not math.isfinite(value):
        return result

    low = result[index - 1] if index > 0 else -math.inf
    high = result[index + 1] if index + 1 < len(result) else math.inf
    result[index] = max(low, min(high, value))
    return result


def even_edges(values: Sequence[float], count: int) -> List[float]:
    """`count` equal-width intervals spanning the observed values.

    The fallback for when the core's own splitter is not to hand -- the dialog's
    "Distribute evenly" uses the core -- and the definition the tests measure
    the core's answer against.
    """
    numbers = [float(v) for v in values if _finite(v)]
    if not numbers:
        return []
    count = max(1, int(count))
    low, high = min(numbers), max(numbers)
    if high <= low:
        # Every run at the same value. One interval, and a degenerate one:
        # widening it would claim a range the data does not have.
        return [low, high]
    step = (high - low) / count
    edges = [low + step * i for i in range(count)]
    edges.append(high)
    return edges


def default_name(low: float, high: float,
                 format_value: Optional[Callable[[float], str]] = None) -> str:
    """The label a group carries until someone renames it."""
    if format_value is None:
        def format_value(value):
            return "{0:g}".format(value)
    return "{0} - {1}".format(format_value(low), format_value(high))


def rename_defaults(groups: Sequence[Group],
                    format_value: Optional[Callable[[float], str]] = None
                    ) -> List[Group]:
    """Regenerate every name from its bounds.

    Used when the bounds move: a group called "0.7 - 0.8" that now spans 0.7 to
    0.9 is worse than no name at all. A name the user typed is not a default and
    is not touched -- see IsoGroupDialog, which tracks which those are.
    """
    return [(default_name(low, high, format_value), low, high, active)
            for _name, low, high, active in groups]


def counts_per_group(groups: Sequence[Group],
                     values: Sequence[float]) -> List[int]:
    """How many of `values` fall in each group.

    Bounds are inclusive at both ends, matching the core's IsoGroup.contains, so
    a value sitting exactly on a shared edge belongs to the earlier group -- the
    first one that claims it. Counting the same way as the filter is the point:
    a dialog that disagreed with what the plot then showed would be worse than
    showing no counts.
    """
    counts = [0] * len(groups)
    for value in values:
        if not _finite(value):
            continue
        value = float(value)
        for index, (_name, low, high, _active) in enumerate(groups):
            if low <= value <= high:
                counts[index] += 1
                break
    return counts


def group_of(groups: Sequence[Group], value: float) -> Optional[int]:
    """Which group claims a value, or None if none does.

    Inclusive at both ends and first claim wins, the same rule
    counts_per_group() applies and the same one the core's IsoGroup.contains
    applies -- so a value sitting exactly on a shared edge belongs to the
    earlier group. Three readers of one rule is two too many for it to live
    anywhere but here.
    """
    if not _finite(value):
        return None
    value = float(value)
    for index, (_name, low, high, _active) in enumerate(groups):
        if low <= value <= high:
            return index
    return None


def values_outside(groups: Sequence[Group],
                   values: Sequence[float]) -> List[float]:
    """The values no group claims.

    Two ways to get here and they deserve the same answer: the user pulled an
    outer edge in to exclude an outlier, or a file was loaded whose values fall
    outside a grouping set up for the old ones. Either way those runs are being
    filtered out, and saying so beats a plot that comes back short for no
    visible reason.
    """
    if not groups:
        return []
    missing = []
    for value in values:
        if not _finite(value):
            continue
        value = float(value)
        if not any(low <= value <= high
                   for _name, low, high, _active in groups):
            missing.append(value)
    return missing


def is_contiguous(groups: Sequence[Group], tolerance: float = 1e-9) -> bool:
    """True when the groups touch end to end, in order.

    What move_edge() guarantees, stated so it can be asserted.
    """
    for first, second in zip(groups, groups[1:]):
        if second[1] < first[1]:
            return False
        span = max(abs(first[2]), abs(second[1]), 1.0)
        if abs(first[2] - second[1]) > tolerance * span:
            return False
    return True


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
