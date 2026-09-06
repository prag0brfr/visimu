# Author: Paulo R. A. Gilz
"""Orthographic 3D projection for the 3D tiles.

Kept out of the widget on purpose. The widget cannot be constructed without a
running QApplication, and this is arithmetic -- the part most worth having
tests for. Nothing here imports Qt.

Why orthographic rather than perspective: the plots exist to have values read
off them. Under perspective the same interval covers different distances
depending on where it sits in the frame, which is exactly the property you do
not want when comparing one part of a trajectory with another.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np


def normalise(values: np.ndarray, bounds: Optional[Tuple[float, float]]) -> np.ndarray:
    """Map values onto [-0.5, 0.5] using the given bounds.

    Three channels in a 3D tile routinely have unrelated units -- metres,
    radians, Mach -- and plotting them raw gives a cube stretched into a needle
    along whichever axis happens to carry the largest numbers. Normalising each
    axis to the same extent is what makes the shape of the trajectory readable.

    A degenerate range (a constant channel, or bounds that are not finite)
    collapses to zero rather than dividing by it, which puts that axis in the
    middle of the cube -- the honest picture of a channel that never moves.
    """
    values = np.asarray(values, dtype=np.float64)
    if bounds is None:
        return np.zeros_like(values)
    low, high = float(bounds[0]), float(bounds[1])
    span = high - low
    if not np.isfinite(span) or span <= 0.0:
        return np.zeros_like(values)
    return (values - low) / span - 0.5


def project(x: np.ndarray, y: np.ndarray, z: np.ndarray,
            azimuth: float, elevation: float):
    """Project normalised 3D points onto the drawing plane.

    Rotate about the vertical (z) axis by ``azimuth`` degrees, tilt the viewer
    by ``elevation`` degrees, then drop the depth coordinate.

    Returns ``(screen_x, screen_y, depth)``. Depth is not used for drawing --
    there is no hidden-line removal -- but it is what a caller would need to
    order overlapping curves, so it is returned rather than discarded.
    """
    a = np.radians(float(azimuth))
    e = np.radians(float(elevation))
    cos_a, sin_a = np.cos(a), np.sin(a)
    cos_e, sin_e = np.cos(e), np.sin(e)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    rx = x * cos_a - y * sin_a
    ry = x * sin_a + y * cos_a

    screen_x = rx
    screen_y = z * cos_e - ry * sin_e
    depth = z * sin_e + ry * cos_e
    return screen_x, screen_y, depth


# The unit cube is the *normalisation frame*, not something drawn: each of the
# three channels is mapped onto [-0.5, 0.5] so they share one shape. It is what
# the default view has to contain, which is why the corners are still needed.
CUBE_CORNERS = np.array(
    [[cx, cy, cz] for cx in (-0.5, 0.5) for cy in (-0.5, 0.5) for cz in (-0.5, 0.5)],
    dtype=np.float64,
)

CUBE_EDGES: Sequence[Tuple[int, int]] = (
    (0, 1), (2, 3), (4, 5), (6, 7),      # along z
    (0, 2), (1, 3), (4, 6), (5, 7),      # along y
    (0, 4), (1, 5), (2, 6), (3, 7),      # along x
)


# The three unit axes of the cube, as an orientation triad. Colours follow the
# convention every 3D tool shares -- X red, Y green, Z blue -- because it is the
# one piece of this interface a user already knows before they open it.
AXIS_VECTORS = (
    ("X", (1.0, 0.0, 0.0), "#ff5252"),
    ("Y", (0.0, 1.0, 0.0), "#3ddc84"),
    ("Z", (0.0, 0.0, 1.0), "#4aa3ff"),
)


def axis_triad(azimuth: float, elevation: float):
    """The projected orientation triad: one entry per axis, sorted back to front.

    Returns ``[(name, sx, sy, colour, depth), ...]`` with sx, sy in [-1, 1] --
    a direction, not a position, so the caller scales it to whatever pixel size
    the gizmo is drawn at. That is what keeps the triad the same size however
    far the user has zoomed in: it is never expressed in data coordinates at
    all.

    Sorted by depth so the caller can paint them in order and have the axis
    nearest the viewer drawn last, over the others.
    """
    entries = []
    for name, vector, colour in AXIS_VECTORS:
        sx, sy, depth = project(
            np.array([vector[0]]), np.array([vector[1]]), np.array([vector[2]]),
            azimuth, elevation,
        )
        entries.append((name, float(sx[0]), float(sy[0]), colour, float(depth[0])))

    entries.sort(key=lambda entry: entry[4])
    return entries


def triad_layout(azimuth: float, elevation: float, size: float,
                 arm: float, head: float, label_gap: float):
    """Where to paint the orientation triad, in the gizmo's own pixel box.

    Pure geometry, with no Qt in it, because the interesting failure is not a
    painting bug but a layout one: a label pushed outside the item's bounding
    rectangle is clipped or leaves artefacts, and that is only visible on
    screen at one particular rotation. Computing it here means it can be
    asserted for every rotation instead.

    Returns one entry per axis, back to front::

        {"name", "colour", "start", "end", "head", "label", "degenerate"}

    with all points as (x, y) in a `size` x `size` box whose y grows downward,
    matching Qt's scene coordinates.
    """
    centre = (size * 0.5, size * 0.5)
    entries = []

    for name, sx, sy, colour, _depth in axis_triad(azimuth, elevation):
        # The projection's y grows upward; the paint surface's grows downward.
        end = (centre[0] + sx * arm, centre[1] - sy * arm)
        dx = end[0] - centre[0]
        dy = end[1] - centre[1]
        length = math.hypot(dx, dy)

        if length <= 1e-6:
            # Pointing straight at or away from the viewer. There is no
            # direction to build an arrowhead from, so it becomes a dot and
            # the label goes to a fixed offset.
            entries.append({
                "name": name, "colour": colour,
                "start": centre, "end": centre, "head": (),
                "label": (centre[0] + label_gap * 0.7,
                          centre[1] - label_gap * 0.7),
                "degenerate": True,
            })
            continue

        ux, uy = dx / length, dy / length
        base = (end[0] - ux * head * 2.0, end[1] - uy * head * 2.0)
        entries.append({
            "name": name, "colour": colour,
            "start": centre, "end": end,
            "head": (
                end,
                (base[0] - uy * head, base[1] + ux * head),
                (base[0] + uy * head, base[1] - ux * head),
            ),
            "label": (centre[0] + ux * (arm + label_gap),
                      centre[1] + uy * (arm + label_gap)),
            "degenerate": False,
        })

    return entries


def thinning_step(count: int, budget: int) -> int:
    """Stride that brings ``count`` points down to about ``budget``.

    A projected path cannot be decimated the way a 2D curve can: min/max
    bucketing needs a monotonic x, and a trajectory through space is monotonic
    in none of the three. Plain striding is the honest option here, and it is
    applied once when the projection is built rather than on every redraw.
    """
    if count <= 0 or budget <= 0:
        return 1
    return max(1, count // budget)


# ---------------------------------------------------------------- grid
#
# A 3D projection without a reference surface is genuinely ambiguous. Under
# orthographic projection there is no perspective cue at all, so a point in the
# middle of the frame could be anywhere along the line of sight, and a
# trajectory that loops back on itself is impossible to read.
#
# What every 3D plotting tool does about this is the same: draw grid lines on
# the *back walls* of the data box -- the three faces facing away from the
# viewer -- and swap which walls those are as the view rotates, so the grid is
# always behind the data and never draws over it. MATLAB, Origin, Tecplot,
# gnuplot, matplotlib's mplot3d and Plotly all render it this way, and the
# convention is old enough that engineers read it without being told.
#
# The tick labels matter as much as the lines. Until this existed a 3D tile had
# no numeric scale anywhere on it: you could see the shape of a trajectory but
# not read a single value off it.


def nice_ticks(low: float, high: float, target: int = 5):
    """Round tick values covering [low, high], about `target` of them.

    Steps of 1, 2, 2.5 or 5 times a power of ten -- the sequence people read
    without effort. Returns an empty list for a degenerate or non-finite range
    rather than inventing a scale for an axis that has none.
    """
    low = float(low)
    high = float(high)
    if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
        return []
    if target < 2:
        target = 2

    span = high - low
    raw = span / target
    magnitude = 10.0 ** math.floor(math.log10(raw))
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = multiple * magnitude
        if raw <= step:
            break

    # Counted rather than accumulated: adding `step` repeatedly drifts, and the
    # drift shows up as a tick landing just outside the box.
    first = math.ceil(low / step - 1e-9)
    last = math.floor(high / step + 1e-9)
    if last < first:
        return []
    return [(first + i) * step for i in range(int(last - first) + 1)]


def format_tick(value: float, step: float) -> str:
    """A tick label with the precision the step calls for, and no more.

    The step decides the decimals: with a step of 0.25 a label of "1" is
    ambiguous next to "1.25", and with a step of 500 a label of "1500.000" is
    just noise.
    """
    if not math.isfinite(value):
        return ""
    if abs(value) < 1e-12:
        value = 0.0

    # How many decimals the *step* itself needs, not how many its magnitude
    # suggests: a step of 0.25 has a magnitude of 10^-1 but needs two places,
    # and rounding it to one turns the tick row into 0, 0.2, 0.5, 0.8, 1.
    decimals = 3
    if step > 0 and math.isfinite(step):
        for places in range(0, 10):
            if abs(round(step, places) - step) <= abs(step) * 1e-9:
                decimals = places
                break

    # Very large or very small numbers read better in exponent form than as a
    # row of zeros.
    if value != 0.0 and (abs(value) >= 1e6 or abs(value) < 1e-4):
        return "{0:.3g}".format(value)

    text = "{0:.{1}f}".format(value, min(decimals, 6))
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def back_walls(azimuth: float, elevation: float):
    """Which side of the box each pair of opposite faces is drawn on.

    Returns ``(x_wall, y_wall, z_wall)``, each -0.5 or +0.5: the face of that
    axis that lies *behind* the data from this viewpoint. Choosing them from
    the view direction is what keeps the grid out of the way -- pick the near
    face instead and every line is drawn over the trajectory.

    Derived from the same rotation `project` uses. `project` returns a depth
    that grows towards the viewer, so the back face is whichever sign of the
    coordinate makes its contribution to depth smaller.
    """
    a = math.radians(float(azimuth))
    e = math.radians(float(elevation))
    cos_a, sin_a = math.cos(a), math.sin(a)
    cos_e, sin_e = math.cos(e), math.sin(e)

    # Contribution of a unit step along each axis to depth.
    towards_x = sin_a * cos_e
    towards_y = cos_a * cos_e
    towards_z = sin_e

    return (
        -0.5 if towards_x > 0.0 else 0.5,
        -0.5 if towards_y > 0.0 else 0.5,
        -0.5 if towards_z > 0.0 else 0.5,
    )


def wall_grid_segments(azimuth: float, elevation: float,
                       fractions_x, fractions_y, fractions_z):
    """Grid lines on the three back walls, in normalised cube coordinates.

    Each `fractions_*` is a sequence of positions in [-0.5, 0.5] -- the tick
    positions of that axis, already normalised. Returns a list of
    ``((x0, y0, z0), (x1, y1, z1))`` segments spanning the walls.

    Every wall carries lines in both of its in-plane directions, so the result
    reads as three panels meeting at the far corner rather than as a wireframe
    box: the box itself was removed on purpose, and putting it back would undo
    that.
    """
    x_wall, y_wall, z_wall = back_walls(azimuth, elevation)
    lo, hi = -0.5, 0.5
    segments = []

    # The y-z wall, at x = x_wall.
    for y in fractions_y:
        segments.append(((x_wall, y, lo), (x_wall, y, hi)))
    for z in fractions_z:
        segments.append(((x_wall, lo, z), (x_wall, hi, z)))

    # The x-z wall, at y = y_wall.
    for x in fractions_x:
        segments.append(((x, y_wall, lo), (x, y_wall, hi)))
    for z in fractions_z:
        segments.append(((lo, y_wall, z), (hi, y_wall, z)))

    # The x-y wall -- the floor, when the viewer is above.
    for x in fractions_x:
        segments.append(((x, lo, z_wall), (x, hi, z_wall)))
    for y in fractions_y:
        segments.append(((lo, y, z_wall), (hi, y, z_wall)))

    return segments


def tick_anchors(azimuth: float, elevation: float,
                 fractions_x, fractions_y, fractions_z):
    """Where each axis's tick labels sit, in normalised cube coordinates.

    The three back walls meet at one corner, and the three box edges radiating
    from it are the ones the labels belong on -- the same placement every 3D
    plot uses, and the only one where all three axes are legible at once
    whatever the rotation.

    Returns ``{"x": [(px, py, pz), ...], "y": [...], "z": [...]}``, one point
    per tick, nudged a little outside the box so a label does not sit on top of
    the grid line it belongs to.
    """
    x_wall, y_wall, z_wall = back_walls(azimuth, elevation)
    out = 0.06   # how far outside the box the labels sit

    return {
        # Along x, on the edge where the y and z back walls meet.
        "x": [(x, y_wall + math.copysign(out, y_wall),
               z_wall + math.copysign(out, z_wall)) for x in fractions_x],
        "y": [(x_wall + math.copysign(out, x_wall), y,
               z_wall + math.copysign(out, z_wall)) for y in fractions_y],
        "z": [(x_wall + math.copysign(out, x_wall),
               y_wall + math.copysign(out, y_wall), z) for z in fractions_z],
    }


def to_fraction(value: float, bounds) -> float:
    """One data value as a position in [-0.5, 0.5]. The scalar form of normalise()."""
    if bounds is None:
        return 0.0
    low, high = float(bounds[0]), float(bounds[1])
    span = high - low
    if not math.isfinite(span) or span <= 0.0:
        return 0.0
    return (float(value) - low) / span - 0.5
