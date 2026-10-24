# Author: Paulo R. A. Gilz
"""Rectangle zoom that takes its axes from the shape of the drag.

A rectangle zoom is the right gesture when you want a region of the plot, and
the wrong one most of the time: what is usually wanted is a stretch of x with
the y range left alone --- a manoeuvre between two instants, every channel of it
still in frame --- or the reverse, a band of y over the whole run.

Doing that with a rectangle means dragging the full height of the tile by eye
and getting it slightly wrong, which rescales y by a few per cent for no reason
and makes two tiles that should be comparable no longer comparable.

So the drag says which it is, by its shape. Drag out an area and you zoom the
area, as before. Drag something close to a horizontal line and only x is taken,
with y left exactly as it was; close to a vertical line and only y. There is no
mode to select and nothing to hold down: the three gestures are distinguished by
the one thing the user is already doing.

The rubber band shows the decision as it is made --- it spans the full height of
the tile for an x-only zoom and the full width for a y-only one --- so the gesture
is confirmed before the mouse is released rather than after.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtCore import QPointF

from ..theme import ui_scale

# The widest a drag's minor dimension may be and still count as a line, in
# pixels at a scale of 1.0. Generous enough that nobody has to drag precisely,
# and well below the height of a tile anyone would zoom an area of.
THIN_PX = 18.0

# And how many times longer the major dimension has to be. The ceiling alone is
# not enough: a 20x16 pixel drag is under it in both directions and is not a
# line in either, it is a small careless rectangle.
THIN_RATIO = 4.0

# Below this, in either direction, the drag is not a gesture at all. Without
# this a two-pixel twitch while clicking zooms the view by a factor of a
# hundred, and the only way back is Fit.
MIN_DRAG_PX = 6.0

AREA = "area"
HORIZONTAL = "x"      # a flat drag: x from the drag, y left alone
VERTICAL = "y"        # an upright drag: y from the drag, x left alone
NOTHING = "none"      # too small to mean anything


def zoom_axes(width: float, height: float, scale: float = 1.0) -> str:
    """Which axes a drag of this shape should zoom.

    `width` and `height` are the drag's extent in pixels and may be negative;
    only their size matters. `scale` is the interface scale, so that the
    thresholds mean the same thing to the eye at 70 % as at 160 %.

    Kept as a function of two numbers, free of any view box, because this is
    the whole of the behaviour and it is worth being able to test it exactly.
    """
    scale = float(scale) if scale and scale > 0 else 1.0
    thin = THIN_PX * scale
    floor = MIN_DRAG_PX * scale

    span = abs(float(width))
    rise = abs(float(height))

    if max(span, rise) < floor:
        return NOTHING
    if rise <= thin and span >= rise * THIN_RATIO:
        return HORIZONTAL
    if span <= thin and rise >= span * THIN_RATIO:
        return VERTICAL
    return AREA


def stretched(origin, current, bounds: QRectF, mode: str):
    """The two corners of the rectangle to zoom, widened along the free axis.

    For an x-only zoom the rectangle is given the full height of the view, and
    for a y-only zoom the full width. That is what makes the rest of this
    require no special case: mapping such a rectangle into data coordinates
    gives back exactly the range that axis already had, so "zoom x, leave y
    alone" is expressed as a rectangle like any other, by the one code path
    that was already correct.
    """
    x0, x1 = float(origin.x()), float(current.x())
    y0, y1 = float(origin.y()), float(current.y())

    if mode == HORIZONTAL:
        y0, y1 = float(bounds.top()), float(bounds.bottom())
    elif mode == VERTICAL:
        x0, x1 = float(bounds.left()), float(bounds.right())

    return QPointF(x0, y0), QPointF(x1, y1)


class DirectionalViewBox(pg.ViewBox):
    """A view box whose rectangle zoom reads the direction of the drag.

    Only the left-button rectangle drag on both axes is taken over. A drag
    with a specific axis --- which is what dragging on an axis itself produces
    --- a middle or right drag, and pan mode are all handed back to pyqtgraph,
    because none of them is the gesture this is about.
    """

    def mouseDragEvent(self, event, axis=None):
        if not self._is_area_zoom(event, axis):
            return super().mouseDragEvent(event, axis=axis)

        event.accept()
        origin = event.buttonDownPos(event.button())
        current = event.pos()
        bounds = self.boundingRect()

        mode = zoom_axes(current.x() - origin.x(),
                         current.y() - origin.y(),
                         ui_scale())

        if mode == NOTHING:
            # Nothing happens, and the band is taken down: a twitch while
            # clicking must leave the view where it was.
            if event.isFinish():
                self._hide_band()
            return

        first, last = stretched(origin, current, bounds, mode)

        if not event.isFinish():
            self.updateScaleBox(first, last)
            return

        self._hide_band()
        rect = self.childGroup.mapRectFromParent(QRectF(first, last))
        self.showAxRect(rect)
        self._remember(rect)

    # ------------------------------------------------------------ helpers

    def _is_area_zoom(self, event, axis) -> bool:
        """Whether this drag is the one we mean to reinterpret."""
        if axis is not None:
            return False
        try:
            if event.button() != Qt.MouseButton.LeftButton:
                return False
        except (AttributeError, TypeError):
            return False
        # state is a plain dict on every pyqtgraph that has this class, but it
        # is read defensively: being wrong here would disable zooming, and
        # falling back to pyqtgraph's own handler never can.
        try:
            return self.state.get("mouseMode") == pg.ViewBox.RectMode
        except (AttributeError, TypeError):
            return False

    def _hide_band(self) -> None:
        try:
            self.rbScaleBox.hide()
        except (AttributeError, RuntimeError):
            pass

    def _remember(self, rect: QRectF) -> None:
        """Add the new view to pyqtgraph's own zoom history.

        Best-effort: the history is what its back-navigation reads, and the
        attribute names have moved between versions. Losing a history entry
        costs one undo step; raising here would cost the zoom.
        """
        try:
            self.axHistoryPointer += 1
            self.axHistory = self.axHistory[:self.axHistoryPointer] + [rect]
        except (AttributeError, TypeError):
            pass
