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

from .. import zoomtrace
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

# A zoom that every tile follows on x. Two ways to ask for it, because two
# habits exist: the right button, and Ctrl with the left. Neither changes what
# the drag means -- an area drag, an x-only drag and a y-only drag all stay
# what they were; the x of the result is simply handed to the other tiles as
# well.
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

    Only a rectangle drag on both axes, in zoom mode, is taken over. A drag
    with a specific axis --- which is what dragging on an axis itself produces
    --- a middle drag, and pan mode are all handed back to pyqtgraph, because
    none of them is the gesture this is about.

    The left button zooms this tile. The right button, or Ctrl with the left,
    zooms this tile and hands the x range it ended up with to `shared_x_zoom`,
    which is how every other tile comes along. The shape of the drag is read
    the same way in both cases: what the modifier changes is who the zoom
    reaches, not what it means.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set by the tile. Called with (low, high) in data coordinates when a
        # zoom was asked to reach the other tiles.
        self.shared_x_zoom = None
        # How many drag events this gesture has delivered, for the trace. A
        # real attribute rather than a getattr default: a ViewBox answers
        # getattr for names it has never heard of.
        self._drag_steps = 0

    def mouseDragEvent(self, event, axis=None):
        # Traced: this is where the band is decided, and a drag that never
        # arrives here is a different fault from one that arrives and is
        # handed back to pyqtgraph.
        start = self._drag_flag(event, "isStart")
        finish = self._drag_flag(event, "isFinish")
        mine = self._is_area_zoom(event, axis)
        if start or finish or not mine:
            zoomtrace.log("vb.drag",
                          axis=axis,
                          button=zoomtrace.button_name(event),
                          modifiers=zoomtrace.modifier_names(event),
                          isStart=start, isFinish=finish, mine=mine,
                          **zoomtrace.view_state(self))

        if not mine:
            # Handed to pyqtgraph: a drag on an axis, a middle drag, or pan
            # mode. Worth seeing in the log, because "the band never appeared"
            # and "the band appeared and did nothing" start here.
            return super().mouseDragEvent(event, axis=axis)

        origin = event.buttonDownPos(event.button())
        current = event.pos()
        bounds = self.boundingRect()

        mode = zoom_axes(current.x() - origin.x(),
                         current.y() - origin.y(),
                         ui_scale())

        self._drag_steps = int(self._drag_steps or 0) + 1
        if start:
            self._drag_steps = 1
        if finish or zoomtrace.should_log_move(self._drag_steps):
            zoomtrace.log("vb.shape", step=self._drag_steps, mode=mode,
                          dx="{0:.1f}".format(current.x() - origin.x()),
                          dy="{0:.1f}".format(current.y() - origin.y()),
                          scale="{0:.2f}".format(ui_scale()),
                          isFinish=finish)

        if mode == NOTHING:
            # Nothing is drawn and nothing is zoomed -- a gesture that ends
            # this small must leave the view where it was -- but while it is
            # still running it is *accepted*, and that is the whole of the fix
            # for a zoom that worked only sometimes.
            #
            # pyqtgraph's scene does not deliver drag events to whoever is
            # under the pointer. It waits until the pointer has moved a few
            # pixels from the press, then offers that first event to the items
            # near it; the one that accepts becomes the scene's `dragItem` and
            # receives every later event of that gesture. Nobody accepting
            # means no dragItem -- and because the search is made on that
            # first event only, no further drag event is ever delivered,
            # however far the mouse then travels.
            #
            # Refusing a first step that was merely small therefore killed the
            # whole gesture. The scene's own threshold is around five pixels
            # and this floor is six (scaled), so whether a zoom worked was
            # decided by where in that narrow window the first delivered move
            # happened to land: a quick flick cleared it, a slow careful drag
            # did not. That is what "the zoom stops working after a while"
            # was -- not a stuck state, a gesture that never started.
            #
            # Accepted only while the drag is live. A finished one has nothing
            # left to claim, and leaving it unaccepted keeps a right-click
            # that was never a drag a right-click, so the view box's own menu
            # still opens on it.
            if event.isFinish():
                self._hide_band()
                zoomtrace.log("vb.nothing", reason="drag ended too small to "
                              "mean anything", **zoomtrace.view_state(self))
            else:
                event.accept()
                zoomtrace.log("vb.small", step=self._drag_steps,
                              reason="kept: the gesture may still grow")
            return

        event.accept()
        first, last = stretched(origin, current, bounds, mode)

        if not event.isFinish():
            self.updateScaleBox(first, last)
            if zoomtrace.should_log_move(self._drag_steps):
                zoomtrace.log("vb.band", step=self._drag_steps,
                              visible=zoomtrace.view_state(self)
                              .get("vb.bandVisible"))
            return

        self._hide_band()
        rect = self.childGroup.mapRectFromParent(QRectF(first, last))
        self.showAxRect(rect)
        self._remember(rect)
        flat_rect = rect.normalized()
        zoomtrace.log("vb.zoomed", mode=mode,
                      rect="{0:.6g}..{1:.6g} x {2:.6g}..{3:.6g}".format(
                          flat_rect.left(), flat_rect.right(),
                          flat_rect.top(), flat_rect.bottom()),
                      **zoomtrace.view_state(self))

        if self.is_global(event) and callable(self.shared_x_zoom):
            # Normalised, because a drag that went right to left gives a
            # rectangle with a negative width and "the x range" has no sign.
            flat = rect.normalized()
            self.shared_x_zoom(float(flat.left()), float(flat.right()))

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _drag_flag(event, name: str) -> bool:
        """isStart/isFinish off a drag event, defensively, for the trace."""
        try:
            return bool(getattr(event, name)())
        except (AttributeError, TypeError):
            return False

    @staticmethod
    def is_global(event) -> bool:
        """Whether this drag was asking for every tile to follow on x.

        The right button, or Ctrl with any button. Read off the event rather
        than from a mode, so there is nothing to set beforehand and nothing
        left switched on afterwards.
        """
        try:
            if event.button() == Qt.MouseButton.RightButton:
                return True
        except (AttributeError, TypeError):
            pass
        try:
            return bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        except (AttributeError, TypeError):
            return False

    def _is_area_zoom(self, event, axis) -> bool:
        """Whether this drag is the one we mean to reinterpret.

        Each refusal is traced separately: which of the three it was says
        whether the gesture never reached the right place, or reached it in
        the wrong mode.
        """
        if axis is not None:
            zoomtrace.log("vb.notmine", why="axis drag", axis=axis)
            return False
        try:
            if event.button() not in (Qt.MouseButton.LeftButton,
                                      Qt.MouseButton.RightButton):
                zoomtrace.log("vb.notmine", why="button",
                              button=zoomtrace.button_name(event))
                return False
        except (AttributeError, TypeError):
            zoomtrace.log("vb.notmine", why="button unreadable")
            return False
        # state is a plain dict on every pyqtgraph that has this class, but it
        # is read defensively: being wrong here would disable zooming, and
        # falling back to pyqtgraph's own handler never can.
        try:
            rect_mode = self.state.get("mouseMode") == pg.ViewBox.RectMode
            if not rect_mode:
                zoomtrace.log("vb.notmine", why="not in RectMode",
                              **zoomtrace.view_state(self))
            return rect_mode
        except (AttributeError, TypeError):
            zoomtrace.log("vb.notmine", why="mouseMode unreadable")
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
