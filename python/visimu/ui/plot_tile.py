# Author: Paulo R. A. Gilz
"""One plot in the grid.

This widget is where the performance of the whole application is decided, so
the strategy is worth stating plainly:

  * Full-resolution samples never reach pyqtgraph. Each curve keeps a read-only
    NumPy view of the C++ buffer, and what gets drawn is a min/max decimation
    down to roughly two points per horizontal pixel.
  * Decimation is redone when the visible x range changes, so zooming in reveals
    real detail rather than a stretched approximation, and zooming out never
    costs more than the tile's pixel width.
  * Range changes are coalesced through a short timer. pyqtgraph emits
    sigRangeChanged continuously while the mouse drags; recomputing on each one
    would decimate hundreds of times per second for no visible benefit.

The net effect is that a tile showing ten million-sample channels draws about
12,000 points, and panning stays smooth.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pyqtgraph as pg
from pyqtgraph.graphicsItems.GraphicsWidgetAnchor import GraphicsWidgetAnchor
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy

from ..model import (
    AppState,
    ColorMode,
    MouseMode,
    StyleMode,
    TileState,
    clamp_3d_half,
    curve_color_key,
    curve_line_key,
    tile_title,
)
from .. import projection, scales, sync, zoomtrace
from ..session import Session
from ..theme import (
    PLOT_AXIS_PT,
    PLOT_LABEL_PT,
    PLOT_LEGEND_PT,
    PLOT_TITLE_PT,
    px,
    pt,
    ui_scale,
    Theme,
    plot_font,
)

# The drawing budget lives in visimu.scales, which imports no Qt, so the
# arithmetic behind it can be tested without a running application -- which is
# exactly where the last mistake in it hid.
from ..scales import (  # noqa: F401
    MAX_TILE_POINTS,
    POINTS_PER_PIXEL,
    resample_key,
    resample_target,
    tick_row_height,
)
from .zoombox import DirectionalViewBox

# Stacking order of the things drawn on a tile. pyqtgraph puts plot curves at
# z = 0, so anything meant to be read *over* the data has to be above that.
# Annotations are furniture the engineer places deliberately; a band or a
# marker hidden behind whichever curve happens to cross it is not doing its job.
VIOLATION_Z = -5    # the comparison's verdict, behind the two series it judges
AREA_Z = 5          # shaded bands: translucent, so the data still reads through
MARKER_Z = 10       # marker lines and their dots

# The two files of a comparison, drawn in two colours. Fixed rather than taken
# from the palette: on a page of sixteen overlays the question is always "which
# of these two is which", and the answer should not change from tile to tile
# because the palette was walked in a different order.
FDIFF_COLOR_A = "#4da3ff"
FDIFF_COLOR_B = "#ff8a3d"
# How much of the shading is colour. A band is a verdict about a stretch of
# the run, and both curves have to stay readable through it.
VIOLATION_ALPHA = 0.22
GRID_3D_Z = -20     # the back walls of a 3D box, behind everything

# Most rows a legend may have. pyqtgraph builds a QGraphicsGridLayout with two
# items per row, so a thousand-run study asked Qt to lay out several thousand
# graphics items in one column -- which is where the application stopped
# responding rather than merely slowed down. Past this the legend says how many
# more there are, which is the only useful thing a list that long can say.
MAX_LEGEND_ROWS = 60

# The dot marking the sample the Inspect panel is reporting. Above the curves
# and the markers, because its whole job is to be findable.
#
# It used to be red, on the grounds that nothing in the palette was. The
# palette now begins with red, so a red dot would read as one more curve --
# exactly the confusion the colour was chosen to avoid. It is drawn instead as
# a white-cored dot with a black rim, which no curve can be because the palette
# holds no achromatic entry, and which stays findable on either theme: the rim
# carries it on the light ground and the core on the dark one.
INSPECT_DOT_Z = 40
INSPECT_DOT_SIZE = 9.0

# How far the pointer may travel between a double click and its release and
# still count as a click rather than the start of a drag, in pixels. Small:
# the hand does not move while double-clicking, and anything larger would
# start swallowing the fit people are asking for.
DOUBLE_CLICK_SLACK = 4.0
INSPECT_DOT_COLOUR = "#ffffff"
INSPECT_DOT_EDGE = "#101215"

# The legend, above all of it. It is a key rather than a reading of the data,
# and now that it is painted on an opaque card the card must not be underneath
# the curves it explains -- a trace crossing over the panel would put a line
# through the very swatches meant to identify it.
LEGEND_Z = 50

# Padding between a label's text and the edge of the ground painted behind it,
# in viewport pixels. Also the slack that makes a one-word caption easy to grab.
LABEL_PAD = 3.0

# How opaque that ground is. Not fully: a caption is furniture laid on the plot,
# and a little of the trace showing through is what keeps it reading as part of
# the same picture rather than as a hole punched in it. Enough that the text
# never has to compete with a curve underneath.
LABEL_GROUND_ALPHA = 0.88


def _label_brush(colour: str):
    """The same ground as paint_label_ground(), for pyqtgraph's own TextItems.

    TextItem draws its own fill, so it takes a brush rather than a paint call;
    keeping both in one place is what stops the crosshair readout and the
    captions drifting apart.
    """
    ground = QColor(colour)
    ground.setAlphaF(LABEL_GROUND_ALPHA)
    return pg.mkBrush(ground)


# The legend gets a more opaque ground than a caption does, and its own border.
# A caption is a word or two that the eye separates from the data by its shape
# alone; the legend is a block of rows whose swatches are themselves drawn in
# the curve colours, so any trace showing through it reads as one more entry.
# Still short of fully opaque, so you can still tell what it is covering.
LEGEND_GROUND_ALPHA = 0.94

# Room between the legend's frame and its rows. pyqtgraph's legend has none,
# which was invisible while there was no ground behind it and looks cramped the
# moment there is -- the text would sit flush against the border. Enough to
# read as a card, and no more: this is four edges of every tile's corner.
LEGEND_PAD = 3

# Between a row's swatch and its text, and between one row and the next.
# pyqtgraph defaults to 25 px horizontally, which on a short label is wider
# than the label; and its rows, at the sizes used here, sit further apart than
# they need to for the block to read as a list.
LEGEND_H_SPACING = 8
LEGEND_V_SPACING = 0


def legend_ground(theme) -> dict:
    """How the card the legend sits on is filled, framed and written on.

    Returned as the keyword arguments LegendItem takes, so the same call
    describes a legend being built and one being re-themed.

    Framed as well as filled, unlike a caption: at 94% opacity over a plot of
    the same background colour the panel has no visible edge at all where it
    covers an empty part of the tile, and the edge is the thing that tells the
    user where the legend stops and the data starts.
    """
    ground = QColor(theme.plot_background)
    ground.setAlphaF(LEGEND_GROUND_ALPHA)
    return {
        "brush": pg.mkBrush(ground),
        "pen": pg.mkPen(theme.plot_grid, width=1.0),
        "labelTextColor": theme.plot_foreground,
    }


def paint_label_ground(painter, rect: QRectF, colour: str) -> None:
    """Fill a rounded panel behind text drawn over the data.

    Every readout on a tile does this -- the crosshair figures, the slope, the
    captions -- because none of them were legible without it: text drawn
    straight onto the curves was dark on dark wherever a trace ran under it, and
    the eye read the two as one object. The fill is the plot background, so the
    label reads as a card laid on the plot rather than as another colour in the
    data.
    """
    ground = QColor(colour)
    ground.setAlphaF(LABEL_GROUND_ALPHA)
    painter.save()
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(ground))
    painter.drawRoundedRect(rect, 2.0, 2.0)
    painter.restore()

# pyqtgraph takes axis label styling as CSS-ish keyword arguments. Named here so
# every setLabel call is styled the same way; a label left unstyled falls back to
# the widget font, which is sized for the panels and looms over a small tile.
def _axis_label_style() -> dict:
    """pyqtgraph takes axis label styling as CSS-ish keyword arguments.

    A function rather than the constant it used to be: the size follows the UI
    scale, and a dictionary built at import time is fixed at whatever the
    scale was when the module loaded -- which is before any workspace has been
    read.
    """
    return {"font-size": "{0}pt".format(pt(PLOT_AXIS_PT))}

def _fit_tick_row(axis, font) -> None:
    """Give the bottom axis the height its tick numbers measure.

    pyqtgraph sizes the axis from the text of the *previous* paint, so the
    first layout after the tick font changes reserves the old font's height
    and clips the numbers when the new one is taller -- which is what hid the
    x values on Linux, where the default font is taller than on Windows.

    Measured here instead, from the font itself, and fixed: right on the first
    paint, on every platform and at every interface scale.

    Best-effort by design. If any of this is unavailable, pyqtgraph's own
    sizing is left in place: slightly cramped numbers are better than a tile
    that fails to build.
    """
    try:
        from PySide6.QtGui import QFontMetricsF

        metrics = QFontMetricsF(font)
        style = getattr(axis, "style", {}) or {}
        offset = style.get("tickTextOffset", (5, 2))
        try:
            offset = float(offset[1])
        except (TypeError, IndexError, ValueError):
            offset = 2.0
        axis.setHeight(tick_row_height(
            metrics.height(),
            tick_length=float(style.get("tickLength", 0) or 0),
            offset=offset,
            scale=ui_scale(),
        ))
    except (AttributeError, RuntimeError, TypeError, ValueError, ImportError):
        pass


_PEN_STYLES = {
    "solid": Qt.PenStyle.SolidLine,
    "dash": Qt.PenStyle.DashLine,
    "dot": Qt.PenStyle.DotLine,
    "dashdot": Qt.PenStyle.DashDotLine,
}

# No text tags -- [-], [o], [--] -- prefixed to legend entries to say what the
# line looks like. pyqtgraph's legend already draws a sample of the actual
# curve, dash pattern and marker included, so such text is a second and worse
# copy of information already there. The swatch is the single thing.

# pyqtgraph anchors a legend by the sign of its offset: a negative x measures
# from the right edge, a negative y from the bottom.
LEGEND_POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right",
                    "custom")

_LEGEND_OFFSETS = {
    "top-left": (10, 10),
    "top-right": (-10, 10),
    "bottom-left": (10, -10),
    "bottom-right": (-10, -10),
}

# Which corner of the legend is pinned to which corner of the view box.
_LEGEND_ANCHORS = {
    "top-left": (0, 0),
    "top-right": (1, 0),
    "bottom-left": (0, 1),
    "bottom-right": (1, 1),
}


class DraggableLegend(pg.LegendItem):
    """A legend that remembers where it was dragged to.

    pyqtgraph's legend is already draggable; what it does not do is tell anyone
    it moved. So every rebuild re-anchored it to the configured corner and the
    drag was silently undone -- the legend snapped back and there was no way to
    keep a position that was not one of the four corners.

    Overriding the drag to report the finished position is all that was
    missing. The position is stored as a *fraction* of the view box rather than
    as pixels, so it stays sensible when the tile is resized or the window is
    maximised.
    """

    def __init__(self, on_moved, **kwargs):
        pg.LegendItem.__init__(self, **kwargs)
        self._on_moved = on_moved
        # pyqtgraph lays the rows out flush with the item's edge. With nothing
        # painted behind them that went unnoticed; with a ground and a border it
        # is the difference between a card and a box drawn round some text.
        # Guarded inside set_metrics(), like everything else touching
        # pyqtgraph internals here: the layout is reachable under this name
        # across the versions this runs against, but a legend that is merely
        # roomier is a far better outcome than one that fails to build.
        self.set_metrics()

    def set_ground(self, brush=None, pen=None, labelTextColor=None) -> None:
        """Fill, frame and write on the panel behind the entries.

        Fill and frame go through opts rather than setBrush()/setPen(), whose
        names have come and gone across pyqtgraph versions; paint() has read
        opts['brush'] and opts['pen'] throughout. The text colour does need its
        setter, because it has to reach the rows already laid out.
        """
        try:
            self.opts["brush"] = brush
            self.opts["pen"] = pen
        except (AttributeError, TypeError):
            return
        if labelTextColor is not None:
            try:
                self.setLabelTextColor(labelTextColor)
            except (AttributeError, RuntimeError, TypeError):
                pass
        try:
            self.update()
        except (AttributeError, RuntimeError):
            pass

    def set_metrics(self) -> None:
        """Re-apply the sizes that follow the UI scale."""
        try:
            self.setLabelTextSize("{0}pt".format(pt(PLOT_LEGEND_PT)))
        except (AttributeError, RuntimeError, TypeError):
            pass
        try:
            pad = px(LEGEND_PAD)
            self.layout.setContentsMargins(pad, pad, pad, pad)
            self.layout.setHorizontalSpacing(px(LEGEND_H_SPACING))
            self.layout.setVerticalSpacing(LEGEND_V_SPACING)
        except (AttributeError, RuntimeError, TypeError):
            pass

    def mouseDragEvent(self, event):
        # Implemented rather than delegated: pyqtgraph's own handler is gated
        # on a `draggable` attribute whose name and default have moved between
        # versions, and the whole point here is that the drag must work.
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        event.accept()
        delta = event.pos() - event.lastPos()
        self.autoAnchor(self.pos() + delta)
        if event.isFinish() and self._on_moved is not None:
            self._on_moved(self.fractional_position())

    def fractional_position(self):
        """Top-left of the legend as a fraction of the view box, or None."""
        parent = self.parentItem()
        if parent is None:
            return None
        area = parent.boundingRect()
        width = area.width()
        height = area.height()
        if width <= 0 or height <= 0:
            return None
        position = self.pos()
        return (float(position.x()) / width, float(position.y()) / height)


class Axis3DGizmo(GraphicsWidgetAnchor, pg.GraphicsObject):
    """An orientation triad pinned to a corner of a 3D tile.

    The convention every simulation viewer uses -- ParaView, Tecplot, EnSight,
    and every CAD package -- is a small set of labelled axes in a corner that
    turns with the camera, rather than a box drawn around the data. It answers
    the only question a projection actually raises ("which way is which?")
    without putting anything on top of the curves.

    It keeps a constant size because it is parented to the *view box*, not to
    the data inside it: pyqtgraph applies the pan/zoom transform to a child
    group, so an item attached directly to the view box lives in viewport
    pixels and is untouched by either. The alternative -- drawing it in data
    coordinates and rescaling it on every range change -- would work but would
    have to be corrected on every zoom step, and would be wrong for the frame
    between the zoom and the correction.
    """

    SIZE = 84          # px, the square the gizmo occupies
    ARM = 26           # px, length of each axis from the origin
    HEAD = 6.0         # px, arrowhead half-width
    LABEL_GAP = 9.0    # px, from the arrow tip to the centre of its label

    def __init__(self, theme: Theme):
        pg.GraphicsObject.__init__(self)
        GraphicsWidgetAnchor.__init__(self)
        self._theme = theme
        self._azimuth = 45.0
        self._elevation = 25.0
        self.setZValue(50)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    # ---------------------------------------------------------------- state

    def set_orientation(self, azimuth: float, elevation: float) -> None:
        if (azimuth, elevation) != (self._azimuth, self._elevation):
            self._azimuth = azimuth
            self._elevation = elevation
            self.update()

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.update()

    def describe(self, entries) -> None:
        """Tooltip text: which parameter each axis is, and over what range.

        The labels stay X/Y/Z so the gizmo does not turn into a legend, but the
        ranges the removed cube used to carry still have to be reachable, and
        hovering is where they belong.
        """
        lines = []
        for axis, name, bounds in entries:
            if bounds is None:
                lines.append("{0}: {1}".format(axis, name))
            else:
                lines.append("{0}: {1}   [{2:.6g}, {3:.6g}]".format(
                    axis, name, bounds[0], bounds[1]))
        self.setToolTip("\n".join(lines))

    # --------------------------------------------------------------- paint

    def boundingRect(self):
        return QRectF(0, 0, self.SIZE, self.SIZE)

    def paint(self, painter, *_args):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Sized from the type scale rather than from the painter's font. The
        # application stylesheet sizes text in pixels, which makes pointSizeF()
        # report -1, and feeding that back to Qt is what produced the
        # "Point size <= 0" warning -- so the previous code had to guess a
        # fallback. Asking for the size by name removes the guess.
        painter.setFont(plot_font(PLOT_LABEL_PT, bold=True))
        metrics = painter.fontMetrics()

        for entry in projection.triad_layout(
                self._azimuth, self._elevation,
                self.SIZE, self.ARM, self.HEAD, self.LABEL_GAP):
            colour = QColor(entry["colour"])

            if entry["degenerate"]:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(colour))
                painter.drawEllipse(QPointF(*entry["start"]), 3.5, 3.5)
            else:
                pen = QPen(colour)
                pen.setWidthF(1.8)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(QPointF(*entry["start"]), QPointF(*entry["end"]))

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(colour))
                painter.drawPolygon(QPolygonF([QPointF(x, y)
                                               for x, y in entry["head"]]))

            label_x, label_y = entry["label"]
            painter.setPen(QPen(colour))
            painter.drawText(
                QPointF(label_x - metrics.horizontalAdvance(entry["name"]) * 0.5,
                        label_y + metrics.capHeight() * 0.5),
                entry["name"],
            )


class TextOverlay(GraphicsWidgetAnchor, pg.GraphicsObject):
    """The tile's captions, painted in viewport pixels.

    Positions are fractions of the tile rather than data coordinates, and the
    item is parented to the view box, so a caption stays where it was put when
    the plot is panned or zoomed. A note that slides off screen the moment you
    zoom is not a note; something that has to move with the data is a marker.
    """

    def __init__(self, on_moved=None, background: str = "#000000"):
        pg.GraphicsObject.__init__(self)
        GraphicsWidgetAnchor.__init__(self)
        self._background = background
        self._notes = []
        self._rect = QRectF(0, 0, 1, 1)
        # Where each visible note was last painted, so a click can be matched
        # to one. Kept from the paint pass because that is where the font
        # metrics -- and therefore the real extent of the text -- are known.
        self._boxes = []
        self._on_moved = on_moved
        self._dragging = None
        self._grab_offset = QPointF(0, 0)
        self.setZValue(45)
        # Captions are draggable, like the legend: typing fractions into the
        # panel to place a label is not how anyone wants to annotate a plot.
        self.setAcceptedMouseButtons(
            Qt.MouseButton.LeftButton if on_moved is not None
            else Qt.MouseButton.NoButton
        )
        # The item spans the whole viewport, because a caption may sit
        # anywhere in it. So the cursor must follow the *text*, not the item:
        # setting it here would put a drag cursor over every pixel of the plot.
        if on_moved is not None:
            self.setAcceptHoverEvents(True)

    def set_background(self, colour: str) -> None:
        """Follow the theme: the ground under a caption is the plot's own."""
        self._background = colour
        self.update()

    def set_notes(self, notes, rect: QRectF) -> None:
        self._notes = [n for n in notes if n.visible and n.text]
        self._rect = QRectF(rect)
        self.prepareGeometryChange()
        self.update()

    def boundingRect(self):
        return QRectF(0, 0, max(1.0, self._rect.width()),
                      max(1.0, self._rect.height()))

    def paint(self, painter, *_args):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        width = self._rect.width()
        height = self._rect.height()
        boxes = []

        for note in self._notes:
            font = painter.font()
            font.setPointSizeF(max(4.0, float(note.size)))
            font.setBold(bool(note.bold))
            painter.setFont(font)

            metrics = painter.fontMetrics()
            x = width * float(note.x)
            top = height * float(note.y)
            lines = str(note.text).splitlines() or [""]
            widest = max((float(metrics.horizontalAdvance(line))
                          for line in lines), default=0.0)
            # A few pixels of slack: it is the grab area as well as the ground,
            # and a one-word caption has to stay easy to catch.
            box = QRectF(x - LABEL_PAD, top - LABEL_PAD,
                         widest + 2 * LABEL_PAD,
                         metrics.lineSpacing() * len(lines) + 2 * LABEL_PAD)

            # An opaque ground under the text. Drawn straight onto the curves,
            # a caption competed with whatever it happened to land on: dark text
            # over a dark trace, and the eye reading the two as one object. The
            # plot background is the right fill because it is what the tile is
            # already made of, so the note reads as a card laid on the plot
            # rather than as another colour in the data.
            paint_label_ground(painter, box, self._background)

            painter.setPen(QPen(QColor(note.color)))
            y = top + metrics.ascent()
            for line in lines:
                painter.drawText(QPointF(x, y), line)
                y += metrics.lineSpacing()

            boxes.append((note, box))

        self._boxes = boxes

    def _note_at(self, point: QPointF):
        # Last painted is topmost, so search backwards.
        for note, box in reversed(self._boxes):
            if box.contains(point):
                return note, box
        return None, None

    def hoverMoveEvent(self, event):
        note, _box = self._note_at(event.pos())
        if note is None:
            self.unsetCursor()
        else:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        event.ignore()

    def hoverLeaveEvent(self, event):
        self.unsetCursor()
        event.ignore()

    def mousePressEvent(self, event):
        if self._on_moved is None or event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        note, box = self._note_at(event.pos())
        if note is None:
            # Nothing under the cursor: let the click reach the plot beneath,
            # so the empty space around a caption still selects the tile.
            event.ignore()
            return
        self._dragging = note
        self._grab_offset = event.pos() - box.topLeft()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()


    def mouseMoveEvent(self, event):
        if self._dragging is None:
            event.ignore()
            return
        width = max(1.0, self._rect.width())
        height = max(1.0, self._rect.height())
        target = event.pos() - self._grab_offset + QPointF(3.0, 3.0)
        # Clamped so a caption cannot be dragged off the tile and lost.
        self._dragging.x = min(0.98, max(0.0, float(target.x()) / width))
        self._dragging.y = min(0.98, max(0.0, float(target.y()) / height))
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        note, self._dragging = self._dragging, None
        self.unsetCursor()
        if note is None:
            event.ignore()
            return
        if self._on_moved is not None:
            self._on_moved(note)
        event.accept()


class CurveEntry:
    """One logical curve: the full-resolution data and how it is styled.

    Deliberately *not* a drawn item. Several hundred curves that share a pen are
    painted by a single item (see DrawGroup), but each remains its own entry
    here, because everything other than painting works per curve: the crosshair
    readout, click-to-inspect, marker dots, area segments and the CSV export all
    ask "which run and which parameter is this".

    The x and y arrays are views over the core's buffers and, since they carry
    their own reference to those buffers, they keep the samples alive for as
    long as the curve is on screen -- which is what makes it safe for the memory
    cap to evict a column at any moment.
    """

    __slots__ = ("x", "y", "label", "key", "curve_key", "source_x")

    def __init__(self, x: np.ndarray, y: np.ndarray, label: str, key: str,
                 curve_key: str = ""):
        self.x = x
        self.y = y
        self.label = label
        # The styling key: which colour, dash and marker this curve draws with.
        self.key = key
        # "file|sim|parameter" -- identity, not appearance. Two rebuilds compare
        # these to decide whether a draw item still shows the same curves and so
        # can be kept; see PlotTile._reuse_signature().
        self.curve_key = curve_key
        # 3D only. There `x` holds *projected* screen coordinates, because that
        # is what a click has to be hit-tested against, and this carries the x
        # parameter's own values for the same samples so the click can still be
        # reported in real units. None on a 2D tile, where `x` already is them.
        self.source_x = None


class DrawGroup:
    """Every curve that shares one pen, painted as a single item.

    This is what makes a study of a thousand runs viewable at all. A tile drew
    one pyqtgraph item per curve, and 1200 runs of two parameters is 2400 items
    holding 1.2 million points between them -- which the core produces in 35 ms
    and Qt then spends minutes on, because the cost is per *item*: a bounding
    rect, a paint call and a scene-graph node each.

    Curves that share a pen cannot be told apart on screen anyway, so they are
    concatenated into one polyline with a NaN between consecutive runs. The NaN
    breaks the line (connect="finite"), so nothing is joined that should not be.
    Twelve palette colours turn 2400 items into about twelve.

    `sampled_for` records the (range, budget) the drawn points were computed
    for, so a redraw that does not move the view -- a style change, an overlay
    update, a marker drag, a resize that does not change the width -- leaves the
    points alone instead of recomputing an identical answer.
    """

    __slots__ = ("item", "entries", "sampled_for")

    def __init__(self, item: pg.PlotDataItem):
        self.item = item
        self.entries: List[CurveEntry] = []
        self.sampled_for = None


class _ProjectedCurve:
    """One run of a 3D tile, in the form a rotation can reuse.

    The normalised cube coordinates rather than the samples: normalising needs
    the bounds across every selected run, which is a whole pass over the data,
    and it does not depend on the angle. `source_x` is the x parameter's own
    values for the same thinned samples, so a click on the projection can be
    reported back in the units the Inspect panel reads.
    """

    __slots__ = ("key", "nx", "ny", "nz", "source_x", "label", "style_key")

    def __init__(self, key, nx, ny, nz, source_x, label, style_key):
        self.key = key
        self.nx, self.ny, self.nz = nx, ny, nz
        self.source_x = source_x
        self.label = label
        self.style_key = style_key


class _Projection:
    """A 3D tile's angle-independent state: the cube, and who draws what."""

    __slots__ = ("bounds", "groups")

    def __init__(self, bounds, groups):
        self.bounds = bounds
        # [(DrawGroup, [_ProjectedCurve, ...])], in draw order.
        self.groups = groups


class _GroupPlan:
    """One group's appearance and membership, before an item is chosen for it.

    A rebuild cannot decide which existing draw item to keep until it knows a
    group's whole membership, and that is not known until every curve has been
    walked. So the walk fills these in, and PlotTile._adopt_groups() then
    matches them against the items already on the plot.
    """

    __slots__ = ("color", "dash", "width", "marker", "symbol_size", "name",
                 "entries")

    def __init__(self, color, dash, width, marker, symbol_size, name):
        self.color = color
        self.dash = dash
        self.width = width
        self.marker = marker
        self.symbol_size = symbol_size
        # The legend row this group owns, or None when it has none.
        self.name = name
        self.entries: List[CurveEntry] = []

    def _pen(self) -> QPen:
        pen = QPen(QColor(self.color))
        pen.setWidthF(self.width)
        pen.setStyle(_PEN_STYLES.get(self.dash, Qt.PenStyle.SolidLine))
        # Cosmetic: the width is in device pixels, so a zoom does not fatten
        # the line.
        pen.setCosmetic(True)
        return pen

    def build_item(self) -> pg.PlotDataItem:
        return pg.PlotDataItem(
            pen=self._pen(),
            antialias=False,
            symbol=self.marker,
            symbolSize=self.symbol_size,
            symbolBrush=self.color,
            symbolPen=pg.mkPen(self.color),
            name=self.name,
        )

    def restyle_item(self, item: pg.PlotDataItem) -> None:
        """Repaint an existing item in this group's colours.

        setSymbol and friends each mark the item dirty and schedule a repaint,
        which is exactly what a restyle should do -- and nothing here touches
        the data, so the points already computed stay valid. `name` is set
        directly because PlotDataItem has no setter for it; the legend reads
        opts["name"] through name().
        """
        item.setPen(self._pen())
        item.setSymbol(self.marker)
        item.setSymbolSize(self.symbol_size)
        item.setSymbolBrush(self.color)
        item.setSymbolPen(pg.mkPen(self.color))
        item.opts["name"] = self.name


class PlotTile(pg.PlotWidget):
    """A single plot showing one x parameter against several y parameters."""

    # Class attributes, not just instance ones set in __init__. Both are read
    # before they are written on some path -- _sync_report by the panel, which
    # can be refreshed before this tile has drawn anything -- and a missing
    # instance attribute on a partially built tile is the kind of absence that
    # reads as "some object" rather than as "not yet", which is how a guard
    # ends up testing the wrong thing. The defaults here are the truthful
    # answers for a tile that has not aligned anything.
    _sync_cache = None
    _sync_report = (0, 0)

    activated = Signal(int)
    # Double-click: fit this tile, and bring the panel that chooses its
    # parameters forward. Two things from one gesture, because they are the
    # two things wanted on arriving at a tile.
    params_requested = Signal(int)
    cursor_moved = Signal(str)
    view_changed = Signal()
    # (file index, sim index, x value) of a point clicked in Inspect mode.
    point_inspected = Signal(int, int, float)
    # (x0, y0, x1, y1) of a slope line dragged out in Inspect mode.
    slope_measured = Signal(float, float, float, float)
    # A marker or an area was dragged; every other tile has to catch up.
    annotation_moved = Signal()
    # The legend was dragged; the Style panel's corner combo has to follow.
    legend_moved = Signal()
    # A caption was dragged; the Text panel's position boxes have to follow.
    text_moved = Signal()
    # (low, high) in x: a zoom that asked every tile to follow. Emitted by the
    # tile the gesture happened in; the window hands it to the others.
    x_zoom_shared = Signal(float, float)
    # A double right-click: fit every tile, not just this one.
    fit_all_requested = Signal()

    # Class-level defaults, and they are load-bearing. pyqtgraph's
    # PlotWidget.__init__ calls setCentralItem, which calls resizeEvent(None) --
    # so our overridden Qt event handlers run *during* super().__init__(),
    # before __init__ has assigned any instance attribute. Worse, PlotWidget
    # defines __getattr__ to forward unknown names to its PlotItem, so a missing
    # attribute surfaces as a bare AttributeError rather than something
    # diagnosable. Defining them on the class makes the early call find a safe
    # value instead.
    _redraw_timer = None
    _curves: Dict[str, CurveEntry] = {}
    _groups: List["DrawGroup"] = []
    # None means "no sheet applied yet", so the first call always applies one.
    # See set_active_border() for why the state has to be remembered at all.
    _border_active = None
    _index = -1
    _suppress_range_event = True
    _slope_origin = None
    # An armed double-click fit: "one", "all", or None. Class attributes, so a
    # press that arrives before the first release reads as "nothing armed"
    # rather than as some object.
    _double_click = None
    _double_click_from = None
    _slope_item = None
    _slope_text = None
    _rotating_from = None
    _gizmo = None
    _texts = None
    _projection = None
    _inspect_dot = None
    _nav_action = None

    def __init__(self, session: Session, state: AppState, index: int,
                 theme: Theme, parent=None):
        # A view box of our own, for one reason: its rectangle zoom reads the
        # direction of the drag, so a flat drag takes x and leaves y alone.
        # It has to be given at construction -- the PlotItem builds its own
        # otherwise, and swapping one in afterwards leaves the axes linked to
        # the box that was replaced.
        super().__init__(parent=parent, viewBox=DirectionalViewBox())
        # A right-button or Ctrl zoom is answered by every tile, so the box
        # tells the tile the x it settled on and the tile passes it up.
        try:
            self.getViewBox().shared_x_zoom = self._announce_x_zoom
        except AttributeError:          # pragma: no cover - not our box
            pass
        self._session = session
        self._state = state
        self._index = index
        self._theme = theme

        self._curves = {}
        self._groups: List[DrawGroup] = []
        self._legend: Optional[pg.LegendItem] = None
        self._overlays: List[pg.GraphicsObject] = []
        # Moves delivered during the drag in progress, for the trace.
        self._drag_moves = 0
        # The violation bands of a comparison overlay. A list of their own
        # rather than _overlays, which _draw_overlays() clears on every pass:
        # these belong to the tile's data and not to the markers and areas
        # that come and go with the annotation tabs.
        self._fdiff_items: List[pg.GraphicsObject] = []
        self._suppress_range_event = False
        # What was last written to the PlotItem's own settings, so a rebuild
        # that would write the same thing writes nothing. See _apply_axes().
        self._axes_applied: Optional[tuple] = None
        # What the markers and bands currently on the plot were drawn from.
        # See _draw_overlays().
        self._overlays_drawn_for: Optional[tuple] = None
        # The alignment, and the shifted axes it produces. Both are caches:
        # see _sync_plan and _shifted_x.
        self._sync_cache = None
        self._shift_cache = {}

        # A window can never be made smaller than the sum of its parts, and a
        # 5x5 page is twenty-five plot widgets. QGraphicsView asks for a
        # generous minimum by default, so twenty-five of them added up to more
        # than a laptop screen is wide -- which is why the window could not be
        # shrunk, and why some Linux window managers refused to maximise it at
        # all rather than grant a geometry the widget said it could not accept.
        # A tile this small is not useful to read, but it is the grid's job to
        # decide that, not the tile's job to make the window unusable.
        self.setMinimumSize(60, 45)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        plot = self.getPlotItem()
        plot.setMenuEnabled(False)

        # clipToView is deliberately OFF, and this is a correctness matter
        # rather than a preference. pyqtgraph's implementation says of itself
        # that it "presumes that x-values have uniform spacing": it takes
        # dx = (x[-1] - x[0]) / (len(x) - 1) and slices the array by index
        # arithmetic from that. Curves that share a pen are drawn as one
        # polyline here -- several runs concatenated with a NaN between them --
        # so x runs 0..100, NaN, 0..100, NaN, 0..100, and both the spacing and
        # the ordering that slice depends on are false. It cut most of the runs
        # out of the item and the tile came back missing curves.
        #
        # That is why it showed up first in ISO colouring: colouring by ISO
        # value is what makes many runs share one pen, so a handful of runs was
        # enough to trigger it. Nothing is lost by turning it off -- the core
        # already decimates to the visible x range before the data crosses into
        # Python, which is the same saving done correctly.
        #
        # Downsampling is ours for the same reason: pyqtgraph's is stride-based
        # and would drop exactly the transients this tool exists to show.
        plot.setDownsampling(auto=False, mode="peak")
        plot.showGrid(x=True, y=True, alpha=0.25)

        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(25)
        self._redraw_timer.timeout.connect(self._on_redraw_timer)

        self.sigRangeChanged.connect(self._on_range_changed)

        # fill= is the whole point: the readout follows the cursor across the
        # curves, and without a ground behind it the figures were dark text on
        # a dark trace wherever one happened to be.
        self._crosshair_label = pg.TextItem(
            anchor=(0, 1), fill=_label_brush(theme.plot_background))
        self._crosshair_label.setFont(plot_font(PLOT_LABEL_PT))
        self._crosshair_label.setVisible(False)
        plot.addItem(self._crosshair_label, ignoreBounds=True)

        # The rubber-band slope line, created once and reused. Building it per
        # drag would add and remove a scene item on every mouse-move event.
        self._slope_origin = None
        self._rotating_from = None
        self._slope_item = pg.PlotCurveItem(
            pen=pg.mkPen("#ffb300", width=1.6, style=Qt.PenStyle.DashLine)
        )
        self._slope_item.setZValue(30)
        self._slope_item.setVisible(False)
        plot.addItem(self._slope_item, ignoreBounds=True)

        # The sample the Inspect panel is reporting. One item, reused: created
        # per click it would be a scene item added and removed on every press.
        self._inspect_dot = pg.ScatterPlotItem(
            size=INSPECT_DOT_SIZE, symbol="o",
            brush=pg.mkBrush(INSPECT_DOT_COLOUR),
            pen=pg.mkPen(INSPECT_DOT_EDGE, width=1.4),
        )
        self._inspect_dot.setZValue(INSPECT_DOT_Z)
        self._inspect_dot.setVisible(False)
        plot.addItem(self._inspect_dot, ignoreBounds=True)

        self._slope_text = pg.TextItem(anchor=(0, 1), color="#ffb300",
                                       fill=_label_brush(theme.plot_background))
        self._slope_text.setFont(plot_font(PLOT_LABEL_PT))
        self._slope_text.setZValue(31)
        self._slope_text.setVisible(False)
        plot.addItem(self._slope_text, ignoreBounds=True)

        # Parented to the view box rather than added to the plot, so the
        # pan/zoom transform -- which pyqtgraph applies to a child group --
        # never reaches it. That is what holds its size constant.
        self._gizmo = Axis3DGizmo(theme)
        self._gizmo.setParentItem(plot.getViewBox())
        self._gizmo.anchor(itemPos=(0, 1), parentPos=(0, 1), offset=(10, -10))
        self._gizmo.setVisible(False)

        self._texts = TextOverlay(self._on_text_moved,
                                  background=theme.plot_background)
        self._texts.setParentItem(plot.getViewBox())
        self._texts.setPos(0, 0)

        self.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.apply_mouse_mode()

    # ------------------------------------------------------------- helpers

    @property
    def index(self) -> int:
        """Position of this tile in the page, for the grid to report against."""
        return self._index

    @property
    def tile_state(self) -> TileState:
        page = self._state.page
        idx = min(self._index, len(page.tiles) - 1)
        return page.tiles[max(0, idx)]

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        # The border colour comes from the theme, so the cached state that lets
        # set_active_border() skip redundant work no longer describes what is on
        # screen. Forget it, and the next call repaints in the new colours.
        self._border_active = None
        if self._gizmo is not None:
            self._gizmo.set_theme(theme)
        if self._texts is not None:
            self._texts.set_background(theme.plot_background)
        # The ground painted behind every on-plot readout is the plot's own
        # background, so it has to follow the theme with it -- a light card left
        # on a dark plot is worse than no card.
        brush = _label_brush(theme.plot_background)
        for item in (self._crosshair_label, self._slope_text):
            try:
                item.fill = brush
            except (AttributeError, RuntimeError):
                pass
        self.setBackground(theme.plot_background)
        plot = self.getPlotItem()
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(theme.plot_grid))
            axis.setTextPen(pg.mkPen(theme.plot_foreground))
            # Tick numbers default to the widget font, which is set for the
            # panels; on a page of tiles they then take more room than the
            # curves they are labelling.
            font = plot_font(PLOT_AXIS_PT)
            axis.setStyle(tickFont=font)
            if axis_name == "bottom":
                _fit_tick_row(axis, font)
        self._crosshair_label.setColor(theme.plot_foreground)
        self.rebuild()

    def required_channels(self):
        """(channel names, whether this tile needs anything) for preloading."""
        tile = self.tile_state
        if tile.is_empty():
            return []
        names = [tile.x_param] + list(tile.y_params)
        if tile.plot3d and tile.z_param:
            names.append(tile.z_param)
        # A difference reads its channel from both of its runs, and those runs
        # need not be among the ones ticked for drawing -- so without this the
        # two columns would be read lazily inside the draw, one at a time, off
        # whatever share the file lives on.
        if tile.diff_param:
            names.append(tile.diff_param)
        # The alignment is found by reading the channel the condition is about,
        # so it has to be resident like any other -- and unlike the ISO
        # parameters below, the whole column is needed, not its first sample.
        if tile.sync_enabled and tile.sync_param:
            names.append(tile.sync_param)
        # Colouring by ISO does *not* put the ISO parameter here, and that is
        # the point. iso_values() reads the first record of each block through
        # the index -- it touches no column memory at all, which is what made
        # the ISO scan 683x faster than loading the column and taking sample
        # zero. Preloading it would parse the whole of, say, MACH across every
        # run of a 550 MB file to answer a question already answered from the
        # header, and then hold it against the memory cap.

        # An area or marker defined on another parameter still has to be drawn
        # here, as a highlighted span or a dot, so its reference channel has to
        # be resident too.
        for area in self._state.areas:
            if area.visible and area.x_param and area.x_param != tile.x_param:
                names.append(area.x_param)

        # A marker defined on TIME needs TIME resident even on a tile plotted
        # against something else, because that is how its position is found.
        for marker in self._state.markers:
            if marker.visible and marker.x_param and marker.x_param != tile.x_param:
                names.append(marker.x_param)
        return names

    def _axis_label(self, param: str) -> str:
        return self._session.label(param) if param else ""

    def _y_axis_label(self, tile: TileState, max_names: int = 3) -> str:
        """The y axis caption, with units when the parameters agree on one.

        Several channels can share an axis; if they all carry the same unit it
        is stated once at the end, and if they disagree it is left off rather
        than labelling the axis with a unit that is wrong for some of the
        curves on it.
        """
        names = tile.y_params
        if not names:
            return ""
        shown = ", ".join(names[:max_names])
        if len(names) > max_names:
            shown += "..."

        units = {self._session.unit(name) for name in names}
        if len(units) == 1:
            unit = units.pop()
            if unit:
                return "{0} [{1}]".format(shown, unit)
        return shown

    def show_placeholder(self, message: str) -> None:
        """Replace the plot contents with a message while data is loading."""
        plot = self.getPlotItem()
        for group in self._groups:
            plot.removeItem(group.item)
        self._groups = []
        self._curves = {}
        if self._legend is not None:
            self._legend.clear()
            self._legend.setVisible(False)
        plot.setTitle(message, color=self._theme.text_dim,
                      size="{0}pt".format(pt(PLOT_TITLE_PT)))
        # The title on screen is now this message, not the one _apply_axes()
        # last wrote, so what it remembers writing no longer describes the
        # plot. Forgetting it is what makes the next rebuild put the real
        # title back -- without this a tile that showed a placeholder while a
        # file loaded kept the placeholder as its heading.
        self._axes_applied = None

    # -------------------------------------------------------- mouse modes

    def apply_mouse_mode(self) -> None:
        """Configure the view box for the current mode.

        Each mode genuinely turns the others off. In Select and Inspect the view
        box has mouse interaction disabled outright, so a stray drag cannot pan
        the plot; in Pan the wheel is swallowed (see wheelEvent) so scrolling
        cannot zoom by accident while you are dragging things about.
        """
        mode = self._state.mouse_mode
        view = self.getPlotItem().vb

        if mode != MouseMode.CROSSHAIR:
            # The dot marks what the Inspect panel is reporting, and outside
            # Inspect the panel is not reporting anything.
            self.clear_inspect_dot()

        if mode in (MouseMode.SELECT, MouseMode.CROSSHAIR):
            view.setMouseEnabled(x=False, y=False)
            view.setMouseMode(pg.ViewBox.PanMode)
        elif mode == MouseMode.PAN:
            view.setMouseEnabled(x=True, y=True)
            view.setMouseMode(pg.ViewBox.PanMode)
        else:  # ZOOM
            view.setMouseEnabled(x=True, y=True)
            view.setMouseMode(pg.ViewBox.RectMode)

        tile = self.tile_state
        if not tile.plot3d:
            # A locked axis stays locked whatever the mouse mode says -- and,
            # just as importantly, an axis that is *no longer* locked stops
            # being clamped. setLimits is a hard bound held on the view box,
            # not on the tile, so it outlives the lock that set it: while this
            # ran only for a tile that was still locked, clearing a lock left
            # the old bound in force until something else happened to call
            # _apply_axis_limits, and a fit made in between was silently
            # clamped to the range the user had just released.
            self._apply_axis_limits(tile)

        if self.tile_state.plot3d:
            # Pan is repurposed as rotate on a 3D tile, so the view box must not
            # also try to move -- but Zoom keeps its meaning: the wheel scales
            # the projection (handled in wheelEvent) and a drag rubber-bands a
            # region of it, which needs the view box left enabled.
            enabled = mode == MouseMode.ZOOM
            view.setMouseEnabled(x=enabled, y=enabled)
            view.setMouseMode(pg.ViewBox.RectMode)

        zoomtrace.log("mousemode", tile=self._index, mode=mode.name,
                      plot3d=tile.plot3d,
                      **zoomtrace.view_state(view))

        movable = self._annotations_movable()
        for overlay in self._overlays:
            if isinstance(overlay, (pg.InfiniteLine, pg.LinearRegionItem)):
                overlay.setMovable(movable)

        cursors = {
            MouseMode.SELECT: Qt.CursorShape.ArrowCursor,
            MouseMode.CROSSHAIR: Qt.CursorShape.CrossCursor,
            MouseMode.PAN: Qt.CursorShape.OpenHandCursor,
            MouseMode.ZOOM: Qt.CursorShape.CrossCursor,
        }
        self.setCursor(cursors.get(mode, Qt.CursorShape.ArrowCursor))

    def wheelEvent(self, event) -> None:
        # Traced too: "the zoom stopped working" may mean the wheel as much as
        # the band, and the two fail for different reasons.
        try:
            notches = int(event.angleDelta().y())
        except (AttributeError, TypeError, ValueError):
            notches = "?"
        zoomtrace.log("wheel", tile=self._index,
                      mode=self._state.mouse_mode.name,
                      delta=notches, plot3d=self.tile_state.plot3d,
                      **zoomtrace.view_state(self.getPlotItem().vb))

        # Pan mode is "move without zooming", so the wheel must do nothing --
        # otherwise a scroll during a drag silently rescales the plot.
        if self._state.mouse_mode in (MouseMode.SELECT, MouseMode.CROSSHAIR,
                                      MouseMode.PAN):
            event.accept()
            return

        if self.tile_state.plot3d:
            # The view box has its mouse handling disabled on a 3D tile (a drag
            # rotates instead of panning), and that also swallows the wheel --
            # so the zoom has to be applied explicitly rather than left to
            # pyqtgraph.
            delta = event.angleDelta().y()
            if delta:
                # 1.0015^delta: one notch is 120 units, so about 20% per notch,
                # and a trackpad's smaller deltas scale smoothly rather than
                # jumping.
                self.zoom_3d(1.0015 ** -delta)
            event.accept()
            return

        super().wheelEvent(event)

    def set_active_border(self, active: bool) -> None:
        """Mark this tile as the one the side panels are editing.

        The early return is not a micro-optimisation, it is the fix for the
        whole grid stalling on a single click. QWidget.setStyleSheet does no
        comparison of its own: handed the string it already has, it still
        repolishes the widget and every child and schedules a repaint. A tile is
        a QGraphicsView, so that repaint is the entire scene -- every curve,
        every point. And _refresh_active_border() calls this on *every* tile,
        on every mouse press anywhere in the grid, because a press emits
        `activated` whether or not the active tile actually changes.

        So one click on a 5x5 page repainted twenty-five plots holding a million
        points between them, to change a border on at most two of them. Applying
        the sheet only when the state really flips leaves exactly those two.
        """
        active = bool(active)
        if active == self._border_active:   # None on the first call
            return
        self._border_active = active
        color = self._theme.accent if active else self._theme.border
        self.setStyleSheet(
            "PlotTile {{ border: {0}px solid {1}; }}".format(2 if active else 1, color)
        )

    def _apply_axes(self, plot, tile: TileState) -> None:
        """The PlotItem's own settings: grid, log mode, axis captions, title.

        Written only when one of them has actually changed. They used to be
        written on every rebuild, and most rebuilds change none of them -- a
        restyled curve, a run ticked, a marker dragged, a page returned to.

        setLogMode is the one that made that expensive rather than merely
        redundant. pyqtgraph propagates it to every data item on the plot,
        which invalidates each curve's cached display data, and then re-enables
        auto-range; the tile then has to undo that in _restore_view(). So a
        rebuild the redraw cache had just arranged to cost nothing threw away
        the transformed points of every curve on the tile and moved the frame,
        twice, for a setting nobody had touched.

        The title is in the same tuple rather than guarded separately: it is a
        pure function of the tile's parameters, so the one comparison answers
        for all of it, and setTitle relays the PlotItem too.
        """
        title = tile_title(tile).replace("\n", "<br>")
        signature = (tile.show_grid, tile.log_y, tile.plot3d, title,
                     self._theme.plot_foreground, ui_scale())
        if signature == self._axes_applied:
            return
        self._axes_applied = signature

        if tile.plot3d:
            # The axes of a projection are not either data axis, so labelling
            # them with parameter names would be a lie. The title carries the
            # three names instead.
            plot.showGrid(x=False, y=False)
        else:
            plot.showGrid(x=tile.show_grid, y=tile.show_grid, alpha=0.25)
        plot.setLogMode(x=False, y=tile.log_y and not tile.plot3d)
        # Left blank since 2.1.0, in both cases. The title names every
        # parameter on the tile and the x channel -- "ALPHA, NZ = f(TIME)" --
        # so a caption beside the axis repeated it and spent a strip of a tile
        # that is often one ninth of a page. The unit reaches the reader
        # through the tick values and the INSPECT tables instead.
        # Hidden rather than set to an empty string. setLabel() shows the
        # label whatever text it is given, so an empty caption still reserved a
        # whole line beside the axis -- a strip of a tile that is often one
        # ninth of a page, spent on nothing, and part of why the tick numbers
        # along the bottom had no room.
        for axis_name in ("bottom", "left"):
            try:
                plot.getAxis(axis_name).showLabel(False)
            except (AttributeError, RuntimeError):
                plot.setLabel(axis_name, "", **_axis_label_style())
        plot.setTitle(title, color=self._theme.plot_foreground,
                      size="{0}pt".format(pt(PLOT_TITLE_PT)))

    # ---------------------------------------------------------- ISO lookup

    # -------------------------------------------------------------- sync

    def _sync_plan(self, tile: TileState, by_file):
        """How far each run moves, plus what to tell the user about it.

        Returns (offsets, found, missing): the shift per run, how many runs the
        condition was located in, and how many it was not. A run with no entry
        in `offsets` is drawn where it always was.

        Cached on everything the answer depends on, including the address and
        length of both channels of every run -- so a reload, or a column
        evicted and read back, recomputes, while the dozen rebuilds a restyle
        or a marker drag causes do not. Without the cache this would scan the
        condition channel of every run on every rebuild.
        """
        if not (tile.sync_enabled and tile.sync_param and tile.x_param):
            self._sync_cache = None
            return {}, 0, 0

        runs = [(file_index, sim_index)
                for file_index, sims in sorted(by_file.items())
                for sim_index in sims]

        series = []
        probe = []
        for key in runs:
            x = self._session.channel(key[0], key[1], tile.x_param)
            y = self._session.channel(key[0], key[1], tile.sync_param)
            series.append((key, x, y))
            probe.append((x.ctypes.data, x.size, y.ctypes.data, y.size))

        signature = (tile.sync_param, tile.sync_condition, tile.sync_level,
                     tile.sync_target, tile.sync_target_value, tile.x_param,
                     tuple(runs), tuple(probe))
        if self._sync_cache is not None and self._sync_cache[0] == signature:
            return self._sync_cache[1]

        events = {key: sync.find_event(x, y, tile.sync_condition,
                                       tile.sync_level)
                  for key, x, y in series}
        shifts = sync.offsets(events, tile.sync_target, tile.sync_target_value,
                              order=runs)
        found = sum(1 for event in events.values() if event.found)
        answer = (shifts, found, len(events) - found)
        self._sync_cache = (signature, answer)
        return answer

    def _shifted_x(self, file_index: int, sim_index: int, x: np.ndarray,
                   offset: float) -> np.ndarray:
        """`x` moved by `offset`, as the same array object every time.

        The identity matters as much as the arithmetic. _reuse_signature
         compares buffer addresses to decide whether a group still draws the
        same thing, so handing out a fresh copy per rebuild would make every
        group look new and re-decimate the whole page on every restyle. Keyed
        on the source buffer, so a reload or an eviction produces a new one.
        """
        if not offset:
            return x
        # setdefault on the instance dictionary rather than a class attribute:
        # a mutable default on the class would be one cache shared by every
        # tile of the page, and two tiles aligning the same run differently
        # would overwrite each other's axis.
        cache = self.__dict__.setdefault("_shift_cache", {})
        key = (file_index, sim_index)
        cached = cache.get(key)
        if (cached is not None and cached[0] == x.ctypes.data
                and cached[1] == x.size and cached[2] == offset):
            return cached[3]
        shifted = sync.shift(x, offset)
        cache[key] = (x.ctypes.data, x.size, offset, shifted)
        return shifted

    def _iso_labels(self, iso_param: str):
        """(file, sim) -> ISO label, plus label -> where it sits in the legend.

        Looking the label up per curve would call into the core for every
        parameter of every run; the value is constant within a run, so once per
        run is enough.

        The second half is what keeps the legend still. Legend rows are emitted
        in the order the draw groups are built, and that order used to be the
        order the groups were first *seen* while walking the selected runs --
        so which ISO group came first depended on which run happened to be
        selected. Tick one run off and the rows reshuffled: 0-10, 10-15, 15-20
        became 15-20, 0-10, 10-15, with the right colours on the right curves
        and the legend in an order nobody chose.

        So each label is ranked by something intrinsic to it instead. Grouped,
        that is the interval's position in the grouping the user set up;
        ungrouped, it is the value itself, which sorts ascending. Computed in
        this same pass because both come out of the same lookup, and this
        lookup is the expensive one.
        """
        labels: Dict[Tuple[int, int], str] = {}
        ranks: Dict[str, float] = {}
        if not iso_param:
            return labels, ranks

        iso_state = self._state.iso_states.get(iso_param)
        grouped = bool(iso_state and iso_state.grouped and iso_state.groups)
        groups = iso_state.groups if grouped else []

        # Sorted, not a bare set. Set iteration order is not something to
        # build a legend on, even when small integers make it look sorted.
        for file_index in sorted({f for f, _s in self._state.plotted_sims()}):
            values = self._session.iso_values(file_index, iso_param)
            for sim_index in range(values.size):
                value = float(values[sim_index])
                if value != value:  # NaN
                    continue
                if grouped:
                    label = ""
                    rank = 0.0
                    for index, (name, low, high, _active) in enumerate(groups):
                        if low <= value <= high:
                            label, rank = name, float(index)
                            break
                    if not label:
                        continue
                else:
                    label = self._session.format_value(value)
                    rank = value
                labels[(file_index, sim_index)] = label
                # The lowest wins, because two runs can carry the same label
                # and a row's place must not depend on which was visited
                # first. Ungrouped, two distinct values can also format to one
                # label -- 0.751 and 0.752 both read "0.75" -- and then the
                # smaller of them is what the row is sorted by.
                if label not in ranks or rank < ranks[label]:
                    ranks[label] = rank
        return labels, ranks

    def _style_key(self, tile: TileState, param: str, sim_key: str,
                   iso_label: str, file_id: str = "",
                   style_iso_label: str = "") -> str:
        return curve_color_key(tile, param, sim_key, iso_label, file_id)

    def _legend_label(self, tile: TileState, param: str, file_name: str,
                      sim_id: str, iso_label: str, dash: str, symbol: str,
                      style_iso_label: str = "") -> str:
        """The text of one legend row.

        Both ISO identities are written, not just the colour's. A curve can be
        coloured by one ISO parameter and dashed by another -- or, the case
        that was plainly broken, coloured by Parameter and dashed by ISO. Then
        `iso_label` is empty, because colour is not keying on ISO and no colour
        labels were looked up at all, and "show ISO in the legend" produced
        nothing: every ISO value collapsed onto one row per parameter, since
        rows with identical text are merged on purpose.

        Written once when the two modes key on the same parameter, so the
        common "colour by ISO and dash by ISO" pairing does not say
        `MACH = 0.75 | MACH = 0.75`.
        """
        # The dash pattern and marker are shown by the legend's own sample of
        # the curve, so they are deliberately not spelled out in the text.
        del dash, symbol
        parts: List[str] = []

        if tile.legend_show_param:
            parts.append(param)

        if tile.legend_show_iso:
            written = set()
            for name, value in (
                (tile.color_iso_param if tile.color_mode == ColorMode.ISO else "",
                 iso_label),
                (tile.style_iso_param if tile.style_mode == StyleMode.ISO else "",
                 style_iso_label),
            ):
                if not name or not value or name in written:
                    continue
                written.add(name)
                parts.append("{0} = {1}".format(name, value))

        if tile.legend_show_file:
            parts.append(file_name)
        if tile.legend_show_sim:
            parts.append(sim_id)

        return " | ".join(parts) if parts else "Line"

    # ------------------------------------------------------------- legend

    def _ensure_legend(self, plot) -> None:
        """Attach our own legend once, so drags can be observed.

        PlotItem.addLegend() builds a plain LegendItem and stores it on the
        plot; there is no hook for "the user moved it". Building ours and
        registering it the same way keeps every other pyqtgraph path working
        while giving the drag somewhere to report to.
        """
        if isinstance(self._legend, DraggableLegend):
            # Already built, but the theme -- or the UI scale -- may have
            # changed under it since. The row text carries its own size, set
            # once at construction, so it has to be written again or the
            # legend stays at the size it was born at while everything around
            # it moves.
            self._legend.set_ground(**legend_ground(self._theme))
            self._legend.set_metrics()
            return
        if plot.legend is not None:
            scene = plot.legend.scene()
            if scene is not None:
                scene.removeItem(plot.legend)
            plot.legend = None

        legend = DraggableLegend(self._on_legend_moved, offset=(10, 10),
                                 labelTextSize="{0}pt".format(pt(PLOT_LEGEND_PT)),
                                 **legend_ground(self._theme))
        legend.setParentItem(plot.getViewBox())
        legend.setZValue(LEGEND_Z)
        plot.legend = legend
        self._legend = legend

    def _place_legend(self, tile: TileState) -> None:
        if self._legend is None:
            return
        if tile.legend_position == "custom":
            area = self.getPlotItem().getViewBox().boundingRect()
            # The same call the drag makes, so a restored position and a
            # freshly dragged one behave identically on resize.
            self._legend.autoAnchor(
                QPointF(tile.legend_x * area.width(),
                        tile.legend_y * area.height())
            )
            return
        self._legend.anchor(
            itemPos=_LEGEND_ANCHORS.get(tile.legend_position, (0, 0)),
            parentPos=_LEGEND_ANCHORS.get(tile.legend_position, (0, 0)),
            offset=_LEGEND_OFFSETS.get(tile.legend_position, (10, 10)),
        )

    def _on_legend_moved(self, fraction) -> None:
        """The user dragged the legend somewhere that is not a corner.

        Rather than snapping it back on the next redraw -- which is what used
        to happen -- the position becomes the tile's, and the corner setting
        switches to "custom" so the combo tells the truth about what is on
        screen.
        """
        if fraction is None:
            return
        tile = self.tile_state
        tile.legend_position = "custom"
        # Clamped so a legend dragged past the edge can still be found again.
        tile.legend_x = max(-0.05, min(0.98, fraction[0]))
        tile.legend_y = max(-0.05, min(0.98, fraction[1]))
        self._state.dirty = True
        self.legend_moved.emit()

    # ------------------------------------------------------------ rebuild

    def rebuild(self) -> None:
        """Recreate every curve from the current selection and tile settings.

        Most rebuilds do not change a single drawn point. Toggling the grid,
        restyling one colour, dragging a marker, editing the legend, coming back
        to a page -- all of them arrive here, and all of them used to throw away
        every item and re-decimate every curve from full resolution for an
        answer identical to the one already on screen.

        So the old items are kept until the new grouping is known, and any group
        drawing exactly the same curves from exactly the same buffers is reused
        as it stands: its pen is updated in place and, because it keeps its
        `sampled_for`, the resample below skips it entirely. Only groups whose
        membership genuinely changed are rebuilt, and only those are re-drawn.
        """
        plot = self.getPlotItem()

        # Kept, not removed: _adopt_groups() below decides what can be reused,
        # and _drop_groups() takes away whatever is left. Holding them also
        # holds a reference to their sample buffers, which is what makes the
        # pointer comparison in _reuse_signature() safe -- a buffer that is
        # still referenced cannot have been freed and another put at its address.
        previous_groups = self._groups
        self._groups = []
        self._curves = {}
        # Rebuilt below if this is still a 3D tile. Dropped first either way:
        # a cache that outlived the thing it describes is worse than none.
        self._projection = None

        # Overlays are NOT dropped here. _draw_overlays() decides at the end
        # whether what is on the plot still describes the annotations and the
        # curves, and keeps the items when it does -- see its docstring.

        # pyqtgraph keeps exactly one legend per PlotItem and addLegend() hands
        # back the existing one, so recreating it per rebuild is not possible --
        # every rebuild would append another copy of every label. Clear its
        # entries instead and reuse it.
        self._legend = plot.legend
        if self._legend is not None:
            self._legend.clear()

        tile = self.tile_state

        # The comparison's bands, dropped unless this tile is still one of its
        # overlays. They belong to the tile *state*, but they are drawn on the
        # tile *widget* -- and the widgets are reused across pages, which is
        # what makes changing page quick. So a widget that drew page one's
        # sixteenth overlay and is now drawing page two's second tile, or a
        # tile on a page that has nothing to do with the comparison, arrives
        # here still holding the bands of whatever it drew last. Only
        # _rebuild_file_diff() puts them back.
        if not tile.is_file_diff() and self._fdiff_items:
            self._clear_violation_bands()
        self._slope_item.setVisible(False)
        self._slope_text.setVisible(False)
        # 2D and 3D want different things from the same mouse mode, so the
        # bindings are re-applied whenever the tile is rebuilt.
        self.apply_mouse_mode()

        self._apply_axes(plot, tile)

        if tile.plot3d:
            # A projection stores screen coordinates, not samples, so nothing
            # here can be reused: _rebuild_3d builds its own groups.
            self._drop_groups(previous_groups)
            self._rebuild_3d(plot, tile)
            self._draw_overlays()
            self._refresh_texts()
            return

        # Leaving 3D has to undo what it set, or a normal plot inherits a
        # square aspect ratio and draws into a fraction of the tile.
        plot.getViewBox().setAspectLocked(False)
        self._gizmo.setVisible(False)

        # A difference names its own two runs, so it draws whether or not
        # anything is ticked in DATA: a difference is a question about two
        # particular runs, and having to tick them first -- which would also
        # draw them -- is a step in the way of the question.
        #
        # The comparison's overlays are the same case for the same reason.
        # They name their run and their two files, and they were generated to
        # answer one question about it -- so unticking a run in DATA
        # afterwards must not blank the page that answered it.
        if tile.is_empty() or (not self._state.plotted_sims()
                               and not tile.is_diff()
                               and not tile.is_file_diff()):
            self._drop_groups(previous_groups)
            if self._legend is not None:
                self._legend.setVisible(False)
            # An empty tile has no frame worth keeping, so the next parameter
            # dropped on it gets fitted rather than inheriting the range left
            # over from whatever used to be here.
            tile.view_initialised = False
            self._sync_report = (0, 0)
            self._draw_overlays()
            self._refresh_texts()
            return

        if tile.show_legend:
            self._ensure_legend(plot)
            self._legend.setVisible(True)
        elif self._legend is not None:
            self._legend.setVisible(False)

        if tile.is_diff():
            # One curve from two runs, so none of the walk below applies: there
            # is nothing to group, nothing to rank and no legend to cap.
            self._rebuild_diff(plot, tile, previous_groups)
            self._frame_after_rebuild(tile)
            self._draw_overlays()
            self._refresh_texts()
            return

        if tile.is_file_diff():
            # Two curves of one channel, from two files, and the bands the
            # comparison found between them. Like a diff, it ignores what is
            # ticked in PARAMS and DATA: the tile was generated to answer one
            # question about one run, and a filter changed three tabs away
            # should not empty it.
            self._rebuild_file_diff(plot, tile, previous_groups)
            self._frame_after_rebuild(tile)
            self._draw_overlays()
            self._refresh_texts()
            return

        # No preload here. The grid hoists it into one background call covering
        # every visible tile, so the data is already resident by the time this
        # runs -- see PlotGrid.required_channels() and MainWindow.refresh_plots().
        by_file: Dict[int, List[int]] = {}
        for file_index, sim_index in self._state.plotted_sims():
            by_file.setdefault(file_index, []).append(sim_index)

        # Before the walk below, because every curve of a run is shifted by
        # the same amount and finding that amount reads a whole channel.
        sync_offsets, sync_found, sync_missing = self._sync_plan(tile, by_file)
        self._sync_report = (sync_found, sync_missing)

        iso_labels, iso_ranks = (
            self._iso_labels(tile.color_iso_param)
            if tile.color_mode == ColorMode.ISO
            else ({}, {})
        )
        style_iso_labels, style_iso_ranks = (
            self._iso_labels(tile.style_iso_param)
            if tile.style_mode == StyleMode.ISO
            else ({}, {})
        )

        styles = self._state.styles
        # Identical legend entries are merged when neither file nor simulation
        # is shown. That is what turns a 50-run ISO plot into one legend row per
        # ISO value instead of fifty rows saying the same thing.
        deduplicate = not tile.legend_show_file and not tile.legend_show_sim
        seen_labels: Set[str] = set()
        # The rows a legend actually shows, as (text, how the sample is drawn).
        # The budget below is spent on these rather than on curves: two files
        # of many runs each, coloured by file and labelled by file, are two
        # rows however many runs they hold, and counting the curves instead
        # spent the whole budget inside the first file and left the second
        # with no row at all.
        seen_rows: Set[tuple] = set()
        hidden_row_keys: Set[tuple] = set()
        legend_rows = 0

        # (pen signature, legend name) -> plan. Curves that look identical are
        # painted together; see DrawGroup for why that is the difference between
        # a thousand-run study being viewable and not.
        #
        # A plan rather than a DrawGroup, because which old item can be reused
        # is only knowable once a group's whole membership is known, and that is
        # not true until this loop has finished.
        plans: Dict[tuple, _GroupPlan] = {}
        order: List[tuple] = []
        # Where each group belongs in the legend. Kept separately from `order`
        # because a group's place is decided by the curves in it, and which
        # curves those are is not known until the walk below has finished.
        ranks: Dict[tuple, tuple] = {}
        param_rank = {name: index for index, name in enumerate(tile.y_params)}
        line_order = {}

        for file_index, sims in sorted(by_file.items()):
            handle = self._session.file(file_index)
            if handle is None:
                continue
            file_name = handle.name
            simulations = handle.simulations

            for sim_index in sims:
                if sim_index >= len(simulations):
                    continue
                x = self._session.channel(file_index, sim_index, tile.x_param)
                if x.size == 0:
                    continue
                offset = sync_offsets.get((file_index, sim_index))
                if offset:
                    x = self._shifted_x(file_index, sim_index, x, offset)

                sim = simulations[sim_index]
                iso_label = iso_labels.get((file_index, sim_index), "")

                for param in tile.y_params:
                    y = self._session.channel(file_index, sim_index, param)
                    if y.size == 0:
                        continue

                    style_iso_label = style_iso_labels.get((file_index, sim_index), "")
                    file_id = handle.path or file_name
                    color_key = curve_color_key(tile, param, sim.key,
                                                iso_label, file_id)
                    line_key = curve_line_key(
                        tile, param, sim.key, file_id, iso_label, style_iso_label)
                    if tile.style_mode != StyleMode.NONE:
                        if line_key not in line_order:
                            line_order[line_key] = len(line_order)
                        styles.assign_line_style(line_key, line_order[line_key])
                    color = styles.color_for(color_key)
                    dash = styles.dash_for(line_key)
                    symbol = styles.symbol_for(line_key)
                    width = styles.width_for(color_key)
                    style_key = color_key

                    label = self._legend_label(
                        tile, param, file_name, sim.sim_id, iso_label, dash,
                        symbol, style_iso_label
                    )

                    marker = symbol or ("o" if tile.show_points else None)

                    show_in_legend = tile.show_legend
                    if show_in_legend and deduplicate:
                        if label in seen_labels:
                            show_in_legend = False
                        else:
                            seen_labels.add(label)
                    if show_in_legend:
                        # A row this legend already has is not a second row,
                        # whatever the mode: same text, same colour, same dash
                        # and same marker is the same row drawn twice.
                        row_key = (label, color, dash, width, marker,
                                   styles.symbol_size)
                        if row_key in seen_rows:
                            show_in_legend = False
                        elif legend_rows >= MAX_LEGEND_ROWS:
                            show_in_legend = False
                            hidden_row_keys.add(row_key)
                        else:
                            seen_rows.add(row_key)
                            legend_rows += 1

                    key = "{0}|{1}|{2}".format(file_index, sim_index, param)
                    entry = CurveEntry(x, y, label, style_key, key)
                    self._curves[key] = entry

                    # The legend name is part of the grouping key, so a curve
                    # that owns a legend row is never merged into another
                    # curve's row -- the legend stays exactly as it was.
                    group_key = (color, dash, width, marker, styles.symbol_size,
                                 label if show_in_legend else None)
                    plan = plans.get(group_key)
                    if plan is None:
                        plan = _GroupPlan(color, dash, width, marker,
                                          styles.symbol_size,
                                          label if show_in_legend else None)
                        plans[group_key] = plan
                        order.append(group_key)
                    plan.entries.append(entry)

                    # Rank by what the curve *is*, never by when it turned up.
                    # Whichever field the colour mode is keying on leads, since
                    # that is what the legend rows are distinguishing.
                    #
                    # Style-by-ISO ranks second wherever colour is not already
                    # keying on ISO. Without it a tile coloured by parameter
                    # and dashed by ISO emitted its rows in the order the runs
                    # happened to be walked, which is the same reshuffling
                    # _iso_labels() went to the trouble of ranking away.
                    style_rank = style_iso_ranks.get(style_iso_label,
                                                     float("inf"))
                    if tile.color_mode == ColorMode.ISO:
                        rank = (iso_ranks.get(iso_label, float("inf")),
                                param_rank.get(param, 0),
                                file_index, sim_index)
                    elif tile.color_mode == ColorMode.PARAMETER:
                        rank = (param_rank.get(param, 0), style_rank,
                                file_index, sim_index)
                    else:
                        rank = (file_index, sim_index, style_rank,
                                param_rank.get(param, 0))
                    seen_rank = ranks.get(group_key)
                    if seen_rank is None or rank < seen_rank:
                        ranks[group_key] = rank

        # The fix for the reshuffling legend. Rows are emitted in the order the
        # groups are built -- pyqtgraph fills the legend from addItem(), and an
        # adopted item has its row put back in the same loop -- so this sort is
        # the legend's order. Before it, that order was the order the groups
        # were first seen while walking the selected runs, which is why ticking
        # one run off could turn 0-10, 10-15, 15-20 into 15-20, 0-10, 10-15:
        # the right colours on the right curves, in an order nobody chose.
        #
        # Stable, so anything the ranks genuinely tie on keeps the order it
        # was found in rather than swapping about between rebuilds.
        order.sort(key=lambda key: ranks[key])

        self._adopt_groups(plot, previous_groups,
                           [plans[k] for k in order])

        # Rows, not curves: the note has to count what the user would have
        # seen, which is one row per distinct entry.
        hidden_rows = len(hidden_row_keys)
        if hidden_rows and self._legend is not None:
            # Say what was left out, and say that it is a cap rather than
            # missing data: the curves those rows would have named are all
            # drawn, and turning File or Simulation off in STYLE merges the
            # identical rows and usually brings the count under the cap.
            spacer = pg.PlotDataItem(
                pen=None,
                name="... and {0} more (legend caps at {1} rows)".format(
                    hidden_rows, MAX_LEGEND_ROWS))
            plot.addItem(spacer)
            self._groups.append(DrawGroup(spacer))

        # Placed now, not before the rows were added. A legend anchored while
        # empty and then filled changes size under its own anchor, and
        # pyqtgraph re-derives the anchor from whichever corner the item is
        # nearest -- so a legend dragged to the middle crept a little further
        # on every rebuild until it was back against an edge. Its final size is
        # only known once its rows are in it.
        if tile.show_legend and self._legend is not None:
            self._place_legend(tile)

        self._frame_after_rebuild(tile)
        self._draw_overlays()
        self._refresh_texts()

    # -------------------------------------------------------------- diff

    def _rebuild_file_diff(self, plot, tile, previous_groups) -> None:
        """Overlay one channel of one run as two files recorded it.

        The counterpart of _rebuild_diff and not a variant of it: a difference
        curve answers "by how much", and this answers "where, and does it
        matter". Both series are drawn as they were recorded, and the bands
        the comparison computed are washed in behind them -- so the eye reads
        the two traces parting company and the colour says which threshold
        that crossing broke.

        The bands are not recomputed here. They were computed once, against a
        configuration that may since have been edited, and they are carried on
        the tile; redoing the arithmetic on every rebuild would also mean
        redoing it on every pan.
        """
        run_a = self._fdiff_run(tile.fdiff_file_a, tile.fdiff_run)
        run_b = self._fdiff_run(tile.fdiff_file_b, tile.fdiff_run)

        plans = []
        for run, colour, handle_index in (
                (run_a, FDIFF_COLOR_A, tile.fdiff_file_a),
                (run_b, FDIFF_COLOR_B, tile.fdiff_file_b)):
            if run is None:
                continue
            x = self._diff_channel(run, tile.x_param)
            y = self._diff_channel(run, tile.fdiff_param)
            if x is None or y is None:
                continue
            count = min(x.size, y.size)
            if count < 1:
                continue
            label = self._fdiff_label(handle_index, tile)
            # Through the registry, so STYLE can restyle a generated page like
            # any other, while the first drawing is the fixed pair of colours.
            key = "fdiff|{0}".format(handle_index)
            styles = self._state.styles
            if not styles.has_color(key):
                styles.set_color(key, colour)
            entry_key = "fdiff|{0}|{1}|{2}".format(
                handle_index, tile.fdiff_run, tile.fdiff_param)
            entry = CurveEntry(x[:count], y[:count], label, key, entry_key)
            self._curves[entry_key] = entry
            plan = _GroupPlan(styles.color_for(key), styles.dash_for(key),
                              styles.width_for(key),
                              styles.symbol_for(key)
                              or ("o" if tile.show_points else None),
                              styles.symbol_size,
                              label if tile.show_legend else None)
            plan.entries.append(entry)
            plans.append(plan)

        if not plans:
            # Neither file carries the run or the channel any more -- the files
            # were closed, or reloaded from a build that renamed things. The
            # saved bands are not drawn over an empty tile: they would be a
            # verdict with nothing to judge.
            self._drop_groups(previous_groups)
            self._clear_violation_bands()
            if self._legend is not None:
                self._legend.setVisible(False)
            self._sync_report = (0, 0)
            return

        self._adopt_groups(plot, previous_groups, plans)
        self._draw_violation_bands(plot, tile)
        self._sync_report = (0, 0)

        if tile.show_legend and self._legend is not None:
            self._place_legend(tile)

    def _fdiff_run(self, file_index: int, run_id: str):
        """The (file, run) pair for a run named by identifier, or None.

        By name, because that is what the two files were matched on: the run
        the user is looking at can sit at a different index in each of them.
        """
        handle = self._session.file(file_index)
        if handle is None or not run_id:
            return None
        for sim_index, sim in enumerate(handle.simulations):
            if sim.sim_id == run_id:
                return (file_index, sim_index)
        return None

    def _fdiff_label(self, file_index: int, tile) -> str:
        """"File1 - RUN2 - BETA", as the specification writes it."""
        handle = self._session.file(file_index)
        name = handle.name if handle is not None else "?"
        return "{0} - {1} - {2}".format(name, tile.fdiff_run,
                                        tile.fdiff_param)

    def _clear_violation_bands(self) -> None:
        plot = self.getPlotItem()
        for item in self._fdiff_items:
            plot.removeItem(item)
        self._fdiff_items = []

    def _draw_violation_bands(self, plot, tile) -> None:
        """Wash in the stretches of x where a threshold was crossed.

        One band per region, in the group's colour, and not movable: unlike an
        AREA band this is not an annotation the user placed but a measurement,
        and dragging it would make it say something that was never computed.
        """
        self._clear_violation_bands()
        colours = getattr(self._state, "diff_group_colors", {}) or {}
        for group, low, high in tile.fdiff_bands:
            colour = QColor(colours.get(group) or "#9aa3ad")
            colour.setAlphaF(VIOLATION_ALPHA)
            if high <= low:
                # A single sample. A region of zero width draws nothing, so it
                # is given the narrowest band that is visible at any zoom: a
                # hairline at the sample, which is where it happened.
                span = (low, low)
            else:
                span = (low, high)
            region = pg.LinearRegionItem(values=span,
                                         brush=pg.mkBrush(colour),
                                         movable=False)
            region.setZValue(VIOLATION_Z)
            region.setToolTip("{0}: {1:g} to {2:g}".format(group, low, high))
            plot.addItem(region, ignoreBounds=True)
            self._fdiff_items.append(region)

    def _rebuild_diff(self, plot, tile, previous_groups) -> None:
        """Draw one curve: a parameter's difference between two runs.

        The two runs are sampled at their own instants, and nothing says the
        two sets of instants agree -- different runs of the same study often
        differ in length, and runs from two files almost always do. So the
        difference is taken on the first run's x, with the second run read at
        those same x by linear interpolation. The first run is the reference
        in a second sense too: the curve covers the stretch of x the two have
        in common, because outside it there is nothing to subtract and
        extrapolating would draw a difference that was never measured.

        np.interp rather than anything of our own: it is one pass in C over a
        sorted array, which is what this is, and the whole operation is two
        channel reads and a subtraction.
        """
        a_run, b_run = tile.diff_runs()
        # Two forms of the same thing. The curve carries the plain one, which
        # is what the crosshair readout prints; the legend row carries the
        # broken one, which is HTML and would print its own markup anywhere
        # else.
        label = self._diff_label(tile, a_run, b_run, wrap=False)
        legend_row = self._diff_label(tile, a_run, b_run, wrap=True)

        x, y = self._diff_samples(tile, a_run, b_run)
        if x is None:
            # Nothing the two runs share, or a channel one of them does not
            # carry. The tile says so in its own right -- the title names the
            # difference -- and an empty plot is the honest drawing of it.
            self._drop_groups(previous_groups)
            if self._legend is not None:
                self._legend.setVisible(False)
            self._sync_report = (0, 0)
            return

        styles = self._state.styles
        # Keyed on the channel, so the colour of a difference of ALPHA is the
        # colour of ALPHA -- and so the STYLE tab can restyle it like any
        # other curve.
        key = curve_color_key(tile, tile.diff_param, "", "", "")
        color = styles.color_for(key)
        dash = styles.dash_for(key)
        symbol = styles.symbol_for(key)
        width = styles.width_for(key)
        marker = symbol or ("o" if tile.show_points else None)

        curve_key = "diff|{0}|{1}|{2}|{3}|{4}".format(
            a_run[0], a_run[1], b_run[0], b_run[1], tile.diff_param)
        entry = CurveEntry(x, y, label, key, curve_key)
        self._curves[curve_key] = entry

        plan = _GroupPlan(color, dash, width, marker, styles.symbol_size,
                          legend_row if tile.show_legend else None)
        plan.entries.append(entry)
        self._adopt_groups(plot, previous_groups, [plan])

        if tile.show_legend and self._legend is not None:
            self._place_legend(tile)

    def _diff_label(self, tile, a_run, b_run, wrap: bool = True) -> str:
        """The name of a difference curve, on one line or two.

        One line was the whole row: channel, both runs and, when the two come
        from different files, both file names -- which is the usual case and
        the longest one. On a tile a quarter of a page wide that row ran past
        the plot it was labelling. So the channel takes the first line and the
        two runs the second:

            ALPHA
            build-a/R3 - build-b/R7

        A legend row is HTML -- pyqtgraph renders it through a LabelItem --
        so the break is a <br> rather than a newline, which the row would
        collapse to a space. `wrap` is off for the crosshair readout and
        anywhere else the name is read as text, where that markup would be
        printed rather than obeyed.
        """
        with_file = a_run[0] != b_run[0]
        return "{0}{1}{2} - {3}".format(
            tile.diff_param,
            "<br>" if wrap else ": ",
            self._run_name(a_run, with_file=with_file),
            self._run_name(b_run, with_file=with_file))

    def _run_name(self, run, with_file: bool = True) -> str:
        """How a (file, run) pair is written for a human."""
        file_index, sim_index = run
        handle = self._session.file(file_index)
        if handle is None:
            return "?"
        try:
            name = handle.simulations[sim_index].sim_id
        except (AttributeError, IndexError):
            return "?"
        if with_file:
            return "{0}/{1}".format(handle.name, name)
        return name

    def _diff_samples(self, tile, a_run, b_run):
        """(x, difference) for the stretch the two runs share, or (None, None).

        The sync offsets are applied first when SYNC is on, because a diff of
        two runs aligned on an event is exactly the case SYNC exists for: the
        same manoeuvre flown twice, started at different clock times.
        """
        x_a = self._diff_channel(a_run, tile.x_param)
        x_b = self._diff_channel(b_run, tile.x_param)
        y_a = self._diff_channel(a_run, tile.diff_param)
        y_b = self._diff_channel(b_run, tile.diff_param)
        if x_a is None or x_b is None or y_a is None or y_b is None:
            return None, None

        # Trim each to the length its own pair agrees on. A channel shorter
        # than the x it is plotted against is a half-written record, and
        # subtracting past the end of one would read whatever follows it.
        n_a, n_b = min(x_a.size, y_a.size), min(x_b.size, y_b.size)
        if n_a < 2 or n_b < 2:
            return None, None
        x_a, y_a = x_a[:n_a], y_a[:n_a]
        x_b, y_b = x_b[:n_b], y_b[:n_b]

        offsets, found, missing = self._sync_plan(
            tile, {a_run[0]: [a_run[1]], b_run[0]: [b_run[1]]})
        self._sync_report = (found, missing)
        shift_a = offsets.get(a_run) or 0.0
        shift_b = offsets.get(b_run) or 0.0
        if shift_a:
            x_a = x_a + np.float32(shift_a)
        if shift_b:
            x_b = x_b + np.float32(shift_b)

        # np.interp needs its reference sorted, and wants no NaN in the x it
        # reads from. An x channel is monotonic in these files, but a bad
        # record is not worth a wrong curve.
        good = np.isfinite(x_b) & np.isfinite(y_b)
        if good.sum() < 2:
            return None, None
        x_b, y_b = x_b[good], y_b[good]
        if not np.all(np.diff(x_b) >= 0.0):
            order = np.argsort(x_b, kind="stable")
            x_b, y_b = x_b[order], y_b[order]

        low = max(float(np.nanmin(x_a)), float(x_b[0]))
        high = min(float(np.nanmax(x_a)), float(x_b[-1]))
        if not (high > low):
            return None, None            # the two never overlap in x

        inside = np.isfinite(x_a) & (x_a >= low) & (x_a <= high)
        if inside.sum() < 2:
            return None, None
        x = np.ascontiguousarray(x_a[inside], dtype=np.float32)
        difference = (y_a[inside].astype(np.float32, copy=False)
                      - np.interp(x, x_b, y_b).astype(np.float32, copy=False))
        return x, np.ascontiguousarray(difference, dtype=np.float32)

    def _diff_channel(self, run, name: str):
        """One channel of one run, or None when it is not there to read."""
        if not name:
            return None
        try:
            data = self._session.channel(run[0], run[1], name)
        except Exception:                      # pragma: no cover - a bad read
            return None
        if data is None or getattr(data, "size", 0) == 0:
            return None
        return data

    def _frame_after_rebuild(self, tile) -> None:
        """Put the view where it belongs, and redraw what needs redrawing.

        The view is the user's. A rebuild happens for all sorts of reasons
        that say nothing about what should be on screen -- restyling a curve,
        toggling the grid, dragging a marker, coming back to a page -- and
        re-fitting on every one of them threw away whatever they had zoomed
        in on. So fit only when the tile has never been framed, which is the
        moment it first acquires a parameter; after that the frame moves when
        the user asks (Fit, double-click) or changes the SCALE limits.
        """
        if tile.view_initialised and self._curves:
            self._restore_view(tile)
        else:
            self._autoscale_if_unlocked()
            if self._curves:
                tile.view_initialised = True
                # The fit was applied with range events suppressed, so mirror
                # it into the tile state by hand -- otherwise the SCALE boxes
                # and the next _restore_view would both be working from the
                # range the tile had before it was fitted.
                self._store_view_range()

        # No force. A group that was newly built has no `sampled_for` and is
        # drawn; a group that was adopted whole keeps the key its points were
        # computed for and is left alone. Forcing here is what made every
        # rebuild re-decimate curves it had just proven identical.
        self._resample_visible()

    # -------------------------------------------------------- group reuse

    def _drop_groups(self, groups: List["DrawGroup"]) -> None:
        """Take a set of draw items off the plot for good."""
        plot = self.getPlotItem()
        for group in groups:
            plot.removeItem(group.item)

    @staticmethod
    def _reuse_signature(entries: List[CurveEntry]) -> tuple:
        """What a group draws, in a form two rebuilds can compare.

        The curve keys say *which* runs and parameters, and the buffer address
        and length say the samples are still the same ones -- a reloaded file,
        or a column evicted by the memory cap and read back, lands somewhere
        else and so compares unequal. Comparing addresses is only sound because
        the old entries are still alive and still referencing those buffers
        while this runs, so nothing else can have been put at the same address.
        """
        return tuple(
            (entry.curve_key, entry.x.ctypes.data, entry.x.size,
             entry.y.ctypes.data, entry.y.size)
            for entry in entries
        )

    def _adopt_groups(self, plot, previous: List["DrawGroup"],
                      plans: List["_GroupPlan"]) -> None:
        """Turn the planned groups into drawn ones, reusing what still fits.

        A plan whose curves match an existing group's exactly takes that
        group's item over: the pen and legend name are updated in place and the
        `sampled_for` key comes with it, so the points already on screen are
        recognised as current and never recomputed. Everything not taken over is
        removed from the plot.
        """
        available: Dict[tuple, List[DrawGroup]] = {}
        for group in previous:
            if not group.entries:
                # The "... and N more" legend spacer draws nothing and carries
                # no curves; it is rebuilt from scratch each time.
                continue
            available.setdefault(self._reuse_signature(group.entries),
                                 []).append(group)

        adopted: Set[int] = set()
        for plan in plans:
            candidates = available.get(self._reuse_signature(plan.entries))
            group = candidates.pop() if candidates else None
            if group is None:
                group = DrawGroup(plan.build_item())
                plot.addItem(group.item)
            else:
                adopted.add(id(group))
                plan.restyle_item(group.item)
                # The legend was cleared above and pyqtgraph only fills it from
                # addItem(), which an adopted item does not go through -- so its
                # row is put back by hand or it silently disappears.
                if plan.name is not None and self._legend is not None:
                    self._legend.addItem(group.item, plan.name)
            group.entries = plan.entries
            self._groups.append(group)

        self._drop_groups([g for g in previous if id(g) not in adopted])

    def _restore_view(self, tile: TileState) -> None:
        """Put the stored range back, without consulting the data.

        _store_view_range() mirrors every pan and zoom into the tile state, and
        a locked axis keeps the limit the user typed there instead, so the
        stored numbers are the right answer in both cases. Auto-ranging has to
        be turned off explicitly: adding items to a pyqtgraph view box otherwise
        makes it re-fit them, which is the same lost zoom by another route.
        """
        view = self.getPlotItem().getViewBox()
        log = bool(tile.log_y and not tile.plot3d)

        self._suppress_range_event = True
        try:
            view.disableAutoRange()
            if tile.x_max > tile.x_min:
                view.setXRange(tile.x_min, tile.x_max, padding=0)
            low = scales.to_view(tile.y_min, log)
            high = scales.to_view(tile.y_max, log)
            if low is not None and high is not None and high > low:
                view.setYRange(low, high, padding=0)
            self._apply_axis_limits(tile)
        finally:
            self._suppress_range_event = False

    # ------------------------------------------------------------------ 3D

    # The arithmetic lives in visimu.projection: it needs no Qt, so it can
    # be tested without a QApplication, which is where the interesting failures
    # would be.
    project_3d = staticmethod(projection.project)
    _normalise = staticmethod(projection.normalise)

    def _rebuild_3d(self, plot, tile: TileState) -> None:
        """Draw the tile as a rotatable 3D projection."""
        if self._legend is not None:
            self._legend.setVisible(False)

        # The overlay list is shared with the 3D box and its tick values, and
        # this is the one place that clears it on the way into 3D -- markers
        # and bands left by a two-dimensional pass have no meaning in a
        # projection and must not survive the switch. Claiming the bookkeeping
        # here is what stops _draw_overlays(), which runs after this, from
        # taking the box away again.
        for overlay in self._overlays:
            plot.removeItem(overlay)
        self._overlays = []
        self._overlays_drawn_for = ("3d",)

        if tile.is_empty() or not self._state.plotted_sims():
            self._gizmo.setVisible(False)
            return

        y_param = tile.y_params[0]
        series = []
        x_bounds = y_bounds = z_bounds = None

        by_file: Dict[int, List[int]] = {}
        for file_index, sim_index in self._state.plotted_sims():
            by_file.setdefault(file_index, []).append(sim_index)
        sync_offsets, _found, _missing = self._sync_plan(tile, by_file)

        for file_index, sim_index in self._state.plotted_sims():
            handle = self._session.file(file_index)
            if handle is None or sim_index >= handle.simulation_count:
                continue
            x = self._session.channel(file_index, sim_index, tile.x_param)
            offset = sync_offsets.get((file_index, sim_index))
            if offset:
                x = self._shifted_x(file_index, sim_index, x, offset)
            y = self._session.channel(file_index, sim_index, y_param)
            z = self._session.channel(file_index, sim_index, tile.z_param)
            count = min(x.size, y.size, z.size)
            if count == 0:
                continue

            # One projection is cheap per point but there can be millions, and
            # a projected path cannot be decimated afterwards without breaking
            # the curve, so thin it here against the tile's pixel budget.
            step = projection.thinning_step(count, max(256, self._target_points()))
            x, y, z = x[:count:step], y[:count:step], z[:count:step]

            for values, name in ((x, "x"), (y, "y"), (z, "z")):
                bounds = Session.bounds(values)
                if bounds is None:
                    continue
                if name == "x":
                    x_bounds = bounds if x_bounds is None else (
                        min(x_bounds[0], bounds[0]), max(x_bounds[1], bounds[1]))
                elif name == "y":
                    y_bounds = bounds if y_bounds is None else (
                        min(y_bounds[0], bounds[0]), max(y_bounds[1], bounds[1]))
                else:
                    z_bounds = bounds if z_bounds is None else (
                        min(z_bounds[0], bounds[0]), max(z_bounds[1], bounds[1]))

            sim = handle.simulations[sim_index]
            series.append((file_index, sim_index, handle.name, sim, x, y, z))

        if not series or x_bounds is None or y_bounds is None or z_bounds is None:
            self._gizmo.setVisible(False)
            return

        self._update_gizmo(tile, x_bounds, y_bounds, z_bounds)
        self._draw_3d_grid(plot, tile, x_bounds, y_bounds, z_bounds)

        styles = self._state.styles
        # The ranks are the 2D legend's ordering; a projection has no legend.
        iso_labels, _iso_ranks = (
            self._iso_labels(tile.color_iso_param)
            if tile.color_mode == ColorMode.ISO
            else ({}, {})
        )
        style_iso_labels = (
            self._iso_labels(tile.style_iso_param)[0]
            if tile.style_mode == StyleMode.ISO
            else {}
        )

        # Grouped by pen exactly as the 2D path is, and for the same reason: a
        # thousand trajectories is a thousand items otherwise. A 3D tile is
        # rebuilt on every step of a rotation, so the item count here costs on
        # every frame of a drag rather than once.
        groups: Dict[tuple, List[np.ndarray]] = {}
        pens: Dict[tuple, QPen] = {}
        gap = np.array([np.nan], dtype=np.float32)
        line_order = {}

        for file_index, sim_index, file_name, sim, x, y, z in series:
            iso_label = iso_labels.get((file_index, sim_index), "")
            style_iso_label = style_iso_labels.get((file_index, sim_index), "")
            handle = self._session.file(file_index)
            file_id = (handle.path or file_name) if handle is not None \
                else file_name
            color_key = curve_color_key(tile, tile.z_param, sim.key,
                                        iso_label, file_id)
            line_key = curve_line_key(
                tile, tile.z_param, sim.key, file_id, iso_label, style_iso_label)
            if tile.style_mode != StyleMode.NONE:
                if line_key not in line_order:
                    line_order[line_key] = len(line_order)
                styles.assign_line_style(line_key, line_order[line_key])
            color = styles.color_for(color_key)
            dash = styles.dash_for(line_key)
            width = styles.width_for(color_key)
            style_key = color_key

            key = "{0}|{1}|{2}".format(file_index, sim_index, tile.z_param)
            member = _ProjectedCurve(
                key,
                self._normalise(x, x_bounds),
                self._normalise(y, y_bounds),
                self._normalise(z, z_bounds),
                np.asarray(x, dtype=np.float32),
                "{0} · {1}".format(sim.sim_id, file_name),
                style_key,
            )

            group_key = (color, dash, width)
            if group_key not in groups:
                groups[group_key] = []
                pen = QPen(QColor(color))
                pen.setWidthF(width)
                pen.setStyle(_PEN_STYLES.get(dash, Qt.PenStyle.SolidLine))
                pen.setCosmetic(True)
                pens[group_key] = pen
            groups[group_key].append(member)

        # Everything above depends on the selection, the parameters and the
        # styling -- none of which a rotation changes. It is kept so that
        # turning the cube only has to re-project: see _reproject_3d().
        plan = []
        for group_key, members in groups.items():
            item = pg.PlotDataItem(
                pen=pens[group_key],
                # antialias, unlike the 2D path. There the curves arrive
                # decimated to pixel resolution so smoothing buys nothing for
                # real cost; here a projected trajectory is a sparse wireframe
                # of long diagonals, which is the case antialiasing is for.
                antialias=True, connect="finite",
            )
            plot.addItem(item)
            group = DrawGroup(item)
            self._groups.append(group)
            plan.append((group, members))

        self._projection = _Projection(
            (x_bounds, y_bounds, z_bounds), plan)
        self._project_into_items(tile)
        self._apply_3d_view(tile)

    def _project_into_items(self, tile: TileState) -> None:
        """Turn the cached cube into screen coordinates at the current angle.

        The only part of a 3D tile that a rotation actually changes, and the
        reason the cache above exists.
        """
        cache = self._projection
        if cache is None:
            return
        x_bounds, y_bounds, z_bounds = cache.bounds
        gap = np.array([np.nan], dtype=np.float32)

        for group, members in cache.groups:
            pieces_x, pieces_y = [], []
            for member in members:
                sx, sy, _depth = self.project_3d(
                    member.nx, member.ny, member.nz,
                    tile.azimuth, tile.elevation)
                sx = np.asarray(sx, dtype=np.float32)
                sy = np.asarray(sy, dtype=np.float32)

                # The stored x/y are the *projected* coordinates: that is what
                # the cursor and the click have to hit-test against, because it
                # is what is drawn. `source_x` carries the x parameter's own
                # values alongside, so a click can be reported back in the
                # units the rest of the application speaks.
                entry = CurveEntry(sx, sy, member.label, member.style_key,
                                   member.key)
                entry.source_x = member.source_x
                self._curves[member.key] = entry

                if pieces_x:
                    pieces_x.append(gap)
                    pieces_y.append(gap)
                pieces_x.append(sx)
                pieces_y.append(sy)

            if pieces_x:
                group.item.setData(np.concatenate(pieces_x),
                                   np.concatenate(pieces_y),
                                   connect="finite")
        del x_bounds, y_bounds, z_bounds

    def _reproject_3d(self) -> bool:
        """Redraw a turned cube without rebuilding it. True if it was done.

        rotate_3d() used to call rebuild(), on every mouse-move of a drag: that
        re-read three channels per run from the core, recomputed their bounds,
        re-thinned them, destroyed and recreated every draw item, and rebuilt
        the grid -- to change two angles. Everything but the projection and the
        grid is the same at every angle, so only those are redone here.
        """
        tile = self.tile_state
        if self._projection is None or not tile.plot3d:
            return False

        plot = self.getPlotItem()
        # The box and its tick labels turn with the cube, so they are the one
        # other thing that has to be redrawn.
        for overlay in self._overlays:
            plot.removeItem(overlay)
        self._overlays = []
        self._update_gizmo(tile, *self._projection.bounds)
        self._draw_3d_grid(plot, tile, *self._projection.bounds)

        self._project_into_items(tile)
        self._refresh_texts()
        return True

    def _draw_3d_grid(self, plot, tile: TileState, x_bounds, y_bounds, z_bounds) -> None:
        """Grid lines and tick values on the back walls of the data box.

        Without a reference surface an orthographic projection cannot be read:
        there is no perspective cue, so a point in the middle of the frame
        could be anywhere along the line of sight, and a trajectory that loops
        back on itself is guesswork. Drawing the three faces that lie *behind*
        the data -- and swapping which those are as the view turns -- is how
        every 3D plotting tool solves it, and it never covers the trajectory.

        The tick values matter as much as the lines: until this, a 3D tile had
        no numeric scale anywhere on it. You could see the shape of a
        trajectory and not read a single number off it.

        The geometry is in visimu.projection, which has no Qt in it, so the
        wall selection can be asserted at every rotation rather than eyeballed
        at one.
        """
        if not tile.show_3d_grid:
            return

        bounds = {"x": x_bounds, "y": y_bounds, "z": z_bounds}
        ticks = {}
        fractions = {}
        for axis, extent in bounds.items():
            values = projection.nice_ticks(extent[0], extent[1], 4)
            ticks[axis] = values
            fractions[axis] = [projection.to_fraction(v, extent) for v in values]

        segments = projection.wall_grid_segments(
            tile.azimuth, tile.elevation,
            fractions["x"], fractions["y"], fractions["z"])
        if not segments:
            return

        # One item for the whole grid: a NaN between segments breaks the
        # polyline, so several hundred lines cost one draw call rather than one
        # item each.
        points_x = []
        points_y = []
        points_z = []
        for start, end in segments:
            points_x.extend((start[0], end[0], float("nan")))
            points_y.extend((start[1], end[1], float("nan")))
            points_z.extend((start[2], end[2], float("nan")))

        screen_x, screen_y, _depth = self.project_3d(
            np.array(points_x), np.array(points_y), np.array(points_z),
            tile.azimuth, tile.elevation)

        colour = QColor(self._theme.plot_foreground)
        colour.setAlphaF(0.22)
        pen = pg.mkPen(colour, width=1.0)
        pen.setCosmetic(True)

        item = pg.PlotDataItem(screen_x, screen_y, pen=pen, antialias=False,
                               connect="finite")
        item.setZValue(GRID_3D_Z)
        plot.addItem(item, ignoreBounds=True)
        self._overlays.append(item)

        # Tick values, on the three box edges radiating from the far corner.
        # Coloured by axis, matching the orientation triad, so three rows of
        # numbers cannot be mistaken for one another.
        anchors = projection.tick_anchors(
            tile.azimuth, tile.elevation,
            fractions["x"], fractions["y"], fractions["z"])

        axis_colours = {name.lower(): colour_hex
                        for name, _vector, colour_hex in projection.AXIS_VECTORS}

        for axis in ("x", "y", "z"):
            values = ticks[axis]
            if len(values) < 2:
                continue
            step = values[1] - values[0]
            label_colour = QColor(axis_colours[axis])
            label_colour.setAlphaF(0.75)

            positions = anchors[axis]
            label_x, label_y, _d = self.project_3d(
                np.array([p[0] for p in positions]),
                np.array([p[1] for p in positions]),
                np.array([p[2] for p in positions]),
                tile.azimuth, tile.elevation)

            for index, value in enumerate(values):
                text = pg.TextItem(projection.format_tick(value, step),
                                   color=label_colour, anchor=(0.5, 0.5),
                                   fill=_label_brush(
                                       self._theme.plot_background))
                text.setFont(plot_font(PLOT_LABEL_PT))
                text.setPos(float(label_x[index]), float(label_y[index]))
                text.setZValue(GRID_3D_Z + 1)
                plot.addItem(text, ignoreBounds=True)
                self._overlays.append(text)

    def _update_gizmo(self, tile: TileState, x_bounds, y_bounds, z_bounds) -> None:
        """Point the orientation triad the same way as the projection."""
        self._gizmo.setVisible(bool(tile.show_3d_axes))
        if not tile.show_3d_axes:
            return
        self._gizmo.set_orientation(tile.azimuth, tile.elevation)
        self._gizmo.describe((
            ("X", self._axis_label(tile.x_param), x_bounds),
            ("Y", self._axis_label(tile.y_params[0] if tile.y_params else ""),
             y_bounds),
            ("Z", self._axis_label(tile.z_param), z_bounds),
        ))

    def _apply_3d_view(self, tile: TileState) -> None:
        """Point the view box at the stored window onto the projection.

        Aspect-locked, because a projection stretched on one axis is not a
        projection of anything.
        """
        half = max(1e-3, float(tile.view3d_half))
        view = self.getPlotItem().getViewBox()
        self._suppress_range_event = True
        try:
            view.setAspectLocked(True)
            view.setRange(
                xRange=(tile.view3d_cx - half, tile.view3d_cx + half),
                yRange=(tile.view3d_cy - half, tile.view3d_cy + half),
                padding=0,
            )
        finally:
            self._suppress_range_event = False

    def _store_3d_view(self, tile: TileState) -> None:
        """Remember where the user zoomed or panned to.

        Without this, a wheel zoom would survive exactly until the next
        rotation -- and rotating is the main thing you do to a 3D plot.
        """
        try:
            (x_low, x_high), (y_low, y_high) = self.getPlotItem().vb.viewRange()
        except (TypeError, ValueError):
            return
        half = (x_high - x_low) * 0.5
        if half <= 0:
            return
        tile.view3d_cx = (x_low + x_high) * 0.5
        tile.view3d_cy = (y_low + y_high) * 0.5
        tile.view3d_half = half

    def zoom_3d(self, factor: float) -> None:
        """Scale the 3D view about its centre. factor > 1 zooms out."""
        tile = self.tile_state
        if not tile.plot3d:
            return
        tile.view3d_half = clamp_3d_half(tile.view3d_half * factor)
        self._state.dirty = True
        self._apply_3d_view(tile)

    def pan_3d(self, dx_pixels: float, dy_pixels: float) -> None:
        """Slide the 3D view by a drag measured in pixels.

        Converted through the current zoom so the object tracks the pointer at
        any scale -- a fixed step in projection units would crawl when zoomed
        out and fly when zoomed in.
        """
        tile = self.tile_state
        if not tile.plot3d:
            return
        view = self.getPlotItem().getViewBox()
        width = max(1.0, float(view.width()))
        height = max(1.0, float(view.height()))
        span = tile.view3d_half * 2.0
        tile.view3d_cx -= dx_pixels / width * span
        # Screen y grows downward, the view's upward.
        tile.view3d_cy += dy_pixels / height * span
        self._state.dirty = True
        self._apply_3d_view(tile)

    def rotate_3d(self, d_azimuth: float, d_elevation: float) -> None:
        tile = self.tile_state
        if not tile.plot3d:
            return
        tile.azimuth = (tile.azimuth + d_azimuth) % 360.0
        # Beyond straight up or straight down the projection turns inside out,
        # so the tilt is clamped rather than wrapped.
        tile.elevation = max(-89.0, min(89.0, tile.elevation + d_elevation))
        self._state.dirty = True
        if not self._reproject_3d():
            self.rebuild()

    # ------------------------------------------------------- shared x zoom

    def _announce_x_zoom(self, low: float, high: float) -> None:
        if high > low:
            self.x_zoom_shared.emit(float(low), float(high))

    def apply_shared_x(self, low: float, high: float) -> bool:
        """Take the x range of a zoom made in another tile. True if applied.

        Three tiles are left out of it, and each for a reason the user would
        give themselves: a 3D tile, whose frame is a box and not a pair of
        ranges; a tile whose x axis is locked, because a lock is an
        instruction; and a tile plotting against a different channel
        altogether, where the numbers of one axis mean nothing on the other.

        The range event is suppressed while it is applied. Twenty-five tiles
        each announcing a view change would have the SCALE boxes and the
        Inspect statistics recomputed twenty-five times over one gesture; the
        window announces it once when the round is done.
        """
        tile = self.tile_state
        if tile.plot3d or tile.lock_x:
            return False
        if not (high > low):
            return False

        self._suppress_range_event = True
        try:
            self.getPlotItem().setXRange(low, high, padding=0)
        finally:
            self._suppress_range_event = False

        tile.x_min, tile.x_max = float(low), float(high)
        tile.view_initialised = True
        self._resample_visible(force=True)
        return True

    # --------------------------------------------------------- resampling

    def _on_range_changed(self, *_args) -> None:
        if self._suppress_range_event:
            return
        self._store_view_range()
        if self._redraw_timer is not None:
            self._redraw_timer.start()
        else:
            self.view_changed.emit()

    def _on_redraw_timer(self) -> None:
        """The coalesced end of a pan or a zoom.

        pyqtgraph emits sigRangeChanged continuously while the mouse drags, and
        the resampling was already debounced onto this timer for that reason.
        `view_changed` was not, and it has two listeners that do real work: the
        SCALE boxes rewrite themselves from the range, and the Inspect tab
        recomputes the statistics of the visible window -- which reads every
        selected channel. Both were running per pixel of a drag to show figures
        nobody can read until the mouse stops.

        The stored range is still updated on every event (above), so nothing
        that reads tile.x_min while dragging sees a stale number; only the
        announcement waits.
        """
        self._resample_visible()
        self.view_changed.emit()

    def _store_view_range(self) -> None:
        """Mirror the visible range into the tile state while it is unlocked.

        This is what lets the SCALE panel's boxes follow the plot as you pan and
        zoom, so ticking Lock freezes exactly what you are looking at. A locked
        axis is left alone -- there the stored values are the input, not the
        output.
        """
        tile = self.tile_state
        if tile.plot3d:
            self._store_3d_view(tile)
            return
        try:
            (x_low, x_high), (y_low, y_high) = self.getPlotItem().vb.viewRange()
        except (TypeError, ValueError):
            return
        if not tile.lock_x:
            tile.x_min, tile.x_max = float(x_low), float(x_high)
        if not tile.lock_y:
            log = bool(tile.log_y and not tile.plot3d)
            tile.y_min = self._from_view_y(y_low, log)
            tile.y_max = self._from_view_y(y_high, log)

    # Log-axis conversion lives in visimu.scales, which imports no Qt so the
    # arithmetic can be tested without a running application. See that module
    # for why the tile's y range is kept in data units throughout.
    _to_view_y = staticmethod(scales.to_view)
    _from_view_y = staticmethod(scales.from_view)

    def _target_points(self) -> int:
        width = max(1, self.getPlotItem().vb.width())
        return max(64, int(width * POINTS_PER_PIXEL))

    def _resample_visible(self, force: bool = False) -> None:
        """Re-decimate every curve for the visible x range.

        Skips curves whose drawn points already answer the current question.
        Redraws that do not move the view are common -- a style change, an
        overlay update, a dragged marker, a resize that leaves the width alone
        -- and each was re-decimating every visible curve from full resolution
        for a result identical to the one already on screen. Pass force=True
        after replacing the underlying data.
        """
        if not self._curves:
            return

        tile = self.tile_state
        if tile.plot3d:
            # The stored arrays are already projected screen coordinates, and a
            # path through 3D space is not monotonic in either of them --
            # decimating by x range would cut the trajectory into pieces. The
            # thinning for a 3D tile happens once, in _rebuild_3d.
            return

        view_range = self.getPlotItem().vb.viewRange()
        x_low, x_high = float(view_range[0][0]), float(view_range[0][1])
        curve_count = sum(len(g.entries) for g in self._groups)
        total_points = sum(min(e.x.size, e.y.size)
                           for g in self._groups for e in g.entries)
        target = resample_target(self._target_points(), curve_count,
                                 total_points)

        # A zero-width or inverted range means "everything"; the core treats
        # x_min >= x_max as the whole-series case.
        x_range: Optional[Tuple[float, float]] = None
        if x_high > x_low:
            span = x_high - x_low
            # A small margin keeps the curve drawn to the tile edges while
            # panning, instead of visibly redrawing at the border.
            x_range = (x_low - span * 0.05, x_high + span * 0.05)

        # Built from the final target, after the budget above has had its say,
        # so the key can never describe a resampling that did not happen. The
        # quantisation, and why the span has to be part of it, are in
        # visimu.scales -- which imports no Qt, so that arithmetic can be
        # tested without a running application.
        key = resample_key(x_range, target, tile.log_y)

        gap = np.array([np.nan], dtype=np.float32)

        for group in self._groups:
            if not force and group.sampled_for == key:
                continue

            pieces_x: List[np.ndarray] = []
            pieces_y: List[np.ndarray] = []
            for entry in group.entries:
                if min(entry.x.size, entry.y.size) == 0:
                    continue
                dx, dy = self._session.decimate(entry.x, entry.y, target, x_range)
                if dx.size == 0:
                    continue
                if pieces_x:
                    # One NaN between runs. connect="finite" breaks the polyline
                    # there, so consecutive runs are never joined by a line
                    # across the plot.
                    pieces_x.append(gap)
                    pieces_y.append(gap)
                pieces_x.append(dx)
                pieces_y.append(dy)

            if not pieces_x:
                group.item.setData([], [])
            else:
                cx = pieces_x[0] if len(pieces_x) == 1 else np.concatenate(pieces_x)
                cy = pieces_y[0] if len(pieces_y) == 1 else np.concatenate(pieces_y)
                if tile.log_y:
                    # pyqtgraph's log mode cannot show non-positive values;
                    # blanking them keeps the rest of the curve visible instead
                    # of dropping it.
                    cy = np.where(cy > 0, cy, np.nan)
                group.item.setData(cx, cy, connect="finite")
            group.sampled_for = key

    def _apply_axis_limits(self, tile: TileState) -> None:
        """Stop a locked axis being panned or zoomed out of its range.

        Setting the range once is not enough: it says where to start, not where
        you may go, so a drag or a wheel took the view straight back outside
        the limit the user had just imposed. setLimits is the hard bound, and
        the mouse is disabled per axis on top of it so a drag on a locked axis
        does not fight the clamp.
        """
        view = self.getPlotItem().getViewBox()

        # A degenerate range would set maxXRange=0 and freeze the view at zero
        # width, which looks exactly like a crash.
        if tile.lock_x and tile.x_max > tile.x_min:
            view.setLimits(xMin=tile.x_min, xMax=tile.x_max,
                           minXRange=None, maxXRange=tile.x_max - tile.x_min)
        else:
            view.setLimits(xMin=None, xMax=None, maxXRange=None)

        log = bool(tile.log_y and not tile.plot3d)
        low = self._to_view_y(tile.y_min, log)
        high = self._to_view_y(tile.y_max, log)
        if tile.lock_y and low is not None and high is not None and high > low:
            view.setLimits(yMin=low, yMax=high, minYRange=None, maxYRange=high - low)
        else:
            view.setLimits(yMin=None, yMax=None, maxYRange=None)

        mode = self._state.mouse_mode
        interactive = mode in (MouseMode.PAN, MouseMode.ZOOM)
        view.setMouseEnabled(x=interactive and not tile.lock_x,
                             y=interactive and not tile.lock_y)

    @staticmethod
    def _y_extent(entry: CurveEntry, window):
        """The extent of a curve's y over an x window, or over all of it.

        With no window this is the summary the core computed when the column
        was parsed, which costs nothing. With one it is a single pass in C++
        that understands a non-monotonic x, so a phase plane is windowed by
        value rather than by a binary search that would see only half the
        trajectory.
        """
        if window is None:
            return Session.bounds(entry.y)
        stats = Session.stats(entry.y, entry.x, window)
        if stats is None:
            return Session.bounds(entry.y)
        return (stats["min"], stats["max"])

    def _autoscale_if_unlocked(self) -> None:
        tile = self.tile_state
        plot = self.getPlotItem()

        if tile.plot3d:
            # A projection has its own fixed, aspect-locked frame.
            return

        log = bool(tile.log_y)

        self._suppress_range_event = True
        try:
            # Before the ranges, not only after them. Every setXRange and
            # setYRange below is clamped by whatever bounds the view box is
            # carrying, so a tile just released from a lock has to be released
            # on the view box first or the fit is computed correctly and then
            # thrown away. It is applied again at the end, where it is what
            # pins a locked axis to the range just set for it.
            self._apply_axis_limits(tile)

            if tile.lock_x:
                plot.setXRange(tile.x_min, tile.x_max, padding=0)
            if tile.lock_y:
                low = self._to_view_y(tile.y_min, log)
                high = self._to_view_y(tile.y_max, log)
                if low is not None and high is not None and high > low:
                    plot.setYRange(low, high, padding=0)

            if not tile.lock_x or not tile.lock_y:
                # Fitting y means fitting what is in view, not what is in the
                # file. With x pinned to a window -- locked in SCALE, or shared
                # across a row of tiles for comparison -- the y extent of the
                # whole channel is the wrong answer, and on a manoeuvre that
                # spans a wide range it leaves the part being examined as a flat
                # line at the bottom of the tile.
                window = None
                if tile.lock_x and tile.x_max > tile.x_min:
                    window = (tile.x_min, tile.x_max)

                x_bounds = None
                y_bounds = None
                for entry in self._curves.values():
                    xb = Session.bounds(entry.x)
                    yb = self._y_extent(entry, window)
                    if xb:
                        x_bounds = xb if x_bounds is None else (
                            min(x_bounds[0], xb[0]), max(x_bounds[1], xb[1]))
                    if yb:
                        y_bounds = yb if y_bounds is None else (
                            min(y_bounds[0], yb[0]), max(y_bounds[1], yb[1]))

                if not tile.lock_x and x_bounds:
                    plot.setXRange(x_bounds[0], x_bounds[1], padding=0.02)
                if not tile.lock_y and y_bounds:
                    # The frame is set in view units. On a log axis that means
                    # log10 of the data, and a channel that crosses zero is
                    # framed on the part that can be drawn at all.
                    view_y = scales.range_to_view(y_bounds[0], y_bounds[1], log)
                    low, high = view_y if view_y is not None else (None, None)
                    if low is not None and high is not None:
                        if high <= low:
                            # A flat channel would otherwise collapse to a
                            # zero-height view and render as an invisible line
                            # on the axis.
                            pad = abs(high) * 0.05 or 0.5
                            low, high = low - pad, high + pad
                        plot.setYRange(low, high, padding=0.05)

            self._apply_axis_limits(tile)
        finally:
            self._suppress_range_event = False

    # ----------------------------------------------------------- overlays

    def _on_text_moved(self, _note) -> None:
        """A caption was dragged. The note already carries its new position --
        the overlay edits it directly so the drag can be shown live -- so all
        that is left is to mark the workspace dirty and let the panel catch up.
        """
        self._state.dirty = True
        self.text_moved.emit()

    def refresh_texts(self) -> None:
        """Public form of the caption repaint, for the panel's live edits."""
        self._refresh_texts()

    def _refresh_texts(self) -> None:
        """Re-lay the captions. Cheap: it paints text, it does not touch data."""
        if self._texts is None:
            return
        self._texts.set_notes(self.tile_state.texts,
                              self.getPlotItem().getViewBox().boundingRect())

    def _overlay_signature(self, tile: TileState) -> tuple:
        """Everything the drawn markers and bands are a function of.

        The annotations themselves, the x channel that decides which of the
        two renderings each one gets, whether they can be dragged, and the
        curves -- because the dot form finds its sample *in* a curve, and the
        band form redraws a stretch of one. The curve buffers are compared by
        address for the same reason and with the same soundness argument as
        _reuse_signature(): the entries are alive and holding them while this
        runs.
        """
        return (
            tile.x_param,
            self._annotations_movable(),
            tuple((m.x_param, m.position, m.visible, m.color, m.dot_size)
                  for m in self._state.markers),
            tuple((a.x_param, a.x_min, a.x_max, a.visible, a.color, a.alpha)
                  for a in self._state.areas),
            tuple(sorted(
                (key, entry.x.ctypes.data, entry.x.size,
                 entry.y.ctypes.data, entry.y.size)
                for key, entry in self._curves.items())),
        )

    def _draw_overlays(self) -> None:
        """Markers and shaded bands.

        A marker is a condition on one parameter. On a tile plotted against that
        same parameter it is a vertical line. On any other tile it is drawn as a
        dot on each curve, at the sample where the condition holds -- so a
        marker at TIME = 5 shows you where five seconds falls on ALPHA = f(NZ).

        The dot form is what makes this worth guarding. Finding the sample
        where the condition holds is a search per curve, and the band form
        redraws a stretch of every curve it crosses, so a thousand-run tile
        paid both on every rebuild -- including the rebuilds the redraw cache
        exists to make free, where not one curve and not one annotation had
        changed. The items are left where they are when nothing they depend on
        has moved.
        """
        plot = self.getPlotItem()
        tile = self.tile_state

        if tile.plot3d:
            # Neither form makes sense in a projection: a vertical line is not
            # a plane, and the dot indices refer to the thinned projected
            # arrays rather than to samples. The 3D box carries the scale.
            #
            # Returning before the clear below, not after: _rebuild_3d() has
            # already put the data box and its tick values in _overlays, and
            # it runs first.
            return

        signature = self._overlay_signature(tile)
        if signature == self._overlays_drawn_for:
            return
        self._overlays_drawn_for = signature

        for overlay in self._overlays:
            plot.removeItem(overlay)
        self._overlays = []

        for marker in self._state.markers:
            if not marker.visible or not marker.x_param:
                continue

            if marker.x_param == tile.x_param:
                movable = self._annotations_movable()
                # No label. pyqtgraph builds an InfLineLabel from `label` and
                # forwards labelOpts straight to TextItem, which takes neither
                # a CSS "font-size" nor much else -- passing one raised on
                # every rebuild, and an exception inside _draw_overlays aborts
                # PlotGrid.refresh() partway through the page, which is why a
                # dragged band moved on the tile under the cursor and nowhere
                # else. Captions are text notes now; see model.Marker.
                line = pg.InfiniteLine(
                    pos=marker.position,
                    angle=90,
                    pen=pg.mkPen(marker.color, width=1.5, style=Qt.PenStyle.DashLine),
                    movable=movable,
                    hoverPen=pg.mkPen(marker.color, width=3.0),
                )
                # Connected whether or not it is movable *now*. The mode can
                # change under an item that is already on the plot --
                # apply_mouse_mode() calls setMovable() on the overlays it
                # finds rather than rebuilding them -- and an item made while
                # dragging was off then became draggable with nothing
                # listening: pyqtgraph moved the line, the model never heard,
                # and no other tile followed. A line that is not movable emits
                # nothing, so the connection costs nothing until it matters.
                line.sigPositionChangeFinished.connect(
                    lambda item, m=marker: self._marker_dragged(m, item)
                )
                if movable:
                    line.setCursor(Qt.CursorShape.SizeHorCursor)
                line.setZValue(MARKER_Z)
                plot.addItem(line, ignoreBounds=True)
                self._overlays.append(line)
            else:
                self._draw_marker_dots(plot, marker)

        for area in self._state.areas:
            if not area.visible or not area.x_param:
                continue
            if area.x_param == tile.x_param:
                color = QColor(area.color)
                color.setAlphaF(area.alpha)
                movable = self._annotations_movable()
                region = pg.LinearRegionItem(
                    values=(area.x_min, area.x_max),
                    brush=pg.mkBrush(color),
                    movable=movable,
                )
                # Connected unconditionally; see the marker above.
                region.sigRegionChangeFinished.connect(
                    lambda item, a=area: self._area_dragged(a, item)
                )
                if movable:
                    # Both edges and the band as a whole become draggable.
                    region.setHoverBrush(pg.mkBrush(self._hover_colour(area)))
                # In front of the curves, not behind them. A band is a
                # translucent wash marking a phase of the run, and the whole
                # point of it is to see which part of the trace it covers --
                # sat underneath, its edges disappeared wherever a curve
                # crossed them, and on a busy tile it was barely visible at
                # all. The alpha is what keeps the data readable through it.
                region.setZValue(AREA_Z)
                plot.addItem(region, ignoreBounds=True)
                self._overlays.append(region)
            else:
                self._draw_area_segments(plot, area)

    # --------------------------------------------------- editing overlays

    def _annotations_movable(self) -> bool:
        """Markers and area edges are draggable in Select mode only.

        Select is the mode where nothing else responds to a drag -- the view
        box has its mouse handling off entirely -- so a drag there is
        unambiguous. In Pan or Zoom the same gesture already means something,
        and in Inspect it draws a slope line.
        """
        return (self._state.mouse_mode == MouseMode.SELECT
                and not self.tile_state.plot3d)

    @staticmethod
    def _hover_colour(area) -> QColor:
        colour = QColor(area.color)
        colour.setAlphaF(min(0.85, area.alpha * 2.0 + 0.12))
        return colour

    def _marker_dragged(self, marker, item) -> None:
        position = float(item.value())
        if position == marker.position:
            return
        marker.position = position
        self._state.dirty = True
        # Redrawing this tile from inside its own item's signal would delete
        # the item mid-callback; the window refreshes every tile instead.
        self.annotation_moved.emit()

    def _area_dragged(self, area, item) -> None:
        low, high = (float(v) for v in item.getRegion())
        low, high = min(low, high), max(low, high)
        if (low, high) == (area.x_min, area.x_max):
            return
        area.x_min, area.x_max = low, high
        self._state.dirty = True
        self.annotation_moved.emit()

    def _draw_area_segments(self, plot, area) -> None:
        """An area from another x parameter, drawn as highlighted curve spans.

        A band on TIME between 10 s and 20 s is a vertical stripe on a time
        plot, but on ALPHA = f(NZ) there is no vertical stripe that means the
        same thing -- the interval maps to a *piece of each curve*. Previously
        such a band simply vanished when you looked at a tile with a different
        x axis, which read as the annotation being lost. Drawing the matching
        span of every curve keeps it present and, on a trajectory, says
        something the stripe could not: where the aircraft actually was during
        those ten seconds.
        """
        color = QColor(area.color)
        pen = pg.mkPen(color, width=5.0)
        pen.setCosmetic(True)

        reference_cache: Dict[Tuple[int, int], Optional[np.ndarray]] = {}

        for key, entry in self._curves.items():
            try:
                file_index, sim_index, _param = key.split("|", 2)
                file_index, sim_index = int(file_index), int(sim_index)
            except ValueError:
                continue

            cache_key = (file_index, sim_index)
            if cache_key not in reference_cache:
                reference = self._session.channel(file_index, sim_index, area.x_param)
                reference_cache[cache_key] = reference if reference.size else None

            reference = reference_cache[cache_key]
            if reference is None:
                continue

            count = min(reference.size, entry.x.size, entry.y.size)
            if count == 0:
                continue
            inside = ((reference[:count] >= area.x_min)
                      & (reference[:count] <= area.x_max))
            if not inside.any():
                continue

            # Break the mask into runs so two separate passes through the band
            # do not get joined by a line across the plot.
            x = np.where(inside, entry.x[:count], np.nan)
            y = np.where(inside, entry.y[:count], np.nan)
            item = pg.PlotCurveItem(x, y, pen=pen, connect="finite")
            item.setZValue(AREA_Z)
            item.setToolTip(area.describe())
            plot.addItem(item, ignoreBounds=True)
            self._overlays.append(item)

    def _draw_marker_dots(self, plot, marker) -> None:
        """One dot per curve, where the marker's condition is met.

        The reference channel is looked up once per run rather than per curve:
        every curve of a run shares it, and on large files that lookup is the
        expensive part.
        """
        spots = []
        reference_cache: Dict[Tuple[int, int], Optional[int]] = {}

        for key, entry in self._curves.items():
            try:
                file_index, sim_index, _param = key.split("|", 2)
                file_index = int(file_index)
                sim_index = int(sim_index)
            except ValueError:
                continue

            cache_key = (file_index, sim_index)
            if cache_key not in reference_cache:
                reference = self._session.channel(file_index, sim_index, marker.x_param)
                if reference.size == 0:
                    reference_cache[cache_key] = None
                else:
                    index = Session.nearest_index(reference, marker.position)
                    # Refuse to place a dot when the requested position is
                    # outside the run: a dot clamped to the last sample would
                    # silently claim the condition was met when it never was.
                    lo = float(reference.min())
                    hi = float(reference.max())
                    span = hi - lo
                    tolerance = max(abs(span) * 1e-6, 1e-9)
                    if marker.position < lo - tolerance or marker.position > hi + tolerance:
                        reference_cache[cache_key] = None
                    else:
                        reference_cache[cache_key] = int(index)

            index = reference_cache[cache_key]
            if index is None:
                continue
            if index >= entry.x.size or index >= entry.y.size:
                continue

            spots.append({"pos": (float(entry.x[index]), float(entry.y[index]))})

        if not spots:
            return

        scatter = pg.ScatterPlotItem(
            size=marker.dot_size,
            pen=pg.mkPen(marker.color, width=1.5),
            brush=pg.mkBrush(marker.color),
            symbol="o",
            hoverable=False,
        )
        scatter.addPoints(spots)
        scatter.setZValue(MARKER_Z)
        scatter.setToolTip(marker.describe())
        plot.addItem(scatter, ignoreBounds=True)
        self._overlays.append(scatter)

    # ------------------------------------------------------------- cursor

    def _on_mouse_moved(self, position) -> None:
        if not self._curves:
            return
        vb = self.getPlotItem().vb
        if not self.sceneBoundingRect().contains(position):
            return

        point = vb.mapSceneToView(position)
        x_value = float(point.x())

        readout = ["{0} = {1:.6g}".format(self.tile_state.x_param or "x", x_value)]
        for entry in list(self._curves.values())[:6]:
            if entry.x.size == 0:
                continue
            index = Session.nearest_index(entry.x, x_value)
            if index < entry.y.size:
                readout.append("{0} = {1:.6g}".format(entry.label, float(entry.y[index])))
        self.cursor_moved.emit("   |   ".join(readout))

    def mousePressEvent(self, event) -> None:
        if self._index >= 0:
            self.activated.emit(self._index)

        # Traced before anything is done to the event: what the scene thinks
        # is in progress *as the press arrives* is the question, and this
        # handler is about to change it.
        state = dict(zoomtrace.view_state(self.getPlotItem().vb))
        state.update(zoomtrace.scene_state(self))
        zoomtrace.log("press", tile=self._index,
                      mode=self._state.mouse_mode.name,
                      button=zoomtrace.button_name(event),
                      modifiers=zoomtrace.modifier_names(event),
                      at=zoomtrace.position(event),
                      armed=self._double_click,
                      plot3d=self.tile_state.plot3d,
                      **state)
        self._drag_moves = 0

        # A fresh press means any earlier gesture is over, whatever became of
        # its release. Belt and braces: nothing downstream should ever find a
        # double click still armed from a gesture that has been and gone.
        self._double_click = None
        self._double_click_from = None

        tile = self.tile_state

        # 3D navigation follows the convention every simulation viewer shares:
        # left drag turns the object, middle drag slides it, right drag zooms,
        # and the wheel zooms too. Rotate alone was not enough -- once you have
        # zoomed in there is no way to reach anything but the centre.
        if tile.plot3d and self._state.mouse_mode == MouseMode.PAN:
            button = event.button()
            if button == Qt.MouseButton.LeftButton:
                modifiers = event.modifiers()
                if modifiers & Qt.KeyboardModifier.ShiftModifier:
                    self._nav_action = "pan"
                elif modifiers & Qt.KeyboardModifier.ControlModifier:
                    self._nav_action = "zoom"
                else:
                    self._nav_action = "rotate"
            elif button == Qt.MouseButton.MiddleButton:
                self._nav_action = "pan"
            elif button == Qt.MouseButton.RightButton:
                self._nav_action = "zoom"
            else:
                self._nav_action = None

            if self._nav_action is not None:
                self._rotating_from = event.position()
                zoomtrace.log("press.taken", by="3d nav",
                              action=self._nav_action)
                event.accept()
                return

        if (self._state.mouse_mode == MouseMode.CROSSHAIR
                and event.button() == Qt.MouseButton.LeftButton):
            # The press starts a potential slope drag; whether it was a click
            # or a drag is only known on release.
            self._slope_origin = self._data_point(event.position())
            self._slope_item.setVisible(False)
            self._slope_text.setVisible(False)
            zoomtrace.log("press.taken", by="slope origin")
            event.accept()
            return

        zoomtrace.log("press.passed", to="pyqtgraph")
        super().mousePressEvent(event)
        # After, as well as before: the scene registers a press by appending
        # to clickEvents, and that is what it later turns into a drag. A press
        # that leaves clickEvents unchanged never reached it.
        zoomtrace.log("press.after", **zoomtrace.scene_state(self))

    def mouseMoveEvent(self, event) -> None:
        # Only while a button is down: a hover is a move too, and tracing
        # those would bury the gesture in the log. Throttled, and the total is
        # reported on the release.
        try:
            dragging = int(event.buttons()) != 0
        except (AttributeError, TypeError, ValueError):
            dragging = False
        if dragging:
            self._drag_moves = int(self._drag_moves or 0) + 1
            if zoomtrace.should_log_move(self._drag_moves):
                zoomtrace.log("move", tile=self._index,
                              n=self._drag_moves,
                              buttons=zoomtrace.buttons_held(event),
                              at=zoomtrace.position(event),
                              **zoomtrace.scene_state(self))

        if self._rotating_from is not None:
            delta = event.position() - self._rotating_from
            self._rotating_from = event.position()

            if self._nav_action == "pan":
                self.pan_3d(delta.x(), delta.y())
            elif self._nav_action == "zoom":
                # Vertical drag, as in every 3D viewer: down zooms out.
                self.zoom_3d(1.006 ** delta.y())
            else:
                # Roughly a quarter turn across the width of a typical tile.
                self.rotate_3d(-delta.x() * 0.4, delta.y() * 0.4)
            event.accept()
            return

        if self._slope_origin is not None:
            current = self._data_point(event.position())
            if current is not None:
                self._update_slope_line(self._slope_origin, current)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        state = dict(zoomtrace.view_state(self.getPlotItem().vb))
        state.update(zoomtrace.scene_state(self))
        zoomtrace.log("release", tile=self._index,
                      mode=self._state.mouse_mode.name,
                      button=zoomtrace.button_name(event),
                      at=zoomtrace.position(event),
                      armed=self._double_click,
                      moves=self._drag_moves,
                      **state)

        if self._double_click is not None:
            # Resolved, and then handed on rather than consumed. pyqtgraph
            # drives its rubber-band zoom from the scene's press, move and
            # release events, so a release that stops here leaves its scene
            # believing a drag is still in progress -- and from then on the
            # band never appears again, on this tile or the next one clicked.
            #
            # That is what "the zoom stops working after a few zooms" was: a
            # double click followed by a drag is exactly what consecutive
            # zooming produces, and this branch ate the release that would
            # have ended it.
            moved = self._resolve_double_click(event)
            zoomtrace.log("release.doubleclick", resolved=True, moved=moved)

        if self._rotating_from is not None:
            self._rotating_from = None
            self._nav_action = None
            zoomtrace.log("release.taken", by="3d nav")
            event.accept()
            return

        if self._slope_origin is not None:
            origin = self._slope_origin
            self._slope_origin = None
            end = self._data_point(event.position())

            if end is not None and self._is_real_drag(origin, end):
                self._update_slope_line(origin, end)
                self.slope_measured.emit(origin[0], origin[1], end[0], end[1])
            else:
                # A click, not a drag: the old behaviour, reading the sample.
                self._slope_item.setVisible(False)
                self._slope_text.setVisible(False)
                self._inspect_at(event.position())
            zoomtrace.log("release.taken", by="slope/inspect")
            event.accept()
            return

        zoomtrace.log("release.passed", to="pyqtgraph",
                      **zoomtrace.scene_state(self))
        super().mouseReleaseEvent(event)
        # The decisive line. A release that reached the scene clears the
        # button it was holding; one that did not leaves dragButtons as it
        # was, and from then on every press is read as the continuation of a
        # drag that has already ended -- which is what a band that never
        # appears again looks like from the outside.
        zoomtrace.log("release.after", **zoomtrace.scene_state(self))

    # -------------------------------------------------------------- slope

    def _data_point(self, position) -> Optional[Tuple[float, float]]:
        """A widget position as data coordinates, or None if off the plot."""
        view = self.getPlotItem().vb
        scene_position = self.mapToScene(position.toPoint())
        if not self.sceneBoundingRect().contains(scene_position):
            return None
        point = view.mapSceneToView(scene_position)
        return float(point.x()), float(point.y())

    def _is_real_drag(self, start, end) -> bool:
        """Did the pointer move far enough to mean a measurement?

        Judged in pixels rather than data units: a few units of TIME and a few
        units of ALTITUDE are not comparable, and the question being asked is
        about the gesture, not the data.
        """
        view = self.getPlotItem().vb
        (x_low, x_high), (y_low, y_high) = view.viewRange()
        width = max(1, view.width())
        height = max(1, view.height())
        dx = abs(end[0] - start[0]) / ((x_high - x_low) or 1.0) * width
        dy = abs(end[1] - start[1]) / ((y_high - y_low) or 1.0) * height
        return (dx * dx + dy * dy) >= 36.0   # 6 px

    def _update_slope_line(self, start, end) -> None:
        self._slope_item.setData([start[0], end[0]], [start[1], end[1]])
        self._slope_item.setVisible(True)

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        slope = dy / dx if dx else float("inf")
        self._slope_text.setText(
            "slope {0:.6g}".format(slope) if dx else "slope: vertical"
        )
        self._slope_text.setPos(end[0], end[1])
        self._slope_text.setVisible(True)

    def _inspect_at(self, position) -> None:
        """Report the curve and sample nearest the click, for the Inspect panel.

        Nearest is measured in *pixels*, not data units: the two axes have
        unrelated scales, so a distance computed in data space would always pick
        whichever channel happened to carry the larger numbers.
        """
        if not self._curves:
            return
        view = self.getPlotItem().vb
        scene_position = self.mapToScene(position.toPoint())
        if not self.sceneBoundingRect().contains(scene_position):
            return

        point = view.mapSceneToView(scene_position)
        target_x = float(point.x())
        target_y = float(point.y())

        (x_low, x_high), (y_low, y_high) = view.viewRange()
        x_span = (x_high - x_low) or 1.0
        y_span = (y_high - y_low) or 1.0

        projected = self.tile_state.plot3d
        best_key = None
        best_distance = None
        best_x = target_x
        best_point = None

        for key, entry in self._curves.items():
            if entry.x.size == 0:
                continue

            if projected:
                # A projected trajectory is not monotonic in either screen
                # axis, so there is nothing to binary-search: nearest_index()
                # assumes a sorted array and on a 3D tile returned whichever
                # sample its bisection happened to land on. Every point of the
                # thinned path is scanned instead, which is a few thousand at
                # most because the thinning is against the tile's pixel budget.
                count = min(entry.x.size, entry.y.size)
                dx = (entry.x[:count].astype(np.float64) - target_x) / x_span
                dy = (entry.y[:count].astype(np.float64) - target_y) / y_span
                squared = dx * dx + dy * dy
                index = int(np.argmin(squared))
                distance = float(squared[index])
                # What the rest of the application means by "where": the x
                # parameter's own value, not the screen coordinate the click
                # landed on. Without this the Inspect panel searched the TIME
                # column for a projection coordinate and reported whatever
                # sample that happened to find.
                source = entry.source_x
                reported = (float(source[index])
                            if source is not None and index < source.size
                            else float(entry.x[index]))
            else:
                index = Session.nearest_index(entry.x, target_x)
                if index >= entry.y.size:
                    continue
                dx = (float(entry.x[index]) - target_x) / x_span
                dy = (float(entry.y[index]) - target_y) / y_span
                distance = dx * dx + dy * dy
                reported = float(entry.x[index])

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_key = key
                best_x = reported
                best_point = (float(entry.x[index]), float(entry.y[index]))

        if best_key is None:
            return
        # In drawn coordinates, which on a 3D tile are the projected ones --
        # the dot has to land on the curve the click picked, wherever that is.
        if best_point is not None:
            self._show_inspect_dot(*best_point)

        try:
            file_index, sim_index, _param = best_key.split("|", 2)
        except ValueError:
            return
        self.point_inspected.emit(int(file_index), int(sim_index), best_x)

    def _show_inspect_dot(self, x: float, y: float) -> None:
        """Mark the sample the Inspect panel is reporting.

        The panel lists forty channels of one run at one instant, and without
        this there is nothing on the plot saying which instant -- the click
        lands on whichever curve was nearest, which is not always the one you
        were aiming at, and a table of numbers cannot tell you that.
        """
        if self._inspect_dot is None:
            return
        self._inspect_dot.setData([x], [y])
        self._inspect_dot.setVisible(True)

    def clear_inspect_dot(self) -> None:
        """Take the mark off. Leaving Inspect, or another tile becoming the
        active one: in both cases the panel is no longer reporting this point,
        so a dot still sitting on it is a claim that is no longer true."""
        if self._inspect_dot is not None:
            self._inspect_dot.setVisible(False)

    def mouseDoubleClickEvent(self, event) -> None:
        """Arm a fit. Left fits this tile and opens PARAMS; right fits them all.

        Double-clicking a tile is how you say "this one, now" -- so as well as
        re-framing it, the left button makes the tile active and opens PARAMS,
        which is where nine times out of ten the next click was going anyway.
        The activation goes first: PARAMS shows the active tile, and opening it
        on the previous one would be worse than not opening it at all.

        Armed rather than done, and that is the fix for a zoom that would not
        stick. Qt reports the second press of two quick clicks as a double
        click whether or not a drag follows it, so two rectangle zooms dragged
        in quick succession near the same spot arrived here -- and the fit
        threw away the zoom the first drag had just made. In Zoom mode, where
        dragging *is* the gesture and one drag follows another, that read as
        the zoom working only sometimes.

        So the fit waits for the release, and happens only if the pointer
        stayed put. A double click that turns out to be the start of a drag
        does nothing instead of undoing the view.
        """
        self.activated.emit(self._index)
        self._double_click = (
            "all" if event.button() == Qt.MouseButton.RightButton else "one")
        self._double_click_from = event.position()
        # Qt reports the second press of two quick clicks as a double click
        # whether a drag follows or not, so this is the event that stands in
        # for a press -- and the scene never sees one. That makes it the prime
        # suspect for a drag the scene will not start.
        zoomtrace.log("doubleclick", tile=self._index,
                      mode=self._state.mouse_mode.name,
                      button=zoomtrace.button_name(event),
                      arming=self._double_click,
                      at=zoomtrace.position(event),
                      **zoomtrace.scene_state(self))
        event.accept()

    def _resolve_double_click(self, event) -> bool:
        """Carry out an armed fit, if the pointer never moved. True if it did.

        Judged in pixels, with the same threshold a slope drag uses: the
        question is whether the hand moved, not what the numbers under it were.
        """
        gesture = self._double_click
        self._double_click = None
        origin = self._double_click_from
        self._double_click_from = None
        if gesture is None:
            return False

        try:
            now = event.position()
            dx = abs(float(now.x()) - float(origin.x()))
            dy = abs(float(now.y()) - float(origin.y()))
            moved = max(dx, dy) > DOUBLE_CLICK_SLACK
        except (AttributeError, TypeError):
            # Nothing to compare: treat it as the click it was reported as.
            moved = False
        if moved:
            # It was the beginning of a drag. The drag itself is lost either
            # way -- Qt has already spent that press on a double click -- but
            # the view the user had is not.
            zoomtrace.log("doubleclick.abandoned", gesture=gesture,
                          dx="{0:.1f}".format(dx), dy="{0:.1f}".format(dy))
            return True

        zoomtrace.log("doubleclick.fired", gesture=gesture)

        if gesture == "all":
            self.fit_all_requested.emit()
        else:
            self.autoscale()
            self.params_requested.emit(self._index)
        return True

    def autoscale(self) -> None:
        """Fit the data, except on axes the user has fixed.

        A locked axis is a deliberate constraint -- often applied across every
        tile sharing a variable so they can be compared -- so auto-scaling must
        not quietly undo it. Only the free axes move.
        """
        tile = self.tile_state
        if tile.plot3d:
            # A projection has no data axes to fit; "fit" means framing the
            # whole cube again. Both of the 2D calls below return early for a
            # 3D tile, which is why double-click and Ctrl+0 previously did
            # nothing at all here.
            tile.reset_3d_view()
            self._state.dirty = True
            self._apply_3d_view(tile)
            self.view_changed.emit()
            return

        self._autoscale_if_unlocked()
        # Only a tile with something in it counts as framed. Fitting an empty
        # one frames nothing, and recording that as "this tile has a view the
        # user chose" is what stopped the first curve being fitted: the way
        # to open PARAMS is to double-click the tile, double-clicking fits
        # it, and on a fresh start the tile being double-clicked is empty. So
        # the gesture that means "I am about to put something here" was the
        # one that decided the something should keep the empty tile's frame.
        if self._curves:
            self._store_view_range()
            tile.view_initialised = True
        self._resample_visible(force=True)
        # The fit is applied with range events suppressed, so that nothing
        # reacts to each of the several setXRange/setYRange calls it makes.
        # That also silenced the one signal the Inspect panel listens to, so
        # statistics followed a pan or a zoom but sat still after a fit --
        # showing figures for a window the plot was no longer displaying.
        self.view_changed.emit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # The pixel budget changed, so the decimation target did too. The guard
        # is for the call that arrives from inside super().__init__(), before
        # the timer exists -- see the class-level defaults above.
        if self._redraw_timer is not None:
            self._redraw_timer.start()
        # Captions are positioned as fractions of the viewport, so a resize
        # moves every one of them.
        self._refresh_texts()
