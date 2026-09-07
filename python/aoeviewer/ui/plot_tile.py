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

from ..model import (
    AppState,
    ColorMode,
    MouseMode,
    TileState,
    clamp_3d_half,
    tile_title,
)
from .. import projection, scales
from ..session import Session
from ..theme import (
    PLOT_AXIS_PT,
    PLOT_LABEL_PT,
    PLOT_LEGEND_PT,
    PLOT_TITLE_PT,
    Theme,
    plot_font,
)

# The drawing budget lives in aoeviewer.scales, which imports no Qt, so the
# arithmetic behind it can be tested without a running application -- which is
# exactly where the last mistake in it hid.
from ..scales import (  # noqa: F401
    MAX_TILE_POINTS,
    POINTS_PER_PIXEL,
    resample_key,
    resample_target,
)

# Stacking order of the things drawn on a tile. pyqtgraph puts plot curves at
# z = 0, so anything meant to be read *over* the data has to be above that.
# Annotations are furniture the engineer places deliberately; a band or a
# marker hidden behind whichever curve happens to cross it is not doing its job.
AREA_Z = 5          # shaded bands: translucent, so the data still reads through
AREA_LABEL_Z = 6    # a band's caption, above its own wash
MARKER_Z = 10       # marker lines and their dots
MARKER_LABEL_Z = 11
GRID_3D_Z = -20     # the back walls of a 3D box, behind everything

# Most rows a legend may have. pyqtgraph builds a QGraphicsGridLayout with two
# items per row, so a thousand-run study asked Qt to lay out several thousand
# graphics items in one column -- which is where the application stopped
# responding rather than merely slowed down. Past this the legend says how many
# more there are, which is the only useful thing a list that long can say.
MAX_LEGEND_ROWS = 60

# pyqtgraph takes axis label styling as CSS-ish keyword arguments. Named here so
# every setLabel call is styled the same way; a label left unstyled falls back to
# the widget font, which is sized for the panels and looms over a small tile.
_AXIS_LABEL_STYLE = {"font-size": "{0}pt".format(PLOT_AXIS_PT)}

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

    def __init__(self, on_moved=None):
        pg.GraphicsObject.__init__(self)
        GraphicsWidgetAnchor.__init__(self)
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
            painter.setPen(QPen(QColor(note.color)))

            metrics = painter.fontMetrics()
            x = width * float(note.x)
            top = height * float(note.y)
            y = top + metrics.ascent()
            lines = str(note.text).splitlines() or [""]
            widest = 0.0
            for line in lines:
                painter.drawText(QPointF(x, y), line)
                widest = max(widest, float(metrics.horizontalAdvance(line)))
                y += metrics.lineSpacing()

            # A few pixels of slack so a one-word caption is still easy to grab.
            boxes.append((note, QRectF(x - 3.0, top - 3.0,
                                       widest + 6.0,
                                       metrics.lineSpacing() * len(lines) + 6.0)))

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

    __slots__ = ("x", "y", "label", "key", "curve_key")

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

    activated = Signal(int)
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
    _slope_item = None
    _slope_text = None
    _rotating_from = None
    _gizmo = None
    _texts = None
    _nav_action = None

    def __init__(self, session: Session, state: AppState, index: int,
                 theme: Theme, parent=None):
        super().__init__(parent=parent)
        self._session = session
        self._state = state
        self._index = index
        self._theme = theme

        self._curves = {}
        self._groups: List[DrawGroup] = []
        self._legend: Optional[pg.LegendItem] = None
        self._overlays: List[pg.GraphicsObject] = []
        self._suppress_range_event = False

        plot = self.getPlotItem()
        plot.setMenuEnabled(False)
        plot.setClipToView(True)
        # Downsampling is ours, not pyqtgraph's -- theirs is stride-based and
        # would drop exactly the transients this tool exists to show.
        plot.setDownsampling(auto=False, mode="peak")
        plot.showGrid(x=True, y=True, alpha=0.25)

        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(25)
        self._redraw_timer.timeout.connect(self._resample_visible)

        self.sigRangeChanged.connect(self._on_range_changed)

        self._crosshair_label = pg.TextItem(anchor=(0, 1))
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

        self._slope_text = pg.TextItem(anchor=(0, 1), color="#ffb300")
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

        self._texts = TextOverlay(self._on_text_moved)
        self._texts.setParentItem(plot.getViewBox())
        self._texts.setPos(0, 0)

        self.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.apply_mouse_mode()

    # ------------------------------------------------------------- helpers

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
        self.setBackground(theme.plot_background)
        plot = self.getPlotItem()
        for axis_name in ("left", "bottom"):
            axis = plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(theme.plot_grid))
            axis.setTextPen(pg.mkPen(theme.plot_foreground))
            # Tick numbers default to the widget font, which is set for the
            # panels; on a page of tiles they then take more room than the
            # curves they are labelling.
            axis.setStyle(tickFont=plot_font(PLOT_AXIS_PT))
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
        if tile.color_mode == ColorMode.ISO and tile.color_iso_param:
            names.append(tile.color_iso_param)

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
                      size="{0}pt".format(PLOT_TITLE_PT))

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
        if not tile.plot3d and (tile.lock_x or tile.lock_y):
            # A locked axis stays locked whatever the mouse mode says.
            self._apply_axis_limits(tile)

        if self.tile_state.plot3d:
            # Pan is repurposed as rotate on a 3D tile, so the view box must not
            # also try to move -- but Zoom keeps its meaning: the wheel scales
            # the projection (handled in wheelEvent) and a drag rubber-bands a
            # region of it, which needs the view box left enabled.
            enabled = mode == MouseMode.ZOOM
            view.setMouseEnabled(x=enabled, y=enabled)
            view.setMouseMode(pg.ViewBox.RectMode)

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

    # ---------------------------------------------------------- ISO lookup

    def _iso_labels(self, iso_param: str) -> Dict[Tuple[int, int], str]:
        """(file, sim) -> ISO label, computed once per rebuild.

        Looking this up per curve would call into the core for every parameter
        of every run; the value is constant within a run, so once per run is
        enough.
        """
        labels: Dict[Tuple[int, int], str] = {}
        if not iso_param:
            return labels

        iso_state = self._state.iso_states.get(iso_param)
        grouped = bool(iso_state and iso_state.grouped and iso_state.groups)
        groups = iso_state.groups if grouped else []

        for file_index in {f for f, _s in self._state.selected_sims}:
            values = self._session.iso_values(file_index, iso_param)
            for sim_index in range(values.size):
                value = float(values[sim_index])
                if value != value:  # NaN
                    continue
                if grouped:
                    label = ""
                    for name, low, high, _active in groups:
                        if low <= value <= high:
                            label = name
                            break
                    if not label:
                        continue
                else:
                    label = self._session.format_value(value)
                labels[(file_index, sim_index)] = label
        return labels

    def _style_key(self, tile: TileState, param: str, sim_key: str,
                   iso_label: str) -> str:
        """The styling key for a curve, matching what the Style panel edits."""
        if tile.color_mode == ColorMode.PARAMETER:
            return "param::" + param
        if tile.color_mode == ColorMode.ISO and tile.color_iso_param and iso_label:
            return "iso::{0}::{1}".format(tile.color_iso_param, iso_label)
        return "sim::" + sim_key

    def _legend_label(self, tile: TileState, param: str, file_name: str,
                      sim_id: str, iso_label: str, dash: str, symbol: str) -> str:
        # The dash pattern and marker are shown by the legend's own sample of
        # the curve, so they are deliberately not spelled out in the text.
        del dash, symbol
        parts: List[str] = []

        if tile.legend_show_param:
            parts.append(param)

        if tile.color_mode == ColorMode.ISO and tile.legend_show_iso and iso_label:
            parts.append("{0} = {1}".format(tile.color_iso_param, iso_label))

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
            return
        if plot.legend is not None:
            scene = plot.legend.scene()
            if scene is not None:
                scene.removeItem(plot.legend)
            plot.legend = None

        legend = DraggableLegend(self._on_legend_moved, offset=(10, 10),
                                 labelTextSize="{0}pt".format(PLOT_LEGEND_PT))
        legend.setParentItem(plot.getViewBox())
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

        for overlay in self._overlays:
            plot.removeItem(overlay)
        self._overlays = []

        # pyqtgraph keeps exactly one legend per PlotItem and addLegend() hands
        # back the existing one, so recreating it per rebuild is not possible --
        # every rebuild would append another copy of every label. Clear its
        # entries instead and reuse it.
        self._legend = plot.legend
        if self._legend is not None:
            self._legend.clear()

        tile = self.tile_state
        self._slope_item.setVisible(False)
        self._slope_text.setVisible(False)
        # 2D and 3D want different things from the same mouse mode, so the
        # bindings are re-applied whenever the tile is rebuilt.
        self.apply_mouse_mode()

        plot.showGrid(x=tile.show_grid, y=tile.show_grid, alpha=0.25)
        plot.setLogMode(x=False, y=tile.log_y and not tile.plot3d)
        if tile.plot3d:
            # The axes of a projection are not either data axis, so labelling
            # them with parameter names would be a lie. The title carries the
            # three names instead.
            plot.setLabel("bottom", "", **_AXIS_LABEL_STYLE)
            plot.setLabel("left", "", **_AXIS_LABEL_STYLE)
            plot.showGrid(x=False, y=False)
        else:
            # Axis labels carry the unit the file declared, which is the one
            # place it genuinely belongs: "TIME" tells you nothing about
            # whether the axis is seconds or minutes.
            plot.setLabel("bottom", self._axis_label(tile.x_param),
                          **_AXIS_LABEL_STYLE)
            plot.setLabel("left", self._y_axis_label(tile),
                          **_AXIS_LABEL_STYLE)
        # Rebuilt every time, so the title tracks parameters being added and
        # removed rather than going stale.
        plot.setTitle(tile_title(tile), color=self._theme.plot_foreground,
                      size="{0}pt".format(PLOT_TITLE_PT))

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

        if tile.is_empty() or not self._state.selected_sims:
            self._drop_groups(previous_groups)
            if self._legend is not None:
                self._legend.setVisible(False)
            # An empty tile has no frame worth keeping, so the next parameter
            # dropped on it gets fitted rather than inheriting the range left
            # over from whatever used to be here.
            tile.view_initialised = False
            self._draw_overlays()
            self._refresh_texts()
            return

        if tile.show_legend:
            self._ensure_legend(plot)
            self._place_legend(tile)
            self._legend.setVisible(True)
        elif self._legend is not None:
            self._legend.setVisible(False)

        # No preload here. The grid hoists it into one background call covering
        # every visible tile, so the data is already resident by the time this
        # runs -- see PlotGrid.required_channels() and MainWindow.refresh_plots().
        by_file: Dict[int, List[int]] = {}
        for file_index, sim_index in sorted(self._state.selected_sims):
            by_file.setdefault(file_index, []).append(sim_index)

        iso_labels = (
            self._iso_labels(tile.color_iso_param)
            if tile.color_mode == ColorMode.ISO
            else {}
        )

        styles = self._state.styles
        # Identical legend entries are merged when neither file nor simulation
        # is shown. That is what turns a 50-run ISO plot into one legend row per
        # ISO value instead of fifty rows saying the same thing.
        deduplicate = not tile.legend_show_file and not tile.legend_show_sim
        seen_labels: Set[str] = set()
        legend_rows = 0
        hidden_rows = 0

        # (pen signature, legend name) -> plan. Curves that look identical are
        # painted together; see DrawGroup for why that is the difference between
        # a thousand-run study being viewable and not.
        #
        # A plan rather than a DrawGroup, because which old item can be reused
        # is only knowable once a group's whole membership is known, and that is
        # not true until this loop has finished.
        plans: Dict[tuple, _GroupPlan] = {}
        order: List[tuple] = []

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

                sim = simulations[sim_index]
                iso_label = iso_labels.get((file_index, sim_index), "")

                for param in tile.y_params:
                    y = self._session.channel(file_index, sim_index, param)
                    if y.size == 0:
                        continue

                    style_key = self._style_key(tile, param, sim.key, iso_label)
                    color = styles.color_for(style_key)
                    dash = styles.dash_for(style_key)
                    symbol = styles.symbol_for(style_key)
                    width = styles.width_for(style_key)

                    label = self._legend_label(
                        tile, param, file_name, sim.sim_id, iso_label, dash, symbol
                    )

                    show_in_legend = tile.show_legend
                    if show_in_legend and deduplicate:
                        if label in seen_labels:
                            show_in_legend = False
                        else:
                            seen_labels.add(label)
                    if show_in_legend:
                        if legend_rows >= MAX_LEGEND_ROWS:
                            show_in_legend = False
                            hidden_rows += 1
                        else:
                            legend_rows += 1

                    marker = symbol or ("o" if tile.show_points else None)

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

        self._adopt_groups(plot, previous_groups,
                           [plans[k] for k in order])

        if hidden_rows and self._legend is not None:
            # Say what was left out rather than silently truncating.
            spacer = pg.PlotDataItem(pen=None, name="... and {0} more".format(
                hidden_rows))
            plot.addItem(spacer)
            self._groups.append(DrawGroup(spacer))

        # The view is the user's. A rebuild happens for all sorts of reasons
        # that say nothing about what should be on screen -- restyling a curve,
        # toggling the grid, dragging a marker, coming back to a page -- and
        # re-fitting on every one of them threw away whatever they had zoomed
        # in on. So fit only when the tile has never been framed, which is the
        # moment it first acquires a parameter; after that the frame moves when
        # the user asks (Fit, double-click) or changes the SCALE limits.
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
        self._draw_overlays()
        self._refresh_texts()

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

    # The arithmetic lives in aoeviewer.projection: it needs no Qt, so it can
    # be tested without a QApplication, which is where the interesting failures
    # would be.
    project_3d = staticmethod(projection.project)
    _normalise = staticmethod(projection.normalise)

    def _rebuild_3d(self, plot, tile: TileState) -> None:
        """Draw the tile as a rotatable 3D projection."""
        if self._legend is not None:
            self._legend.setVisible(False)

        if tile.is_empty() or not self._state.selected_sims:
            self._gizmo.setVisible(False)
            return

        y_param = tile.y_params[0]
        series = []
        x_bounds = y_bounds = z_bounds = None

        for file_index, sim_index in sorted(self._state.selected_sims):
            handle = self._session.file(file_index)
            if handle is None or sim_index >= handle.simulation_count:
                continue
            x = self._session.channel(file_index, sim_index, tile.x_param)
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
        iso_labels = (
            self._iso_labels(tile.color_iso_param)
            if tile.color_mode == ColorMode.ISO
            else {}
        )

        # Grouped by pen exactly as the 2D path is, and for the same reason: a
        # thousand trajectories is a thousand items otherwise. A 3D tile is
        # rebuilt on every step of a rotation, so the item count here costs on
        # every frame of a drag rather than once.
        groups: Dict[tuple, List[np.ndarray]] = {}
        pens: Dict[tuple, QPen] = {}
        gap = np.array([np.nan], dtype=np.float32)

        for file_index, sim_index, file_name, sim, x, y, z in series:
            iso_label = iso_labels.get((file_index, sim_index), "")
            style_key = self._style_key(tile, tile.z_param, sim.key, iso_label)
            color = styles.color_for(style_key)
            dash = styles.dash_for(style_key)
            width = styles.width_for(style_key)

            sx, sy, _depth = self.project_3d(
                self._normalise(x, x_bounds),
                self._normalise(y, y_bounds),
                self._normalise(z, z_bounds),
                tile.azimuth, tile.elevation,
            )
            sx = np.asarray(sx, dtype=np.float32)
            sy = np.asarray(sy, dtype=np.float32)

            key = "{0}|{1}|{2}".format(file_index, sim_index, tile.z_param)
            # The stored x/y are the *projected* coordinates, which is what the
            # cursor and inspect code needs to hit-test against what is drawn.
            self._curves[key] = CurveEntry(
                sx, sy, "{0} · {1}".format(sim.sim_id, file_name), style_key)

            group_key = (color, dash, width)
            if group_key not in groups:
                groups[group_key] = []
                pen = QPen(QColor(color))
                pen.setWidthF(width)
                pen.setStyle(_PEN_STYLES.get(dash, Qt.PenStyle.SolidLine))
                pen.setCosmetic(True)
                pens[group_key] = pen
            pieces = groups[group_key]
            if pieces:
                pieces.append((gap, gap))
            pieces.append((sx, sy))

        for group_key, pieces in groups.items():
            cx = np.concatenate([p[0] for p in pieces])
            cy = np.concatenate([p[1] for p in pieces])
            item = pg.PlotDataItem(cx, cy, pen=pens[group_key], antialias=True,
                                   connect="finite")
            plot.addItem(item)
            group = DrawGroup(item)
            # Nothing re-samples a 3D tile -- the thinning happens once, above --
            # so the group holds only the item, for removal on the next rebuild.
            self._groups.append(group)

        self._apply_3d_view(tile)

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

        The geometry is in aoeviewer.projection, which has no Qt in it, so the
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
                                   color=label_colour, anchor=(0.5, 0.5))
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
        self.rebuild()

    # --------------------------------------------------------- resampling

    def _on_range_changed(self, *_args) -> None:
        if self._suppress_range_event:
            return
        self._store_view_range()
        if self._redraw_timer is not None:
            self._redraw_timer.start()
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

    # Log-axis conversion lives in aoeviewer.scales, which imports no Qt so the
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
        # aoeviewer.scales -- which imports no Qt, so that arithmetic can be
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

    def _draw_overlays(self) -> None:
        """Markers and shaded bands.

        A marker is a condition on one parameter. On a tile plotted against that
        same parameter it is a vertical line. On any other tile it is drawn as a
        dot on each curve, at the sample where the condition holds -- so a
        marker at TIME = 5 shows you where five seconds falls on ALPHA = f(NZ).
        """
        plot = self.getPlotItem()
        tile = self.tile_state

        if tile.plot3d:
            # Neither form makes sense in a projection: a vertical line is not
            # a plane, and the dot indices refer to the thinned projected
            # arrays rather than to samples. The 3D box carries the scale.
            return

        for marker in self._state.markers:
            if not marker.visible or not marker.x_param:
                continue

            if marker.x_param == tile.x_param:
                movable = self._annotations_movable()
                line = pg.InfiniteLine(
                    pos=marker.position,
                    angle=90,
                    pen=pg.mkPen(marker.color, width=1.5, style=Qt.PenStyle.DashLine),
                    label=marker.label or None,
                    labelOpts={"color": marker.color, "position": 0.92,
                               "font-size": "{0}pt".format(PLOT_LABEL_PT)},
                    movable=movable,
                    hoverPen=pg.mkPen(marker.color, width=3.0),
                )
                if movable:
                    line.setCursor(Qt.CursorShape.SizeHorCursor)
                    line.sigPositionChangeFinished.connect(
                        lambda item, m=marker: self._marker_dragged(m, item)
                    )
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
                if movable:
                    # Both edges and the band as a whole become draggable.
                    region.setHoverBrush(pg.mkBrush(self._hover_colour(area)))
                    region.sigRegionChangeFinished.connect(
                        lambda item, a=area: self._area_dragged(a, item)
                    )
                # In front of the curves, not behind them. A band is a
                # translucent wash marking a phase of the run, and the whole
                # point of it is to see which part of the trace it covers --
                # sat underneath, its edges disappeared wherever a curve
                # crossed them, and on a busy tile it was barely visible at
                # all. The alpha is what keeps the data readable through it.
                region.setZValue(AREA_Z)
                plot.addItem(region, ignoreBounds=True)
                self._overlays.append(region)
                if area.label:
                    caption = pg.TextItem(area.label, color=area.color,
                                          anchor=(0.5, 0.0))
                    caption.setFont(plot_font(PLOT_LABEL_PT))
                    caption.setPos((area.x_min + area.x_max) * 0.5,
                                   self._view_top())
                    caption.setZValue(AREA_LABEL_Z)
                    plot.addItem(caption, ignoreBounds=True)
                    self._overlays.append(caption)
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

    def _view_top(self) -> float:
        """Data y of the top of the visible area, for captions."""
        try:
            (_x_low, _x_high), (_y_low, y_high) = self.getPlotItem().vb.viewRange()
        except (TypeError, ValueError):
            return 0.0
        return float(y_high)

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
                event.accept()
                return

        if (self._state.mouse_mode == MouseMode.CROSSHAIR
                and event.button() == Qt.MouseButton.LeftButton):
            # The press starts a potential slope drag; whether it was a click
            # or a drag is only known on release.
            self._slope_origin = self._data_point(event.position())
            self._slope_item.setVisible(False)
            self._slope_text.setVisible(False)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
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
        if self._rotating_from is not None:
            self._rotating_from = None
            self._nav_action = None
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
            event.accept()
            return

        super().mouseReleaseEvent(event)

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

        best_key = None
        best_distance = None
        best_x = target_x

        for key, entry in self._curves.items():
            if entry.x.size == 0:
                continue
            index = Session.nearest_index(entry.x, target_x)
            if index >= entry.y.size:
                continue
            dx = (float(entry.x[index]) - target_x) / x_span
            dy = (float(entry.y[index]) - target_y) / y_span
            distance = dx * dx + dy * dy
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_key = key
                best_x = float(entry.x[index])

        if best_key is None:
            return
        try:
            file_index, sim_index, _param = best_key.split("|", 2)
        except ValueError:
            return
        self.point_inspected.emit(int(file_index), int(sim_index), best_x)

    def mouseDoubleClickEvent(self, event) -> None:
        """Refit the view, honouring whichever axes are locked."""
        self.autoscale()
        event.accept()

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
