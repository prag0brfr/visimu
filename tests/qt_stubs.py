# Author: Paulo R. A. Gilz
"""Just enough PySide6 and pyqtgraph to import the drawing code without Qt.

The redraw path is where the application's responsiveness is decided, and it is
also the part that cannot be exercised on a build machine: PySide6 is a large
binary wheel and a plot widget needs a display. Testing it only by hand meant
that the two worst performance bugs found so far -- a point budget that raised
the target instead of lowering it, and a border repaint that redrew every tile
in the grid on every click -- were both invisible to the suite.

So the modules are stubbed. The stubs are deliberately dumb: they record what
was asked of them and do nothing else. That is enough for the questions worth
asking here, which are all of the form "given this state, does the code decide
to do the expensive thing?" -- how Qt would then draw it is not in question.

Import this module before ``aoeviewer.ui.plot_tile``; it installs itself into
sys.modules on import.
"""

from __future__ import annotations

import sys
import types


class _Any:
    """Accepts any construction, any attribute and any call.

    Attributes are created on demand and remembered, so ``Qt.PenStyle.DashLine``
    is the same object every time and can be used as a dictionary key -- which
    plot_tile does, at import time.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = _Any()
        object.__setattr__(self, name, value)
        return value

    def __call__(self, *args, **kwargs):
        return _Any()


class Signal:
    """A class attribute that becomes a per-instance recorder."""

    def __init__(self, *types_):
        self._types = types_
        self._name = None

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        bound = instance.__dict__.get("_signal_" + str(self._name))
        if bound is None:
            bound = _BoundSignal()
            instance.__dict__["_signal_" + str(self._name)] = bound
        return bound


class _BoundSignal:
    def __init__(self):
        self.emitted = []
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def disconnect(self, slot=None):
        self._slots = []

    def emit(self, *args):
        self.emitted.append(args)
        for slot in list(self._slots):
            slot(*args)


class QWidget:
    """The bit of QWidget the tiles actually lean on.

    setStyleSheet counts its calls rather than ignoring them: that count is the
    subject of a test, because Qt repolishes and repaints on every one of them
    however little the sheet changed.
    """

    def __init__(self, *args, **kwargs):
        self.style_sheets = []
        self._enabled = True

    def setStyleSheet(self, sheet):
        self.style_sheets.append(sheet)

    def styleSheet(self):
        return self.style_sheets[-1] if self.style_sheets else ""

    def setEnabled(self, enabled):
        self._enabled = bool(enabled)

    def isEnabled(self):
        return self._enabled

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


class PlotDataItem:
    """Records the pen, the name and the points it was last given."""

    def __init__(self, *args, **kwargs):
        self.opts = dict(kwargs)
        self.data_calls = []
        self.pen_calls = []
        self.symbol_calls = []
        if len(args) >= 2:
            self.data_calls.append((args[0], args[1]))

    def setData(self, *args, **kwargs):
        self.data_calls.append(args)

    def setPen(self, pen):
        self.pen_calls.append(pen)

    def setSymbol(self, symbol):
        self.symbol_calls.append(symbol)
        self.opts["symbol"] = symbol

    def setSymbolSize(self, size):
        self.opts["symbolSize"] = size

    def setSymbolBrush(self, brush):
        self.opts["symbolBrush"] = brush

    def setSymbolPen(self, pen):
        self.opts["symbolPen"] = pen

    def name(self):
        return self.opts.get("name")

    def setVisible(self, visible):
        self.opts["visible"] = visible


class LegendItem:
    def __init__(self, *args, **kwargs):
        self.rows = []
        self.visible = True

    def clear(self):
        self.rows = []

    def addItem(self, item, name=None):
        self.rows.append((item, name))

    def removeItem(self, item):
        self.rows = [row for row in self.rows if row[0] is not item]

    def setVisible(self, visible):
        self.visible = bool(visible)

    def scene(self):
        return None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


class PlotItem:
    """Tracks which items are on the plot, which is what the reuse tests ask."""

    def __init__(self):
        self.items = []
        self.added = []
        self.removed = []
        self.legend = None
        self._view_range = [[0.0, 1.0], [0.0, 1.0]]
        self.vb = _ViewBox()

    def addItem(self, item, *args, **kwargs):
        self.items.append(item)
        self.added.append(item)
        name = kwargs.get("name", getattr(item, "opts", {}).get("name"))
        if name is not None and self.legend is not None:
            self.legend.addItem(item, name)

    def removeItem(self, item):
        if item in self.items:
            self.items.remove(item)
        self.removed.append(item)
        if self.legend is not None:
            self.legend.removeItem(item)

    def getViewBox(self):
        return self.vb

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


class _ViewBox:
    def __init__(self):
        self._range = [[0.0, 1.0], [0.0, 1.0]]
        self._width = 900

    def viewRange(self):
        return self._range

    def width(self):
        return self._width

    def boundingRect(self):
        return _Any()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


class PlotWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._plot_item = PlotItem()

    def getPlotItem(self):
        return self._plot_item


class QComboBox:
    """Real enough for the style table: findData has to return an int."""

    def __init__(self, *args, **kwargs):
        self._items = []          # (label, data)
        self._index = -1
        self.currentTextChanged = _BoundSignal()
        self.currentIndexChanged = _BoundSignal()

    def addItem(self, label, data=None):
        self._items.append((label, data))
        if self._index < 0:
            self.setCurrentIndex(0)

    def addItems(self, labels):
        for label in labels:
            self.addItem(label)

    def clear(self):
        self._items = []
        self._index = -1

    def count(self):
        return len(self._items)

    def findData(self, data):
        for index, (_label, value) in enumerate(self._items):
            if value == data:
                return index
        return -1

    def setCurrentIndex(self, index):
        if index == self._index or not (0 <= index < len(self._items)):
            if not (0 <= index < len(self._items)):
                return
        self._index = index
        self.currentIndexChanged.emit(index)
        self.currentTextChanged.emit(self._items[index][0])

    def setCurrentText(self, text):
        for index, (label, _value) in enumerate(self._items):
            if label == text:
                self.setCurrentIndex(index)
                return

    def currentText(self):
        return self._items[self._index][0] if self._index >= 0 else ""

    def currentData(self):
        return self._items[self._index][1] if self._index >= 0 else None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


class QTableWidget:
    """A grid of items and cell widgets, with Qt's ownership behaviour.

    The one behaviour that matters here: shrinking the row count destroys the
    widgets in the rows that go. Reusing a row's widgets is only safe because
    rows are never reordered, and a stub that quietly kept them would hide the
    difference.
    """

    def __init__(self, rows=0, columns=0):
        self._rows = 0
        self._columns = columns
        self._items = {}
        self._widgets = {}
        self.updates_enabled = True
        self.created = 0        # cell widgets ever installed
        self.setRowCount(rows)

    def setRowCount(self, count):
        for (row, column) in list(self._items):
            if row >= count:
                del self._items[(row, column)]
        for (row, column) in list(self._widgets):
            if row >= count:
                del self._widgets[(row, column)]
        self._rows = count

    def rowCount(self):
        return self._rows

    def setItem(self, row, column, item):
        self._items[(row, column)] = item

    def item(self, row, column):
        return self._items.get((row, column))

    def setCellWidget(self, row, column, widget):
        self._widgets[(row, column)] = widget
        self.created += 1

    def cellWidget(self, row, column):
        return self._widgets.get((row, column))

    def setUpdatesEnabled(self, enabled):
        self.updates_enabled = bool(enabled)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


class QTableWidgetItem:
    def __init__(self, text=""):
        self._text = text
        self._tooltip = ""
        self._flags = None

    def setText(self, text):
        self._text = text

    def text(self):
        return self._text

    def setToolTip(self, text):
        self._tooltip = text

    def setFlags(self, flags):
        self._flags = flags

    def flags(self):
        return self._flags


class _AutoModule(types.ModuleType):
    """Hands out a permissive class for any name nobody has faked.

    The UI imports several dozen Qt names and uses almost none of them in the
    logic worth testing, so faking each one by hand would be noise. Anything a
    test actually inspects is set explicitly in install() and wins over this.
    """

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        made = type(name, (_Any,), {})
        setattr(self, name, made)
        return made


def install() -> None:
    """Put the stub modules into sys.modules, whether or not Qt is installed.

    Unconditional on purpose. What these tests ask is whether *our* code decides
    to do expensive work, and that answer must not depend on whether the machine
    running the suite happens to have PySide6 -- a test that passes on the build
    server and cannot run on the developer's laptop is worth very little. Using
    the real Qt would also mean a QApplication and a display for what is
    ultimately a question about a few comparisons.

    The consequence is that importing this module poisons Qt for the rest of the
    process, so no other test module may import PySide6. None does: the GUI is
    otherwise checked by parsing it (see test_construction_order).
    """
    core = _AutoModule("PySide6.QtCore")
    core.Signal = Signal
    core.Qt = _Any()
    for name in ("QPointF", "QRectF", "QTimer", "QObject", "QThread"):
        setattr(core, name, type(name, (_Any,), {}))

    gui = _AutoModule("PySide6.QtGui")
    for name in ("QBrush", "QColor", "QPainter", "QPen", "QPolygonF", "QFont"):
        setattr(gui, name, type(name, (_Any,), {}))

    widgets = _AutoModule("PySide6.QtWidgets")
    widgets.QWidget = QWidget
    widgets.QComboBox = QComboBox
    widgets.QTableWidget = QTableWidget
    widgets.QTableWidgetItem = QTableWidgetItem

    pyside = types.ModuleType("PySide6")
    pyside._aoe_stub = True
    pyside.__version__ = "stub"
    pyside.QtCore, pyside.QtGui, pyside.QtWidgets = core, gui, widgets

    pg = _AutoModule("pyqtgraph")
    pg._aoe_stub = True
    pg.__version__ = "stub"
    pg.PlotWidget = PlotWidget
    pg.PlotDataItem = PlotDataItem
    pg.PlotItem = PlotItem
    pg.LegendItem = LegendItem
    pg.GraphicsObject = type("GraphicsObject", (_Any,), {})
    pg.TextItem = type("TextItem", (_Any,), {})
    pg.InfiniteLine = type("InfiniteLine", (_Any,), {})
    pg.LinearRegionItem = type("LinearRegionItem", (_Any,), {})
    pg.ScatterPlotItem = type("ScatterPlotItem", (_Any,), {})
    pg.mkPen = lambda *a, **k: _Any()
    pg.mkBrush = lambda *a, **k: _Any()

    anchor_module = types.ModuleType(
        "pyqtgraph.graphicsItems.GraphicsWidgetAnchor")
    anchor_module.GraphicsWidgetAnchor = type(
        "GraphicsWidgetAnchor", (_Any,), {})
    items = types.ModuleType("pyqtgraph.graphicsItems")
    items.GraphicsWidgetAnchor = anchor_module
    pg.graphicsItems = items

    sys.modules.update({
        "PySide6": pyside,
        "PySide6.QtCore": core,
        "PySide6.QtGui": gui,
        "PySide6.QtWidgets": widgets,
        "pyqtgraph": pg,
        "pyqtgraph.graphicsItems": items,
        "pyqtgraph.graphicsItems.GraphicsWidgetAnchor": anchor_module,
    })


install()
