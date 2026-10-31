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

Import this module before ``visimu.ui.plot_tile``; it installs itself into
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
        # QWidget.blockSignals is how the panels build a list without their own
        # handlers firing on every item, so a stub that ignored it would test a
        # path the application never takes.
        self.blocked = False

    def connect(self, slot):
        self._slots.append(slot)

    def disconnect(self, slot=None):
        self._slots = []

    def emit(self, *args):
        if self.blocked:
            return
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

    def setVisible(self, visible):
        self._visible = bool(visible)

    def isVisible(self):
        return bool(getattr(self, "_visible", True))

    def setFocusPolicy(self, policy):
        # Recorded: which widget may take the focus decides whether a key
        # reaches the one that handles it.
        self.focus_policy = policy

    def testAttribute(self, _attribute):
        # False, and real rather than invented, because the tab shortcuts ask
        # a widget whether it takes typed text and an _Any answer is truthy --
        # which would make every widget in the window look like a text box.
        return False

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


class QApplication(_Any):
    """The few class methods the application asks of itself.

    Real answers rather than invented ones: whether a letter may be taken as
    a tab shortcut turns on which widget has the focus and whether a dialog
    is up, so both have to be answerable -- and "nothing focused, no dialog"
    is the honest default for a stub with no event loop.
    """

    @staticmethod
    def focusWidget():
        return None

    @staticmethod
    def activeModalWidget():
        return None

    @staticmethod
    def instance():
        return None


class QAbstractButton(QWidget):
    """A checkable button that really remembers its text and its state.

    The tab rail is a column of these and nothing else, and everything worth
    testing about it -- which tab is current, what order the names are in,
    what the tooltip advertises -- is read back off them. An _Any would answer
    every one of those questions with a fresh object.
    """

    clicked = Signal(bool)

    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self._text = ""
        self._tip = ""
        self._checkable = False
        self._checked = False
        self.focus_policy = None
        self._height = 40

    def setText(self, text):
        self._text = text

    def text(self):
        return self._text

    def setToolTip(self, tip):
        self._tip = tip

    def toolTip(self):
        return self._tip

    def setCheckable(self, value):
        self._checkable = bool(value)

    def isCheckable(self):
        return self._checkable

    def setChecked(self, value):
        self._checked = bool(value)

    def isChecked(self):
        return self._checked

    def underMouse(self):
        return False

    def height(self):
        return self._height


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

    def setZValue(self, z):
        self.opts["z"] = z

    def zValue(self):
        return self.opts.get("z", 0)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


class _GraphicsGridLayout:
    """Just the contents margins, which is all the legend sets on it."""

    def __init__(self):
        self.margins = (0, 0, 0, 0)

    def setContentsMargins(self, left, top, right, bottom):
        self.margins = (left, top, right, bottom)

    def getContentsMargins(self):
        return self.margins

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


class LegendItem:
    def __init__(self, *args, **kwargs):
        self.rows = []
        self.visible = True
        self.z = 0.0
        self.layout = _GraphicsGridLayout()
        # Real enough for the ground tests: pyqtgraph keeps the fill, the
        # border and the text colour in opts, and paint() reads them from
        # there.
        self.opts = {
            "brush": kwargs.get("brush"),
            "pen": kwargs.get("pen"),
            "labelTextColor": kwargs.get("labelTextColor"),
            "labelTextSize": kwargs.get("labelTextSize"),
            "offset": kwargs.get("offset"),
        }

    def setLabelTextColor(self, colour):
        self.opts["labelTextColor"] = colour

    def setZValue(self, z):
        self.z = float(z)

    def zValue(self):
        return self.z

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


def _clamped(view, axis, low, high):
    """A requested range, held inside whatever bounds the view box carries.

    Tolerant of a view box a test built by hand rather than from ViewBox
    below: those have no bounds to hold anything inside, so the range is its
    own answer.
    """
    low, high = float(low), float(high)
    clamp = getattr(view, "clamp", None)
    if not callable(clamp):
        return low, high
    try:
        result = clamp(axis, low, high)
        first, second = result
        return float(first), float(second)
    except (TypeError, ValueError):
        return low, high


class ViewBox(_Any):
    """pyqtgraph's view box, with the two pieces our subclass needs.

    The mode constants are read off the *class*, which _Any cannot answer: it
    invents attributes on instances only.

    mouseDragEvent has to be a real method rather than an invented attribute
    for a different reason. The tile's view box hands every drag it does not
    reinterpret back to pyqtgraph with super(), and super() looks through the
    class dictionaries -- it does not go through __getattr__. An invented
    attribute is therefore invisible to it, and a stub without this raised
    exactly where the real one does the work. Recording the call also lets a
    test assert that a pan, an axis drag or a right-drag was handed back
    rather than quietly swallowed.
    """

    PanMode = 3
    RectMode = 1

    def mouseDragEvent(self, event, axis=None):
        self.__dict__.setdefault("delegated_drags", []).append((event, axis))

    # setLimits is a hard bound the view box *keeps*, and keeping it is the
    # point: it outlives the lock that set it, so a stub that accepted the
    # call and forgot it reported a fit as working when the real view box
    # would have clamped it straight back to the range the user had just
    # released. Modelled, therefore, rather than invented.

    def setLimits(self, **kwargs):
        limits = self.__dict__.setdefault("limits", {})
        limits.update(kwargs)
        self.__dict__.setdefault("limit_writes", []).append(dict(kwargs))

    def clamp(self, axis, low, high):
        """What pyqtgraph does to a requested range: hold it inside the hard
        bounds, and inside the maximum span, before it becomes the view."""
        limits = self.__dict__.get("limits") or {}
        if not isinstance(limits, dict):
            limits = {}
        if axis == "x":
            lo_key, hi_key, span_key = "xMin", "xMax", "maxXRange"
        else:
            lo_key, hi_key, span_key = "yMin", "yMax", "maxYRange"
        floor, ceiling = limits.get(lo_key), limits.get(hi_key)
        span = limits.get(span_key)

        if span is not None and (high - low) > span:
            middle = (low + high) / 2.0
            low, high = middle - span / 2.0, middle + span / 2.0
        if floor is not None and low < floor:
            low, high = floor, high + (floor - low)
        if ceiling is not None and high > ceiling:
            low, high = low - (high - ceiling), ceiling
        if floor is not None:
            low = max(low, floor)
        if ceiling is not None:
            high = min(high, ceiling)
        return low, high


class _Axis:
    """One of a plot's axes, recording what the tile does to it.

    Falling through to _Any was enough while the tile only styled the axes.
    It stopped being enough once the tile started *sizing* the bottom one: a
    fresh _Any per getAxis() call cannot show that a height was set, nor what
    it was, which is the whole of the behaviour that keeps the tick numbers
    inside the tile.
    """

    def __init__(self, name, plot):
        self.name = name
        self._plot = plot
        self.label_shown = None
        self.heights = []
        self.style_calls = []
        # The defaults pyqtgraph itself carries, because the tile reads them
        # to work out how much room the numbers need.
        self.style = {"tickTextOffset": (5, 2), "tickLength": -5}

    def showLabel(self, show=True):
        self.label_shown = bool(show)
        # Counted as a label write, like setLabel: "the caption was decided on
        # this rebuild" is the thing the reuse tests are asking about, and
        # hiding it is a way of deciding it.
        self._plot.label_calls.append((self.name, None if not show else ""))

    def setHeight(self, height=None):
        self.heights.append(height)

    def setStyle(self, **kwargs):
        self.style_calls.append(kwargs)
        self.style.update(kwargs)

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
        self.titles = []
        self.title_sizes = []
        self.grid_calls = []
        self.log_calls = []
        self.label_calls = []
        self._axes = {}
        # Range writes are recorded because fitting is behaviour now: a tile
        # that has just acquired its first curve is supposed to frame itself,
        # and every later rebuild is supposed to leave the frame alone. Without
        # these the fit falls through to _Any and no test can see either half.
        self.range_writes = []

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

    def setXRange(self, low, high, **kwargs):
        # Through the view box's bounds, as the real one goes.
        low, high = _clamped(self.vb, "x", low, high)
        self.range_writes.append(("x", float(low), float(high)))
        self._view_range[0] = [float(low), float(high)]
        self.vb._range[0] = [float(low), float(high)]

    def setYRange(self, low, high, **kwargs):
        low, high = _clamped(self.vb, "y", low, high)
        self.range_writes.append(("y", float(low), float(high)))
        self._view_range[1] = [float(low), float(high)]
        self.vb._range[1] = [float(low), float(high)]

    def getViewBox(self):
        return self.vb

    # Recorded rather than swallowed. These four are the plot's own settings,
    # and whether a rebuild writes them *again* with what it wrote last time
    # is the behaviour under test: setLogMode in particular propagates to
    # every data item and re-enables auto-range, so a redundant write is not
    # free. A stub that let them fall through to _Any could not tell a guarded
    # rebuild from an unguarded one.
    def setTitle(self, title=None, **kwargs):
        self.titles.append(title)
        # The size is a point string the tile builds from the UI scale, so it
        # is the one place a test can see the scale reach pyqtgraph.
        self.title_sizes.append(kwargs.get("size"))

    def showGrid(self, **kwargs):
        self.grid_calls.append(kwargs)

    def setLogMode(self, **kwargs):
        self.log_calls.append(kwargs)

    def setLabel(self, axis=None, text=None, **kwargs):
        self.label_calls.append((axis, text))

    def getAxis(self, name):
        """The same object every time, so what was done to it can be read."""
        if name not in self._axes:
            self._axes[name] = _Axis(name, self)
        return self._axes[name]

    @property
    def axis_writes(self):
        """How many times the four settings above have been written."""
        return (len(self.titles) + len(self.grid_calls)
                + len(self.log_calls) + len(self.label_calls))

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


class _ViewBox:
    """The view box a PlotItem owns, as far as the tile uses one.

    setLimits is modelled rather than invented, and the ranges a PlotItem sets
    are held inside it: the bound a lock puts here outlives the lock, so a
    stub that forgot the call reported a fit as working where the real view
    box clamps it straight back to the range just released. See ViewBox.clamp
    above, whose behaviour this shares.
    """

    def __init__(self):
        self._range = [[0.0, 1.0], [0.0, 1.0]]
        self._width = 900
        self.limits = {}
        self.limit_writes = []

    def viewRange(self):
        return self._range

    def width(self):
        return self._width

    def boundingRect(self):
        return _Any()

    def setLimits(self, **kwargs):
        self.limits.update(kwargs)
        self.limit_writes.append(dict(kwargs))

    def clamp(self, axis, low, high):
        return ViewBox.clamp(self, axis, low, high)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


class PlotWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._plot_item = PlotItem()
        # Mouse events the tile hands back to Qt. Real methods rather than
        # invented attributes, because super() looks through the class
        # dictionaries and does not go through __getattr__ -- so a tile
        # passing a press or a release on raised where the real widget does
        # the work. Recorded, so a test can tell "handed back" from
        # "swallowed".
        self.events_passed_on = []

    def getPlotItem(self):
        return self._plot_item

    def mousePressEvent(self, event):
        self.events_passed_on.append(("press", event))

    def mouseReleaseEvent(self, event):
        self.events_passed_on.append(("release", event))

    def mouseDoubleClickEvent(self, event):
        self.events_passed_on.append(("double", event))

    def wheelEvent(self, event):
        self.events_passed_on.append(("wheel", event))


class QComboBox:
    """Real enough for the style table: findData has to return an int."""

    def __init__(self, *args, **kwargs):
        self._items = []          # (label, data)
        self._index = -1
        self.focus_policy = None
        self.currentTextChanged = _BoundSignal()
        self.currentIndexChanged = _BoundSignal()

    def setFocusPolicy(self, policy):
        # Recorded, not ignored: a cell widget that takes focus when clicked
        # makes the view drop the selection around it, which is the
        # difference between setting one row's style and setting five.
        self.focus_policy = policy

    def setEditable(self, value):
        self.editable = bool(value)

    def isEditable(self):
        return bool(getattr(self, "editable", False) is True)

    def showPopup(self):
        # The real one drops a list window; here it only has to exist, so
        # StyleCombo can call up to it.
        self.popup_shown = True

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

    def itemText(self, index):
        return self._items[index][0] if 0 <= index < len(self._items) else ""

    def itemData(self, index):
        return self._items[index][1] if 0 <= index < len(self._items) else None

    # What a QVariant will and will not hand back intact. Anything else --
    # a tuple, a list, an object -- goes into item data and does not come out
    # of findData comparing equal to what went in, so the real widget answers
    # "not in the list" for something that is in it. Modelled, because a stub
    # that compared Python values reported a panel as working when the real
    # one could not select a row at all: the DIFF tab's two run boxes reset to
    # "--" on every refresh and the runs could not be chosen.
    _MATCHABLE = (str, int, float, bool, type(None))

    def findData(self, data):
        if not isinstance(data, self._MATCHABLE):
            return -1
        for index, (_label, value) in enumerate(self._items):
            if isinstance(value, self._MATCHABLE) and value == data:
                return index
        return -1

    def findText(self, text):
        # An int, like Qt's. Without this the catch-all __getattr__ hands back
        # an _Any, and `if combo.findText(name) >= 0` raises a TypeError that
        # looks like a bug in the panel rather than a gap in the stub.
        for index, (label, _value) in enumerate(self._items):
            if label == text:
                return index
        return -1

    def blockSignals(self, blocked):
        self.currentTextChanged.blocked = bool(blocked)
        self.currentIndexChanged.blocked = bool(blocked)
        return False

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

    def currentIndex(self):
        # An int, like Qt's, and for the same reason findText above is one: a
        # panel that reads a choice by position -- which is what a choice
        # whose value Qt cannot match for us has to do -- would otherwise be
        # comparing an _Any against a length.
        return self._index

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
        # Real enough to test with: a change made on one of several selected
        # rows is supposed to land on all of them, and an _Any selection
        # could not show that either way.
        self._selected = set()
        self.selection_mode = None
        self.selection_behavior = None
        self.setRowCount(rows)

    def setSelectionMode(self, mode):
        self.selection_mode = mode

    def setSelectionBehavior(self, behavior):
        self.selection_behavior = behavior

    def selectRow(self, row):
        """Select a row, if Qt would have allowed it.

        Honouring the flag is the whole point: selecting a row the real
        widget would refuse is how a stub reports that a feature works when
        it cannot work at all.
        """
        row = int(row)
        item = self._items.get((row, 0))
        flags = None if item is None else item.flags()
        if flags is not None and not (int(flags) & ITEM_IS_SELECTABLE):
            return
        self._selected.add(row)

    def clearSelection(self):
        self._selected.clear()

    def selectedItems(self):
        """Column 0's item for each selected row, as SelectRows gives."""
        items = []
        for row in sorted(self._selected):
            if row >= self._rows:
                continue
            item = self._items.get((row, 0))
            if item is not None:
                items.append(item)
        return items

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
        item._row, item._column = row, column
        item.table = self
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
        # Filled in by QTableWidget.setItem. Qt's items know where they sit,
        # and the dialog's itemChanged handler asks them.
        self._row = -1
        self._column = -1
        self.table = None

    def row(self):
        return self._row

    def column(self):
        return self._column

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


class QFont:
    """Real enough to check that a size was actually asked for.

    The point of the type scale is that sizes are stated rather than read back
    off a widget, so a stub whose pointSizeF() returned a dummy would not be
    able to tell the difference.
    """

    class SpacingType:
        AbsoluteSpacing = "absolute"
        PercentageSpacing = "percentage"

    def __init__(self, other=None):
        self._point_size = getattr(other, "_point_size", -1.0)
        self._bold = getattr(other, "_bold", False)
        self._spacing = None

    def setPointSizeF(self, size):
        self._point_size = float(size)

    def setPointSize(self, size):
        self._point_size = float(size)

    def pointSizeF(self):
        return self._point_size

    def pointSize(self):
        return int(self._point_size)

    def setBold(self, bold):
        self._bold = bool(bold)

    def bold(self):
        return self._bold

    def setLetterSpacing(self, kind, spacing):
        self._spacing = (kind, spacing)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


class QListWidgetItem:
    """A row of text with a check state, some data and a tooltip.

    Unhashable, deliberately, because the real one is: Qt gives
    QListWidgetItem comparison operators and no __hash__, so PySide6's wrapper
    cannot go in a set or a dict. A stub that allowed it reported a panel as
    working while set(widget.selectedItems()) raised TypeError on the user's
    machine -- which is how the PARAMS move buttons shipped doing nothing.

    setText raises itemChanged in Qt exactly as a click on the checkbox does,
    which is the trap the ISO panel's badges have to avoid; the stub emits it
    for the same reason.
    """

    # See the class docstring: not hashable, like the real one.
    __hash__ = None

    def __init__(self, text=""):
        self._text = text
        self._tooltip = ""
        self._flags = 0
        self._check = None
        self._data = {}
        self.list = None
        # See QTreeWidgetItem.check_writes: the in-place refresh path writes a
        # box only when it is wrong, and this is how that is asserted.
        self.check_writes = 0

    def _changed(self):
        if self.list is not None:
            self.list.itemChanged.emit(self)

    def setText(self, text):
        self._text = text
        self._changed()

    def text(self):
        return self._text

    def setToolTip(self, text):
        self._tooltip = text

    def toolTip(self):
        return self._tooltip

    def setFlags(self, flags):
        self._flags = flags

    def flags(self):
        return self._flags

    def setHidden(self, hidden):
        self._hidden = bool(hidden)

    def isHidden(self):
        return getattr(self, "_hidden", False)

    def setCheckState(self, state):
        self.check_writes += 1
        self._check = state
        self._changed()

    def checkState(self):
        return self._check

    def setData(self, role, value):
        self._data[role] = value

    def data(self, role):
        return self._data.get(role)


class _Index:
    """A QModelIndex, as far as anything here reads one."""

    def __init__(self, row):
        self._row = row

    def row(self):
        return self._row


class _SelectionModel:
    """What a view's selectionModel() answers: the selection, in rows.

    Rows rather than items, which is the honest shape: Qt's selection lives
    in the model and an item is only a convenience wrapper over a row.
    """

    def __init__(self, widget):
        self._widget = widget

    def selectedIndexes(self):
        return [_Index(row) for row in self._widget.selected_rows()]


class QListWidget:
    def __init__(self, *args, **kwargs):
        self._items = []
        self._current = -1
        self._signals_blocked = 0
        self.updates_enabled = True
        self.itemChanged = _BoundSignal()
        self.currentRowChanged = _BoundSignal()

    def row(self, item):
        """Where an item sits, or -1. The real one resolves this in C++, so
        it works whichever Python wrapper it is handed."""
        for index, candidate in enumerate(self._items):
            if candidate is item:
                return index
        return -1

    def selected_rows(self):
        """The rows selectedItems() names. Tests select by replacing
        selectedItems; this keeps the two answers consistent."""
        rows = []
        for item in (self.selectedItems() or []):
            row = self.row(item)
            if row >= 0:
                rows.append(row)
        return rows

    def selectionModel(self):
        return _SelectionModel(self)

    def selectedItems(self):
        return []

    def addItem(self, item):
        if not isinstance(item, QListWidgetItem):
            item = QListWidgetItem(str(item))
        item.list = self
        self._items.append(item)
        return item

    def takeItem(self, row):
        if 0 <= row < len(self._items):
            item = self._items.pop(row)
            item.list = None
            return item
        return None

    def blockSignals(self, blocked):
        previous = self._signals_blocked > 0
        self._signals_blocked += 1 if blocked else -1
        if self._signals_blocked < 0:
            self._signals_blocked = 0
        return previous

    def setUpdatesEnabled(self, enabled):
        self.updates_enabled = bool(enabled)

    def clear(self):
        for item in self._items:
            item.list = None
        self._items = []
        self._current = -1

    def count(self):
        return len(self._items)

    def item(self, row):
        return self._items[row] if 0 <= row < len(self._items) else None

    def currentRow(self):
        return self._current

    def setCurrentRow(self, row):
        self._current = row
        self.currentRowChanged.emit(row)

    def currentItem(self):
        return self.item(self._current)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


class QTextBrowser:
    """Records the rich text it was given, which is what the tests read back."""

    def __init__(self, *args, **kwargs):
        self.html = ""

    def setHtml(self, html):
        self.html = html

    def toHtml(self):
        return self.html

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


class QTreeWidgetItem:
    """A node with per-column text, a check state, data and children.

    Constructed the way Qt allows it: `QTreeWidgetItem(parent, ["text"])`, where
    the parent is either the tree (a top-level row) or another item. That is how
    the ISO panel builds its parameter nodes, and getting the two cases right is
    the point of having this at all.
    """

    def __init__(self, parent=None, texts=None):
        self._texts = list(texts or [])
        self._tooltips = {}
        self._flags = 0
        self._checks = {}
        self._foregrounds = {}
        # How many times a box on this row has been written. The in-place
        # refresh path only writes a box that is actually wrong, and "wrote
        # nothing" is the assertion that keeps it that way.
        self.check_writes = 0
        self._data = {}
        self._children = []
        self._parent = None
        self.tree = None
        self._expanded = False

        if isinstance(parent, QTreeWidget):
            parent.addTopLevelItem(self)
        elif isinstance(parent, QTreeWidgetItem):
            parent.addChild(self)

    # --- structure
    def addChild(self, child):
        child._parent = self
        child.tree = self.tree
        self._children.append(child)

    def setHidden(self, hidden):
        self._hidden = bool(hidden)

    def isHidden(self):
        return bool(getattr(self, "_hidden", False))

    def child(self, index):
        return self._children[index] if 0 <= index < len(self._children) else None

    def childCount(self):
        return len(self._children)

    def parent(self):
        return self._parent

    def setExpanded(self, expanded):
        self._expanded = bool(expanded)
        if self.tree is not None:
            signal = (self.tree.itemExpanded if self._expanded
                      else self.tree.itemCollapsed)
            signal.emit(self)

    def isExpanded(self):
        return self._expanded

    # --- content
    def _changed(self):
        if self.tree is not None:
            self.tree.itemChanged.emit(self, 0)

    def setText(self, column, text):
        while len(self._texts) <= column:
            self._texts.append("")
        self._texts[column] = text
        self._changed()

    def text(self, column=0):
        return self._texts[column] if column < len(self._texts) else ""

    def setToolTip(self, column, text):
        self._tooltips[column] = text

    def toolTip(self, column=0):
        return self._tooltips.get(column, "")

    def setForeground(self, column, brush):
        self._foregrounds[column] = brush

    def foreground(self, column=0):
        return self._foregrounds.get(column)

    def setFlags(self, flags):
        self._flags = flags

    def flags(self):
        return self._flags

    def setCheckState(self, column, state):
        self.check_writes += 1
        self._checks[column] = state
        self._changed()

    def checkState(self, column=0):
        return self._checks.get(column)

    def setData(self, column, role, value):
        self._data[(column, role)] = value

    def data(self, column, role):
        return self._data.get((column, role))


class QTreeWidget:
    # Qt's own, named because the report sets its rows selectable whole.
    class SelectionBehavior:
        SelectItems = 0
        SelectRows = 1

    def __init__(self, *args, **kwargs):
        self._items = []
        self._current = None
        self._widgets = {}
        self.itemChanged = _BoundSignal()
        self.itemExpanded = _BoundSignal()
        self.itemCollapsed = _BoundSignal()
        self.itemDoubleClicked = _BoundSignal()
        self.itemClicked = _BoundSignal()
        self.customContextMenuRequested = _BoundSignal()
        self.updates_enabled = True
        self.signals_blocked = False

    def addTopLevelItem(self, item):
        item.tree = self
        item._parent = None
        for child in item._children:
            child.tree = self
        self._items.append(item)

    def topLevelItem(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def topLevelItemCount(self):
        return len(self._items)

    def clear(self):
        for item in self._items:
            item.tree = None
        self._items = []
        self._current = None
        self._widgets = {}

    # A real cell widget, because what a report row *reads* as lives in the
    # widget and not in the item's text: a coloured group name is a label, and
    # a per-line action is a button. An assertion against the item's text
    # would pass while the window showed nothing.
    def setItemWidget(self, item, column, widget):
        self._widgets[(id(item), column)] = widget

    def itemWidget(self, item, column):
        return self._widgets.get((id(item), column))

    def double_click(self, item, column=0):
        """What Qt does when a row is double-clicked."""
        self._current = item
        self.itemDoubleClicked.emit(item, column)

    def setCurrentItem(self, item):
        self._current = item

    def currentItem(self):
        return self._current

    def setUpdatesEnabled(self, enabled):
        self.updates_enabled = bool(enabled)

    def blockSignals(self, blocked):
        was = self.signals_blocked
        self.signals_blocked = bool(blocked)
        self.itemChanged.blocked = self.signals_blocked
        self.itemExpanded.blocked = self.signals_blocked
        self.itemCollapsed.blocked = self.signals_blocked
        return was

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


class QLineEdit:
    """Text and the signal that goes with it. The panels filter through these."""

    def __init__(self, text="", *args, **kwargs):
        self._text = text
        # Whether Qt's own clear button is on. Recorded, because "every text
        # filter has one" is the behaviour being asked for and an invented
        # answer would be truthy whether or not anyone had asked for it.
        self.clear_button = False
        self.textChanged = _BoundSignal()
        self.textEdited = _BoundSignal()

    def setClearButtonEnabled(self, enabled):
        self.clear_button = bool(enabled)

    def setText(self, text):
        self._text = text
        self.textChanged.emit(text)

    def typed(self, text):
        """What a keystroke does: both signals, textEdited first."""
        self._text = text
        self.textEdited.emit(text)
        self.textChanged.emit(text)

    def text(self):
        return self._text

    def cursorPosition(self):
        return len(self._text)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


class QPushButton:
    """Text, tooltip and a clicked signal.

    Most buttons in the panels are fire-and-forget, which an auto-made stub
    serves well enough. This exists for the few whose *label* is part of the
    behaviour -- DATA retitles Plot all when a filter narrows its scope -- and
    a test that reads a label back needs something that remembers one.
    """

    def __init__(self, text="", *args, **kwargs):
        self._text = text
        self._tip = ""
        self.focus_policy = None
        self.clicked = _BoundSignal()

    def setFocusPolicy(self, policy):
        # See QComboBox.setFocusPolicy above.
        self.focus_policy = policy

    def setText(self, text):
        self._text = text

    def text(self):
        return self._text

    def setToolTip(self, tip):
        self._tip = tip

    def toolTip(self):
        return self._tip

    # Real, because "the button is dead until there is something to press it
    # for" is a thing the panels decide and a thing worth asserting. The
    # placeholder answered every isEnabled() with something truthy.
    def setEnabled(self, enabled):
        self._enabled = bool(enabled)

    def isEnabled(self):
        return getattr(self, "_enabled", True)

    def setVisible(self, visible):
        self._visible = bool(visible)

    def isVisible(self):
        return getattr(self, "_visible", True)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


class QLabel:
    """Records the text it was set to; the panels report through these."""

    def __init__(self, text="", *args, **kwargs):
        self._text = text
        self._visible = True
        self.word_wrap = False

    def setText(self, text):
        self._text = text

    def text(self):
        return self._text

    def setWordWrap(self, wrap):
        self.word_wrap = bool(wrap)

    # A hidden label is how a panel says "there is nothing to report here",
    # and an assertion on its text cannot tell the difference.
    def setVisible(self, visible):
        self._visible = bool(visible)

    def isVisible(self):
        return bool(self._visible)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


class QSpinBox:
    def __init__(self, *args, **kwargs):
        self._value = 0
        self._min, self._max = 0, 99
        self.valueChanged = _BoundSignal()

    def setRange(self, low, high):
        self._min, self._max = int(low), int(high)

    def minimum(self):
        return self._min

    def maximum(self):
        return self._max

    def setValue(self, value):
        value = max(self._min, min(self._max, int(value)))
        if value != self._value:
            self._value = value
            self.valueChanged.emit(value)

    def value(self):
        return self._value

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


class _Enum:
    """A namespace of distinct integers, which is what Qt's enums are here.

    Item flags are combined with `|` and check states are compared for
    equality, so the one thing the stubs cannot do is hand back an opaque
    object: `flags() | Qt.ItemFlag.ItemIsUserCheckable` has to be arithmetic.
    """

    def __init__(self, **names):
        for index, (name, value) in enumerate(sorted(names.items())):
            setattr(self, name, value)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        # Anything not named above is a flag nobody in the code inspects; a
        # distinct bit keeps `|` working and comparisons meaningful.
        value = 1 << (16 + len(self.__dict__))
        object.__setattr__(self, name, value)
        return value


# Qt's item flags are a bitmask, and the one that decides whether a row can
# be picked at all is ItemIsSelectable. setFlags *replaces* the default set
# rather than adding to it, so an item given only ItemIsEnabled is not
# selectable -- which is worth modelling, because a stub that let every row
# be selected reported a multi-row feature as working when the real widget
# would not allow a single row to be chosen.
ITEM_IS_ENABLED = 1
ITEM_IS_SELECTABLE = 2


def _make_qt():
    qt = _Any()
    qt.ItemFlag = _Enum(NoItemFlags=0, ItemIsEnabled=ITEM_IS_ENABLED,
                        ItemIsSelectable=ITEM_IS_SELECTABLE,
                        ItemIsEditable=4, ItemIsUserCheckable=8)
    qt.CheckState = _Enum(Unchecked=0, PartiallyChecked=1, Checked=2)
    # Real values, because whether a cell widget takes focus when clicked
    # decides whether a multi-row selection survives being acted on.
    qt.FocusPolicy = _Enum(NoFocus=0, TabFocus=1, ClickFocus=2,
                           StrongFocus=11, WheelFocus=15)
    qt.ItemDataRole = _Enum(DisplayRole=0, UserRole=256, ToolTipRole=3)
    qt.TextFormat = _Enum(PlainText=0, RichText=1, AutoText=2)
    # The modifier bits are Qt's own, because the tab shortcuts combine them
    # with `|` at class-definition time and test a key event against the mask.
    qt.KeyboardModifier = _Enum(NoModifier=0,
                                ShiftModifier=0x02000000,
                                ControlModifier=0x04000000,
                                AltModifier=0x08000000,
                                MetaModifier=0x10000000)
    # Distinct, and that is all the rail asks of them.
    qt.Key = _Enum(Key_Up=0x01000013, Key_Down=0x01000015,
                   Key_Left=0x01000012, Key_Right=0x01000014,
                   Key_Home=0x01000010, Key_End=0x01000011,
                   Key_Escape=0x01000000, Key_Return=0x01000004)
    qt.WidgetAttribute = _Enum(WA_InputMethodEnabled=1,
                               WA_TranslucentBackground=2)
    return qt


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


class QPointF:
    """A point that holds its coordinates.

    An _Any placeholder was enough while points were only ever handed back to
    Qt. The zoom decides which axes a drag means from the coordinates, so they
    have to be real numbers for any of that to be reachable from a test.
    """

    def __init__(self, x=0.0, y=0.0):
        self._x = float(x)
        self._y = float(y)

    def x(self):
        return self._x

    def y(self):
        return self._y

    def __eq__(self, other):
        return (isinstance(other, QPointF)
                and (self._x, self._y) == (other._x, other._y))

    def __repr__(self):
        return "QPointF({0}, {1})".format(self._x, self._y)


class QRectF:
    """A rectangle that holds its edges. Takes (x, y, w, h) or two corners."""

    def __init__(self, *args):
        if len(args) == 4:
            x, y, w, h = (float(a) for a in args)
        elif len(args) == 2 and all(isinstance(a, QPointF) for a in args):
            first, second = args
            x, y = min(first.x(), second.x()), min(first.y(), second.y())
            w, h = abs(second.x() - first.x()), abs(second.y() - first.y())
        else:
            x = y = w = h = 0.0
        self._x, self._y, self._w, self._h = x, y, w, h

    def left(self):
        return self._x

    def top(self):
        return self._y

    def right(self):
        return self._x + self._w

    def bottom(self):
        return self._y + self._h

    def width(self):
        return self._w

    def height(self):
        return self._h

    def topLeft(self):
        return QPointF(self._x, self._y)

    def normalized(self):
        return self

    def __repr__(self):
        return "QRectF({0}, {1}, {2}, {3})".format(
            self._x, self._y, self._w, self._h)


class QFontMetricsF:
    """Font metrics with a fixed, plausible answer.

    A real measurement needs a font engine and a display. What the tile does
    with the number is the part worth testing, so the stub returns a constant
    and the tests assert on the arithmetic applied to it.
    """

    LINE_HEIGHT = 14.0

    def __init__(self, *args, **kwargs):
        pass

    def height(self):
        return self.LINE_HEIGHT

    def horizontalAdvance(self, text=""):
        return 7.0 * len(str(text))

    def ascent(self):
        return self.LINE_HEIGHT * 0.8

    def descent(self):
        return self.LINE_HEIGHT * 0.2


class _LayoutItem:
    """What Qt's layouts hand back: an item wrapping a widget."""

    def __init__(self, widget):
        self._widget = widget

    def widget(self):
        return self._widget


class QBoxLayout(_Any):
    """A box layout that really keeps its items, in order.

    Everything else about a layout is invented on demand like any other stub,
    but membership is not: the tab rail rearranges itself by taking every
    button out and putting them back in a new order, so a count() that never
    reaches zero is an endless loop rather than a wrong answer.
    """

    def __init__(self, *args, **kwargs):
        _Any.__init__(self, *args, **kwargs)
        self.items = []

    def addWidget(self, widget, *args, **kwargs):
        self.items.append(widget)

    def insertWidget(self, index, widget, *args, **kwargs):
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = len(self.items)
        self.items.insert(max(0, index), widget)

    def addStretch(self, *args, **kwargs):
        # A stretch is a spacer, not a widget: it occupies a place and has no
        # widget of its own, which is exactly what None models here.
        self.items.append(None)

    def addSpacing(self, *args, **kwargs):
        self.items.append(None)

    def addLayout(self, layout, *args, **kwargs):
        self.items.append(layout)

    def count(self):
        return len(self.items)

    def itemAt(self, index):
        if 0 <= index < len(self.items):
            return _LayoutItem(self.items[index])
        return None

    def takeAt(self, index):
        if 0 <= index < len(self.items):
            return _LayoutItem(self.items.pop(index))
        return None

    def widgets(self):
        """The widgets in it, in order. For tests to read the arrangement."""
        return [item for item in self.items if item is not None]


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
    core.Qt = _make_qt()
    for name in ("QObject", "QThread"):
        setattr(core, name, type(name, (_Any,), {}))

    # Qt's own numbers, and on the class: the auto-module's placeholder only
    # answers lookups on an instance, so QEvent.Type raised where the real one
    # does not -- and the tab shortcuts compare a key press against it.
    class QEvent(_Any):
        Type = _Enum(MouseButtonPress=2, MouseButtonRelease=3,
                     MouseMove=5, KeyPress=6, KeyRelease=7,
                     FocusIn=8, FocusOut=9, Enter=10, Leave=11,
                     Wheel=31, ShortcutOverride=51)

    core.QEvent = QEvent
    # These two carry numbers rather than inventing attributes: the zoom reads
    # a drag's coordinates to decide which axes it means.
    core.QPointF = QPointF
    core.QRectF = QRectF

    # QTimer needs two real behaviours.
    #
    # singleShot must actually call back: code defers work to the next turn of
    # the event loop when doing it now would destroy a widget Qt is still
    # using, and a no-op would quietly test the half of that path that does
    # nothing. Running it straight away is the closest a stub gets to "very
    # soon".
    #
    # An *instance* must record that it was started without firing. The timers
    # built this way are debouncers -- the caption commit, the resample at the
    # end of a drag, the parameter filter -- and what they are for is that the
    # work does NOT happen yet. Firing on start would test the opposite of the
    # behaviour, so a test asks for it explicitly with fire().
    class QTimer(_Any):
        timeout = Signal()

        def __init__(self, *args, **kwargs):
            _Any.__init__(self, *args, **kwargs)
            self.pending = False
            self.single_shot = False
            self.interval_ms = 0

        @staticmethod
        def singleShot(_msec, callback):
            callback()

        def setSingleShot(self, value):
            self.single_shot = bool(value)

        def setInterval(self, msec):
            self.interval_ms = int(msec)

        def start(self, msec=None):
            if msec is not None:
                self.interval_ms = int(msec)
            self.pending = True

        def stop(self):
            self.pending = False

        def isActive(self):
            return self.pending

        def fire(self):
            """What the event loop would do when the interval elapses."""
            if not self.pending:
                return False
            self.pending = False
            self.timeout.emit()
            return True

    core.QTimer = QTimer

    gui = _AutoModule("PySide6.QtGui")
    # Real enough to measure with: the tile asks the font how tall a row of
    # tick numbers is, and an _Any answer would make that arithmetic
    # unreachable from a test.
    gui.QFontMetricsF = QFontMetricsF
    gui.QFontMetrics = QFontMetricsF
    for name in ("QBrush", "QColor", "QPainter", "QPen", "QPolygonF"):
        setattr(gui, name, type(name, (_Any,), {}))
    gui.QFont = QFont

    widgets = _AutoModule("PySide6.QtWidgets")

    # Qt's scoped enums live on the class, and the auto-module's placeholder
    # only answers attribute lookups on an *instance* -- so QSizePolicy.Policy
    # raised where the real one does not. Named here because the tab rail sets
    # its buttons Fixed in both directions, and the rail is worth building in
    # a test: what is in that column is what a short screen is short of.
    class _Policy:
        Fixed = "fixed"
        Expanding = "expanding"
        Preferred = "preferred"
        Minimum = "minimum"
        Ignored = "ignored"

    class QSizePolicy(_Any):
        Policy = _Policy

    # Scoped enums on the class, which the auto-module's placeholder cannot
    # answer: the panels name them while building their lists.
    class _AbstractItemView(_Any):
        class SelectionMode:
            NoSelection = 0
            SingleSelection = 1
            ExtendedSelection = 3

        class SelectionBehavior:
            SelectItems = 0
            SelectRows = 1

        class EditTrigger:
            NoEditTriggers = 0

    class QDialog(QWidget):
        """Enough of a dialog to drive one: a title, and the two ways out.

        Shown and closed rather than run: every dialog in this program is
        non-modal, and a stub that blocked in exec() would hang the suite.
        """

        def __init__(self, parent=None, *args, **kwargs):
            QWidget.__init__(self, parent)
            self.accepted = _BoundSignal()
            self.rejected = _BoundSignal()
            self.finished = _BoundSignal()
            self.result_code = None
            self.title = ""
            self.modal = False
            self.shown = False

        def setWindowTitle(self, title):
            self.title = title

        def windowTitle(self):
            return self.title

        def setModal(self, modal):
            self.modal = bool(modal)

        def isModal(self):
            return self.modal

        def show(self):
            self.shown = True

        def raise_(self):
            pass

        def activateWindow(self):
            pass

        def accept(self):
            self.result_code = 1
            self.shown = False
            self.accepted.emit()
            self.finished.emit(1)

        def reject(self):
            self.result_code = 0
            self.shown = False
            self.rejected.emit()
            self.finished.emit(0)

    widgets.QDialog = QDialog
    # A scoped enum on the class, which the auto-module's placeholder cannot
    # answer: the report names it to give its run column the window's width.
    class QHeaderView(_Any):
        class ResizeMode:
            Interactive = 0
            Stretch = 1
            Fixed = 2
            ResizeToContents = 3

    widgets.QHeaderView = QHeaderView
    widgets.QSizePolicy = QSizePolicy
    widgets.QApplication = QApplication
    widgets.QAbstractItemView = _AbstractItemView
    widgets.QAbstractButton = QAbstractButton
    widgets.QBoxLayout = QBoxLayout
    widgets.QVBoxLayout = QBoxLayout
    widgets.QHBoxLayout = QBoxLayout
    widgets.QWidget = QWidget
    widgets.QComboBox = QComboBox
    widgets.QTableWidget = QTableWidget
    widgets.QTableWidgetItem = QTableWidgetItem
    widgets.QListWidget = QListWidget
    widgets.QListWidgetItem = QListWidgetItem
    widgets.QTextBrowser = QTextBrowser
    widgets.QTreeWidget = QTreeWidget
    widgets.QTreeWidgetItem = QTreeWidgetItem
    widgets.QLabel = QLabel
    widgets.QLineEdit = QLineEdit
    widgets.QPushButton = QPushButton
    widgets.QSpinBox = QSpinBox

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
    # ViewBox's mode constants are read off the *class*, and _Any only answers
    # attribute lookups on instances -- so an auto-made stub raised where the
    # real one just has two integers.
    pg.ViewBox = ViewBox
    pg.GraphicsObject = type("GraphicsObject", (_Any,), {})
    pg.TextItem = type("TextItem", (_Any,), {})
    # Draggable for real, because whether a drag reaches the model is the
    # difference between an annotation that moves on every tile and one that
    # moves only under the cursor. The signals are recorders, so a test can
    # do what pyqtgraph does: move the item, then announce it.

    class InfiniteLine(_Any):
        sigPositionChangeFinished = Signal(object)

        def __init__(self, *args, **kwargs):
            _Any.__init__(self, *args, **kwargs)
            self.movable = bool(kwargs.get("movable", False))
            self._value = float(kwargs.get("pos", 0.0) or 0.0)

        def setMovable(self, movable):
            self.movable = bool(movable)

        def value(self):
            return self._value

        def drag_to(self, value):
            """What pyqtgraph does when the user lets go of the line."""
            self._value = float(value)
            self.sigPositionChangeFinished.emit(self)

    class LinearRegionItem(_Any):
        sigRegionChangeFinished = Signal(object)

        def __init__(self, *args, **kwargs):
            _Any.__init__(self, *args, **kwargs)
            self.movable = bool(kwargs.get("movable", False))
            self._region = tuple(kwargs.get("values", (0.0, 1.0)))

        def setMovable(self, movable):
            self.movable = bool(movable)

        def getRegion(self):
            return self._region

        def drag_to(self, low, high):
            self._region = (float(low), float(high))
            self.sigRegionChangeFinished.emit(self)

    pg.InfiniteLine = InfiniteLine
    pg.LinearRegionItem = LinearRegionItem
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
