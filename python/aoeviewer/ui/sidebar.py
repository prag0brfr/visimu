# Author: Paulo R. A. Gilz
"""Narrow vertical tab rail down the left edge.

Qt has no vertical-text tab widget, and QTabBar's West position rotates the
*whole* tab including its frame, which looks wrong against a flat panel. So the
buttons paint their own rotated text:
a ~26 px column with the label reading bottom-to-top, which costs almost no
horizontal space no matter how many tabs there are.

It holds tabs and nothing else. Anything that is not a panel selector belongs
in the toolbar: the column's height is the one thing a short screen is short
of, and a button that selects no panel is a button in the wrong place.

Four ways to reach a panel, because twelve of them down a narrow column is a
lot of aiming: click it, press its letter, scroll over the rail, or walk it
with the arrow keys. And one way to rearrange them: drag a name up or down.

Order is by *position*, identity is by *panel index*. The two are kept apart
deliberately: everything outside this module -- the panel stack, the window's
tab handler, the workspace -- refers to a panel by the index it was built
with, so dragging the rail about cannot make the window show the wrong panel.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import QEvent, QObject, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..theme import RAIL_FONT_PT, Theme, px, pt

# The rail is pure chrome down the left edge, so it gives up space first: it
# has to be wide enough to read a rotated word in and no wider. TEXT_PADDING is
# the slack around that word, and it was set when the text was larger.
RAIL_WIDTH = 21
MIN_BUTTON_HEIGHT = 34
TEXT_PADDING = 10

# How far a name has to be dragged before it changes places, as a fraction of
# its own length. Half means the swap happens when the pointer has reached
# where the neighbour's middle was, which is where the eye expects it.
DRAG_STEP_FRACTION = 0.5

# One letter per panel, pressed on its own. Unique by construction -- the
# self-test asserts it -- and chosen from the label wherever the obvious
# letter was still free: ISO gives up I to INSPECT, which is reached far more
# often, and takes its O; SCALE takes its A, since AREA has R.
#
# A letter and not Ctrl+something because these are navigation, not commands:
# twelve more Ctrl bindings would collide with the window's own, and the
# filter below already makes sure a bare letter never reaches a tab while
# anything is being typed into.
SHORTCUTS: Dict[str, str] = {
    "DATA": "D",
    "PARAMS": "P",
    "COMPUTE": "C",
    "DIFF": "F",
    "ISO": "O",
    "STYLE": "S",
    "SCALE": "A",
    "SYNC": "Y",
    "3D": "3",
    "INSPECT": "I",
    "AREA": "R",
    "MARKERS": "M",
    "TEXT": "T",
}


def shortcut_for(label: str) -> str:
    """The letter that selects `label`, or "" when it has none."""
    return SHORTCUTS.get(label, "")


class VerticalTabButton(QAbstractButton):
    """A checkable button whose label is drawn rotated 90 degrees."""

    def __init__(self, text: str, theme: Theme, parent: Optional[QWidget] = None,
                 panel_index: int = 0, shortcut: str = ""):
        super().__init__(parent)
        self._theme = theme
        self.panel_index = panel_index
        self.shortcut_key = shortcut
        # Set by the rail. Called with (self, +1 or -1) while the name is being
        # dragged; returns True when the move happened.
        self.drag_step = None
        self._press_y = None
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        # The rail takes the focus, not its buttons: the arrow keys are the
        # rail's, and a button holding the focus would eat them one at a time.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.refresh_tip()

        self.refresh_metrics()

    def refresh_tip(self) -> None:
        """The tooltip, which is where the shortcut is advertised.

        On hover, because a letter painted into a 21 px column would have to
        come out of the label's own space, and the label is the thing being
        aimed at.
        """
        name = self.text().title()
        if self.shortcut_key:
            self.setToolTip("{0}  --  press {1}, or drag to reorder"
                            .format(name, self.shortcut_key))
        else:
            self.setToolTip(name)

    def refresh_metrics(self) -> None:
        """Re-read the font from the UI scale and re-measure.

        The size is set outright rather than derived from self.font(): the
        application stylesheet sizes text in pixels, which makes pointSizeF()
        report -1, so "a little smaller than the body text" silently became
        "whatever the clamp says". Called again whenever the scale moves,
        because the rail's width is the height of its text and a rail that did
        not follow would clip every label.
        """
        font = QFont(self.font())
        font.setPointSizeF(pt(RAIL_FONT_PT))
        font.setBold(True)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.6)
        self.setFont(font)
        self.updateGeometry()

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.update()

    def sizeHint(self) -> QSize:
        metrics = self.fontMetrics()
        # The button's height is the text's *width*, because the text is turned
        # on its side.
        height = max(px(MIN_BUTTON_HEIGHT),
                     metrics.horizontalAdvance(self.text()) + px(TEXT_PADDING))
        return QSize(px(RAIL_WIDTH), height)

    # ------------------------------------------------------------ dragging

    @staticmethod
    def _local_y(event) -> float:
        """The pointer's y inside this button, across Qt versions."""
        try:
            return float(event.position().y())
        except AttributeError:
            return float(event.y())

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_y = self._local_y(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton and callable(self.drag_step):
            if self._press_y is None:
                # Re-seeded after a move: the button has just travelled a whole
                # place under the pointer, so where the drag started from is no
                # longer where it was measured.
                self._press_y = self._local_y(event)
            else:
                travel = self._local_y(event) - self._press_y
                threshold = max(1.0, self.height() * DRAG_STEP_FRACTION)
                if abs(travel) >= threshold:
                    if self.drag_step(self, 1 if travel > 0 else -1):
                        self._press_y = None
                        return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._press_y = None
        super().mouseReleaseEvent(event)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        theme = self._theme
        rect = self.rect()

        if self.isChecked():
            background = QColor(theme.accent)
            foreground = QColor(theme.accent_text)
        elif self.underMouse():
            background = QColor(theme.surface)
            foreground = QColor(theme.text)
        else:
            background = QColor(theme.panel)
            foreground = QColor(theme.text_dim)

        painter.fillRect(rect, background)

        # A thin accent stripe on the inner edge marks the selected tab even
        # when the theme's accent and panel colours are close in value.
        if self.isChecked():
            painter.fillRect(rect.width() - 2, 0, 2, rect.height(), QColor(theme.accent))

        painter.save()
        painter.translate(rect.width() / 2.0, rect.height() / 2.0)
        painter.rotate(-90.0)
        painter.setPen(QPen(foreground))
        metrics = painter.fontMetrics()
        text = self.text()
        painter.drawText(
            -metrics.horizontalAdvance(text) / 2.0,
            metrics.height() / 2.0 - metrics.descent(),
            text,
        )
        painter.restore()

    def enterEvent(self, event) -> None:
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.update()
        super().leaveEvent(event)


class TabRail(QWidget):
    """The column of vertical tab buttons. Emits the index that was clicked."""

    tab_selected = Signal(int)
    order_changed = Signal()

    def __init__(self, labels: List[str], theme: Theme, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._theme = theme
        self._labels = list(labels)
        self._buttons: List[VerticalTabButton] = []
        # Panel indices, in the order they are shown. Rearranged by dragging;
        # _buttons is never reordered, so a button's place in that list stays
        # its panel index for the life of the rail.
        self._order: List[int] = list(range(len(labels)))

        self.setObjectName("Panel")
        self.setFixedWidth(px(RAIL_WIDTH))
        self.setAutoFillBackground(True)
        # Clicking the rail gives it the focus, which is what makes the arrow
        # keys work: they move between tabs here and must keep meaning what
        # they always meant inside a panel's own lists and boxes.
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 4, 0, 4)
        self._layout.setSpacing(1)

        for index, label in enumerate(labels):
            button = VerticalTabButton(label, theme, self, index,
                                       shortcut_for(label))
            button.clicked.connect(lambda _checked=False, i=index: self._select(i))
            button.drag_step = self._drag_step
            self._buttons.append(button)

        self._relayout()

        # Nothing below the tabs. The light/dark toggle used to be pinned to
        # the foot of this column, where it was one more thing competing for
        # the vertical space the tabs themselves need on a short screen -- and
        # it is not a tab: it selects no panel. It is in the toolbar now,
        # beside the interface-size buttons, with the rest of what changes how
        # the window looks rather than what it shows.

        if self._buttons:
            self._buttons[0].setChecked(True)

    # -------------------------------------------------------------- layout

    def _relayout(self) -> None:
        """Put the buttons in the layout in display order.

        The stretch goes in last and is taken out first, so the names stay at
        the top of the column however they are rearranged.
        """
        while self._layout.count():
            self._layout.takeAt(0)
        for panel_index in self._order:
            self._layout.addWidget(self._buttons[panel_index])
        self._layout.addStretch(1)

    # --------------------------------------------------------------- order

    def order(self) -> List[str]:
        """The labels, top to bottom. What the workspace stores."""
        return [self._labels[i] for i in self._order]

    def order_indices(self) -> List[int]:
        return list(self._order)

    def set_order(self, labels: List[str]) -> None:
        """Arrange the rail to match `labels`, as far as it can.

        Anything it does not recognise is ignored and anything it was not told
        about keeps its place at the end, so a workspace written by another
        version -- with a panel this one has dropped, or without one it has
        gained -- still arranges the tabs it has in common.
        """
        wanted: List[int] = []
        for label in labels or []:
            if not isinstance(label, str):
                continue
            if label in self._labels:
                index = self._labels.index(label)
                if index not in wanted:
                    wanted.append(index)
        for index in range(len(self._labels)):
            if index not in wanted:
                wanted.append(index)
        if wanted == self._order:
            return
        self._order = wanted
        self._relayout()

    def move(self, from_position: int, to_position: int) -> bool:
        """Move the name at one position to another. True if anything moved."""
        count = len(self._order)
        if not (0 <= from_position < count) or not (0 <= to_position < count):
            return False
        if from_position == to_position:
            return False
        order = list(self._order)
        order.insert(to_position, order.pop(from_position))
        self._order = order
        self._relayout()
        self.order_changed.emit()
        return True

    def _drag_step(self, button: VerticalTabButton, step: int) -> bool:
        try:
            position = self._order.index(button.panel_index)
        except ValueError:
            return False
        return self.move(position, position + step)

    # ---------------------------------------------------------- navigation

    def step(self, delta: int) -> None:
        """Select the tab `delta` places from the current one.

        Clamped rather than wrapped: a wheel that rolls off the end and back
        to the top loses the user's place in a column of twelve.
        """
        if not self._order:
            return
        try:
            position = self._order.index(self.current())
        except ValueError:
            position = 0
        target = min(max(position + int(delta), 0), len(self._order) - 1)
        if target == position:
            return
        self._select(self._order[target])

    def wheelEvent(self, event) -> None:
        """Scrolling over the names walks the tabs.

        Only over the names: this handler belongs to the rail, and the panel
        itself is a different widget in a scroll area, where the wheel goes on
        scrolling the panel.
        """
        delta = 0
        try:
            delta = event.angleDelta().y()
        except AttributeError:
            pass
        if delta:
            self.step(-1 if delta > 0 else 1)
            event.accept()
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Left):
            self.step(-1)
            event.accept()
            return
        if key in (Qt.Key.Key_Down, Qt.Key.Key_Right):
            self.step(1)
            event.accept()
            return
        if key == Qt.Key.Key_Home:
            if self._order:
                self._select(self._order[0])
            event.accept()
            return
        if key == Qt.Key.Key_End:
            if self._order:
                self._select(self._order[-1])
            event.accept()
            return
        super().keyPressEvent(event)

    def panel_for_key(self, letter: str) -> int:
        """Which panel a letter selects, or -1 for none of them."""
        if not letter:
            return -1
        letter = letter.upper()
        for button in self._buttons:
            if button.shortcut_key and button.shortcut_key.upper() == letter:
                return button.panel_index
        return -1

    # ------------------------------------------------------------ selection

    def _select(self, index: int) -> None:
        self.set_current(index)
        self.tab_selected.emit(index)

    def set_current(self, index: int) -> None:
        for i, button in enumerate(self._buttons):
            button.setChecked(i == index)

    def current(self) -> int:
        for i, button in enumerate(self._buttons):
            if button.isChecked():
                return i
        return 0

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        for button in self._buttons:
            button.set_theme(theme)
        self.update()

    def refresh_metrics(self) -> None:
        """Follow a change in the UI scale.

        The rail is as wide as its rotated text is tall, so its own width and
        every button's have to be re-measured together; doing one without the
        other either clips the labels or leaves a blank column beside them.
        """
        self.setFixedWidth(px(RAIL_WIDTH))
        for button in self._buttons:
            button.refresh_metrics()
        self.updateGeometry()
        self.update()


class TabKeyFilter(QObject):
    """Turns a bare letter into a tab selection, except while typing.

    It has to be an application-wide filter rather than a QShortcut or a
    handler on the window. A QShortcut on a plain letter wins against the
    widget that has the focus, so typing a page name would jump tabs instead
    of writing a "d"; and a handler on the window never sees the key at all,
    because a list or a table takes letters for its own type-ahead search and
    never passes them up.

    So the key is read before anyone else gets it, and handed on untouched
    whenever something is being typed into -- a line edit, a spin box, an
    editable combo, a text area, or a cell editor, which is a line edit in a
    table. The same goes for a modifier being held, which belongs to one of
    the window's own commands, and for a modal dialog, which owns the
    keyboard while it is up.
    """

    # Everything here takes typed text. A cell being edited in a table is a
    # QLineEdit parented to the view, so it is covered by the first entry.
    _TEXT_WIDGETS = (QLineEdit, QAbstractSpinBox, QTextEdit, QPlainTextEdit)

    _MODIFIERS = (Qt.KeyboardModifier.ControlModifier
                  | Qt.KeyboardModifier.AltModifier
                  | Qt.KeyboardModifier.MetaModifier)

    def __init__(self, rail: TabRail, select, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._rail = rail
        self._select = select

    def eventFilter(self, obj, event) -> bool:
        try:
            if event.type() != QEvent.Type.KeyPress:
                return False
            if event.modifiers() & self._MODIFIERS:
                return False
            letter = event.text()
        except (AttributeError, TypeError):
            return False
        if not letter or not letter.strip():
            return False
        if self.is_typing():
            return False
        index = self._rail.panel_for_key(letter)
        if index < 0:
            return False
        self._select(index)
        return True

    @classmethod
    def is_typing(cls) -> bool:
        """Whether the keyboard currently belongs to something being typed in."""
        try:
            if QApplication.activeModalWidget() is not None:
                return True
            focused = QApplication.focusWidget()
        except (AttributeError, RuntimeError):
            return True            # cannot tell: leave the key alone
        if focused is None:
            return False
        if isinstance(focused, cls._TEXT_WIDGETS):
            return True
        if isinstance(focused, QComboBox) and focused.isEditable():
            return True
        # The general answer, for anything custom that takes text: Qt marks
        # such a widget itself, and asking it is more durable than keeping a
        # list of classes up to date.
        try:
            return bool(focused.testAttribute(
                Qt.WidgetAttribute.WA_InputMethodEnabled))
        except (AttributeError, TypeError):
            return False
