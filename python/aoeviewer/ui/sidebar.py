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
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QAbstractButton, QSizePolicy, QVBoxLayout, QWidget

from ..theme import RAIL_FONT_PT, Theme, px, pt

# The rail is pure chrome down the left edge, so it gives up space first: it
# has to be wide enough to read a rotated word in and no wider. TEXT_PADDING is
# the slack around that word, and it was set when the text was larger.
RAIL_WIDTH = 21
MIN_BUTTON_HEIGHT = 34
TEXT_PADDING = 10


class VerticalTabButton(QAbstractButton):
    """A checkable button whose label is drawn rotated 90 degrees."""

    def __init__(self, text: str, theme: Theme, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._theme = theme
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setToolTip(text.title())

        self.refresh_metrics()

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

    def __init__(self, labels: List[str], theme: Theme, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._theme = theme
        self._buttons: List[VerticalTabButton] = []

        self.setObjectName("Panel")
        self.setFixedWidth(px(RAIL_WIDTH))
        self.setAutoFillBackground(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(1)

        for index, label in enumerate(labels):
            button = VerticalTabButton(label, theme, self)
            button.clicked.connect(lambda _checked=False, i=index: self._select(i))
            layout.addWidget(button)
            self._buttons.append(button)

        # Nothing below the tabs. The light/dark toggle used to be pinned to
        # the foot of this column, where it was one more thing competing for
        # the vertical space the tabs themselves need on a short screen -- and
        # it is not a tab: it selects no panel. It is in the toolbar now,
        # beside the interface-size buttons, with the rest of what changes how
        # the window looks rather than what it shows.
        layout.addStretch(1)

        if self._buttons:
            self._buttons[0].setChecked(True)

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
