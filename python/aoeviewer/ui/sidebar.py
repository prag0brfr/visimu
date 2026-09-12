# Author: Paulo R. A. Gilz
"""Narrow vertical tab rail down the left edge.

Qt has no vertical-text tab widget, and QTabBar's West position rotates the
*whole* tab including its frame, which looks wrong against a flat panel. So the
buttons paint their own rotated text:
a ~26 px column with the label reading bottom-to-top, which costs almost no
horizontal space no matter how many tabs there are.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QAbstractButton, QSizePolicy, QVBoxLayout, QWidget

from ..theme import RAIL_FONT_PT, Theme

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

        # Set outright rather than derived from self.font(). The application
        # stylesheet sizes text in pixels, which makes pointSizeF() report -1,
        # so "a little smaller than the body text" silently became "whatever
        # the clamp says" -- see the type scale in aoeviewer.theme.
        font = QFont(self.font())
        font.setPointSizeF(RAIL_FONT_PT)
        font.setBold(True)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.6)
        self.setFont(font)

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.update()

    def sizeHint(self) -> QSize:
        metrics = self.fontMetrics()
        # The button's height is the text's *width*, because the text is turned
        # on its side.
        height = max(MIN_BUTTON_HEIGHT, metrics.horizontalAdvance(self.text()) + TEXT_PADDING)
        return QSize(RAIL_WIDTH, height)

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
    theme_toggled = Signal()

    def __init__(self, labels: List[str], theme: Theme, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._theme = theme
        self._buttons: List[VerticalTabButton] = []

        self.setObjectName("Panel")
        self.setFixedWidth(RAIL_WIDTH)
        self.setAutoFillBackground(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(1)

        for index, label in enumerate(labels):
            button = VerticalTabButton(label, theme, self)
            button.clicked.connect(lambda _checked=False, i=index: self._select(i))
            layout.addWidget(button)
            self._buttons.append(button)

        layout.addStretch(1)

        # Light/dark toggle pinned to the bottom.
        self._theme_button = VerticalTabButton("L / D", theme, self)
        self._theme_button.setCheckable(False)
        self._theme_button.setToolTip("Switch between light and dark")
        self._theme_button.clicked.connect(self.theme_toggled.emit)
        layout.addWidget(self._theme_button)

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
        self._theme_button.set_theme(theme)
        self.update()
