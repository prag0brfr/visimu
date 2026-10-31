# Author: Paulo R. A. Gilz
"""Word-style grid size chooser.

A button that drops down a 5x5 matrix of cells; sweeping the mouse over it
highlights the layout you would get, and clicking commits it. Far quicker than
two spin boxes, and instantly recognisable.
"""

from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QToolButton, QWidget, QWidgetAction, QMenu

from ..model import MAX_TILES
from ..theme import PLOT_LABEL_PT, Theme, plot_font

MAX_ROWS = 5
MAX_COLS = 5


class GridSizeGrid(QWidget):
    """The matrix of cells inside the dropdown."""

    size_chosen = Signal(int, int)   # rows, cols
    size_hovered = Signal(int, int)

    CELL = 20
    GAP = 3
    MARGIN = 8
    LABEL_HEIGHT = 20

    def __init__(self, theme: Theme, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._theme = theme
        self._rows = 0
        self._cols = 0
        self.setMouseTracking(True)
        self.setFixedSize(
            self.MARGIN * 2 + MAX_COLS * (self.CELL + self.GAP) - self.GAP,
            self.MARGIN * 2 + MAX_ROWS * (self.CELL + self.GAP) - self.GAP
            + self.LABEL_HEIGHT,
        )

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.update()

    def _cell_at(self, x: int, y: int) -> Tuple[int, int]:
        col = (x - self.MARGIN) // (self.CELL + self.GAP)
        row = (y - self.MARGIN) // (self.CELL + self.GAP)
        if 0 <= col < MAX_COLS and 0 <= row < MAX_ROWS:
            return int(row) + 1, int(col) + 1
        return 0, 0

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        theme = self._theme

        painter.fillRect(self.rect(), QColor(theme.surface))

        for row in range(MAX_ROWS):
            for col in range(MAX_COLS):
                x = self.MARGIN + col * (self.CELL + self.GAP)
                y = self.MARGIN + row * (self.CELL + self.GAP)
                rect = QRect(x, y, self.CELL, self.CELL)

                selected = row < self._rows and col < self._cols
                # Cells beyond the 25-tile ceiling are drawn faintly so an
                # impossible layout cannot be picked by accident.
                allowed = (row + 1) * (col + 1) <= MAX_TILES

                if selected and allowed:
                    painter.fillRect(rect, QColor(theme.accent))
                elif allowed:
                    painter.fillRect(rect, QColor(theme.panel))
                else:
                    painter.fillRect(rect, QColor(theme.window))

                painter.setPen(QPen(QColor(theme.border)))
                painter.drawRect(rect)

        painter.setPen(QPen(QColor(theme.text if self._rows else theme.text_dim)))
        # From the type scale, not from the painter's font: the application
        # stylesheet sizes text in pixels, so pointSizeF() reports -1 here and
        # the max() below was only ever choosing its own floor.
        painter.setFont(plot_font(PLOT_LABEL_PT))
        label = (
            "{0} x {1}   ({2} tiles)".format(self._rows, self._cols, self._rows * self._cols)
            if self._rows
            else "pick a size"
        )
        painter.drawText(
            QRect(0, self.height() - self.LABEL_HEIGHT, self.width(), self.LABEL_HEIGHT),
            Qt.AlignmentFlag.AlignCenter,
            label,
        )

    def mouseMoveEvent(self, event) -> None:
        position = event.position().toPoint()
        rows, cols = self._cell_at(position.x(), position.y())
        if (rows, cols) != (self._rows, self._cols):
            self._rows, self._cols = rows, cols
            self.size_hovered.emit(rows, cols)
            self.update()

    def leaveEvent(self, event) -> None:
        self._rows = self._cols = 0
        self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        position = event.position().toPoint()
        rows, cols = self._cell_at(position.x(), position.y())
        if rows and cols and rows * cols <= MAX_TILES:
            self.size_chosen.emit(rows, cols)


class GridSizeButton(QToolButton):
    """Toolbar button showing the current layout, with the picker as its menu."""

    size_chosen = Signal(int, int)

    def __init__(self, theme: Theme, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._theme = theme
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setToolTip("Choose the plot grid layout")

        self._menu = QMenu(self)
        self._grid = GridSizeGrid(theme, self._menu)
        self._grid.size_chosen.connect(self._on_chosen)

        action = QWidgetAction(self._menu)
        action.setDefaultWidget(self._grid)
        self._menu.addAction(action)
        self.setMenu(self._menu)

        self.set_current(1, 1)

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        self._grid.set_theme(theme)

    def set_current(self, rows: int, cols: int) -> None:
        self.setText("Grid  {0} x {1}  ".format(rows, cols))

    def _on_chosen(self, rows: int, cols: int) -> None:
        self._menu.close()
        self.set_current(rows, cols)
        self.size_chosen.emit(rows, cols)
