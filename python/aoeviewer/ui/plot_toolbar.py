# Author: Paulo R. A. Gilz
"""The bar of page and tile controls sitting directly above the plot grid.

Page and tile management belongs next to what it manipulates, not in a side
panel: you pick a tile by clicking it, so the buttons that move and copy that
tile should be within a few pixels of it.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QToolButton,
    QWidget,
)

from ..model import AppState
from ..theme import Theme, px
from . import icons
from .grid_picker import GridSizeButton


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Plain)
    line.setFixedWidth(1)
    return line


class PlotToolbar(QWidget):
    """Page selection and naming, grid layout, and tile arrangement."""

    pages_changed = Signal()        # page list or layout changed: rebuild the grid
    tiles_changed = Signal()        # tile contents moved: redraw
    active_tile_changed = Signal(int)
    fit_requested = Signal()        # re-frame every tile

    def __init__(self, state: AppState, theme: Theme, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._state = state
        self._theme = theme
        self._updating = False

        self.setObjectName("Panel")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        # --- pages ----------------------------------------------------------
        layout.addWidget(QLabel("Page"))

        self.prev_button = QToolButton()
        self.prev_button.setText("‹")
        self.prev_button.setToolTip("Previous page")
        self.prev_button.clicked.connect(lambda: self._step_page(-1))
        layout.addWidget(self.prev_button)

        self.page_combo = QComboBox()
        self.page_combo.setMinimumWidth(150)
        self.page_combo.setToolTip("Switch page")
        self.page_combo.currentIndexChanged.connect(self._on_page_selected)
        layout.addWidget(self.page_combo)

        self.next_button = QToolButton()
        self.next_button.setText("›")
        self.next_button.setToolTip("Next page")
        self.next_button.clicked.connect(lambda: self._step_page(1))
        layout.addWidget(self.next_button)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Page name")
        self.name_edit.setMaximumWidth(170)
        self.name_edit.setToolTip("Type a name and press Enter, e.g. Speed Plots")
        self.name_edit.returnPressed.connect(self._rename_page)
        self.name_edit.editingFinished.connect(self._rename_page)
        layout.addWidget(self.name_edit)

        for text, handler, tip in (
            ("+", self._add_page, "Add a page"),
            ("−", self._remove_page, "Remove this page"),
            ("◄", lambda: self._move_page(-1), "Move this page earlier"),
            ("►", lambda: self._move_page(1), "Move this page later"),
        ):
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tip)
            button.clicked.connect(handler)
            layout.addWidget(button)

        layout.addWidget(_separator())

        # --- grid layout ----------------------------------------------------
        self.grid_button = GridSizeButton(theme)
        self.grid_button.size_chosen.connect(self._on_grid_chosen)
        layout.addWidget(self.grid_button)

        layout.addWidget(_separator())

        # --- tile arrangement ------------------------------------------------
        self.tile_label = QLabel("Tile 1")
        self.tile_label.setMinimumWidth(46)
        layout.addWidget(self.tile_label)

        self.move_buttons = {}
        for direction, symbol, tip in (
            ("left", "◀", "Swap with the tile to the left"),
            ("up", "▲", "Swap with the tile above"),
            ("down", "▼", "Swap with the tile below"),
            ("right", "▶", "Swap with the tile to the right"),
        ):
            button = QToolButton()
            button.setText(symbol)
            button.setToolTip(tip)
            button.clicked.connect(lambda _c=False, d=direction: self._move_tile(d))
            layout.addWidget(button)
            self.move_buttons[direction] = button

        layout.addWidget(_separator())

        for text, handler, tip in (
            ("Copy", self._copy_tile, "Copy this tile's configuration"),
            ("Cut", self._cut_tile, "Copy this tile and empty it"),
            ("Paste", self._paste_tile, "Replace this tile with the copied one"),
            ("Clear", self._clear_tile, "Empty this tile"),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(handler)
            layout.addWidget(button)

        layout.addWidget(_separator())

        # --- framing ---------------------------------------------------------
        # Fit was in the main toolbar, among the workspace and export actions.
        # It belongs here for the same reason the tile buttons do: it acts on
        # the tiles, and a control that acts on what you picked by clicking
        # should be within a few pixels of it. Behind its own rule, because it
        # moves no tile and copies nothing -- it re-frames what is already
        # there.
        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip(
            "Auto-scale every tile, honouring axis locks  (Ctrl+0)\n"
            "Double-click one plot to fit just that one")
        self.fit_button.clicked.connect(self.fit_requested.emit)
        layout.addWidget(self.fit_button)

        layout.addStretch(1)
        self._apply_icons()

    # ------------------------------------------------------------- refresh

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.grid_button.set_theme(theme)
        self._apply_icons()

    def _apply_icons(self) -> None:
        """Re-paint the icons this bar owns.

        They are painted at a pixel size, so neither the theme's colours nor
        the interface scale reaches them through the stylesheet; this is
        called from set_theme(), which the window calls for both.
        """
        self.fit_button.setIcon(icons.icon("fit", self._theme.text, px(14)))

    def refresh(self) -> None:
        self._updating = True
        try:
            self.page_combo.clear()
            for page in self._state.pages:
                self.page_combo.addItem(page.name)
            self.page_combo.setCurrentIndex(self._state.active_page)

            page = self._state.page
            self.name_edit.setText(page.name)
            self.grid_button.set_current(page.rows, page.cols)

            self.prev_button.setEnabled(self._state.active_page > 0)
            self.next_button.setEnabled(
                self._state.active_page < len(self._state.pages) - 1
            )
            self.tile_label.setText("Tile {0}".format(self._state.active_tile + 1))
            self._update_move_buttons()
        finally:
            self._updating = False

    def _update_move_buttons(self) -> None:
        page = self._state.page
        cols = max(1, page.cols)
        index = self._state.active_tile
        row, col = divmod(index, cols)
        self.move_buttons["left"].setEnabled(col > 0)
        self.move_buttons["right"].setEnabled(
            col < cols - 1 and index + 1 < page.tile_count
        )
        self.move_buttons["up"].setEnabled(row > 0)
        self.move_buttons["down"].setEnabled(
            row < page.rows - 1 and index + cols < page.tile_count
        )

    # -------------------------------------------------------------- pages

    def _on_page_selected(self, index: int) -> None:
        if self._updating or index < 0:
            return
        self._state.active_page = index
        self._state.active_tile = 0
        self.refresh()
        self.pages_changed.emit()

    def _step_page(self, delta: int) -> None:
        target = self._state.active_page + delta
        if 0 <= target < len(self._state.pages):
            self._state.active_page = target
            self._state.active_tile = 0
            self.refresh()
            self.pages_changed.emit()

    def _rename_page(self) -> None:
        if self._updating:
            return
        name = self.name_edit.text().strip()
        page = self._state.page
        if not name or name == page.name:
            return
        self._state.rename_page(self._state.active_page, name)
        self.refresh()
        self.pages_changed.emit()

    def _add_page(self) -> None:
        self._state.add_page()
        self.refresh()
        self.pages_changed.emit()

    def _remove_page(self) -> None:
        if len(self._state.pages) <= 1:
            return
        self._state.remove_page(self._state.active_page)
        self.refresh()
        self.pages_changed.emit()

    def _move_page(self, delta: int) -> None:
        self._state.active_page = self._state.move_page(self._state.active_page, delta)
        self.refresh()
        self.pages_changed.emit()

    def _on_grid_chosen(self, rows: int, cols: int) -> None:
        page = self._state.page
        page.rows, page.cols = rows, cols
        self._state.active_tile = min(self._state.active_tile, page.tile_count - 1)
        self._state.dirty = True
        self.refresh()
        self.pages_changed.emit()

    # -------------------------------------------------------------- tiles

    def _move_tile(self, direction: str) -> None:
        if self._state.move_tile(direction):
            self.refresh()
            self.tiles_changed.emit()
            self.active_tile_changed.emit(self._state.active_tile)

    def _copy_tile(self) -> None:
        self._state.copy_tile()

    def _cut_tile(self) -> None:
        self._state.cut_tile()
        self.tiles_changed.emit()

    def _paste_tile(self) -> None:
        if self._state.paste_tile():
            self.tiles_changed.emit()

    def _clear_tile(self) -> None:
        self._state.clear_tile()
        self.tiles_changed.emit()
