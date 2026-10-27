# Author: Paulo R. A. Gilz
"""The grid of plot tiles for the active page."""

from __future__ import annotations

from typing import Dict, List, Set

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QWidget

from ..logging_setup import get_logger
from ..model import AppState
from ..session import Session
from ..theme import Theme
from .plot_tile import PlotTile

_log = get_logger("grid")


class PlotGrid(QWidget):
    """Lays out up to 5x5 tiles and keeps them in sync with the page state."""

    tile_activated = Signal(int)
    tile_params_requested = Signal(int)
    cursor_moved = Signal(str)
    view_changed = Signal()
    point_inspected = Signal(int, int, float)
    slope_measured = Signal(float, float, float, float)
    annotation_moved = Signal()
    legend_moved = Signal()
    text_moved = Signal()
    x_zoom_shared = Signal(float, float)

    def __init__(self, session: Session, state: AppState, theme: Theme, parent=None):
        super().__init__(parent)
        self._session = session
        self._state = state
        self._theme = theme
        self._tiles: List[PlotTile] = []
        self._rows = 0
        self._cols = 0

        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setSpacing(3)

        self.rebuild_layout()

    @property
    def tiles(self) -> List[PlotTile]:
        return self._tiles

    def rebuild_layout(self, force: bool = False) -> None:
        """Lay out tile widgets for the current rows x cols.

        Creating a pyqtgraph PlotWidget is not cheap -- tens of milliseconds
        each, so a 5x5 page costs most of a second. Switching *page* does not
        change the grid shape most of the time, so when the shape matches what
        is already on screen the existing widgets are kept and simply redrawn.
        That is the difference between page changes feeling instant and feeling
        sticky.
        """
        page = self._state.page
        if (not force and self._tiles
                and len(self._tiles) == page.tile_count
                and (self._rows, self._cols) == (page.rows, page.cols)):
            for tile in self._tiles:
                tile.apply_mouse_mode()
            return

        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._tiles.clear()

        index = 0
        for row in range(page.rows):
            for col in range(page.cols):
                if index >= page.tile_count:
                    break
                tile = PlotTile(self._session, self._state, index, self._theme, self)
                tile.activated.connect(self.tile_activated.emit)
                tile.params_requested.connect(self.tile_params_requested.emit)
                tile.cursor_moved.connect(self.cursor_moved.emit)
                tile.view_changed.connect(self.view_changed.emit)
                tile.point_inspected.connect(self.point_inspected.emit)
                tile.slope_measured.connect(self.slope_measured.emit)
                tile.annotation_moved.connect(self.annotation_moved.emit)
                tile.legend_moved.connect(self.legend_moved.emit)
                tile.text_moved.connect(self.text_moved.emit)
                tile.x_zoom_shared.connect(self.x_zoom_shared.emit)
                tile.apply_theme(self._theme)
                self._layout.addWidget(tile, row, col)
                self._tiles.append(tile)
                index += 1

        # Clear every stretch first. QGridLayout remembers them per index, so
        # going from 3x3 to 2x2 would otherwise leave row 2 and column 2 still
        # claiming a share of the space and the tiles would not fill the window.
        from ..model import MAX_TILES
        for index in range(MAX_TILES):
            self._layout.setRowStretch(index, 0)
            self._layout.setColumnStretch(index, 0)

        for row in range(page.rows):
            self._layout.setRowStretch(row, 1)
        for col in range(page.cols):
            self._layout.setColumnStretch(col, 1)

        self._layout.invalidate()
        self._layout.activate()
        self._rows, self._cols = page.rows, page.cols

    def required_channels(self) -> Dict[int, List[str]]:
        """Every channel the visible tiles need, grouped by file.

        Hoisting this out of the tiles turns N lazy per-column loads into one
        batched, multi-threaded read -- and, because it is collected before any
        drawing happens, it can be done on a worker thread.
        """
        names: Set[str] = set()
        for tile in self._tiles:
            names.update(tile.required_channels())
        if not names:
            return {}

        by_file: Dict[int, List[str]] = {}
        for file_index, _sim in self._state.plotted_sims():
            by_file.setdefault(file_index, sorted(names))
        # A differencing tile names two runs of its own, and nothing says
        # either is ticked for drawing -- so their files have to be in the
        # batch as well, or the diff falls back to reading its columns one at
        # a time while the window waits.
        for tile in self._tiles:
            state = tile.tile_state
            runs = state.diff_runs()
            if runs is None:
                continue
            for file_index, _sim_index in runs:
                by_file.setdefault(file_index, sorted(names))
        return by_file

    def plotted_sims_by_file(self) -> Dict[int, List[int]]:
        """The runs that will be drawn, grouped by file.

        Renamed from selected_sims_by_file when the ISO filters stopped writing
        into the DATA selection: "selected" is now one of the two constraints
        rather than the answer, and a preload keyed on the wrong one of them
        would read columns for runs nothing is going to draw.
        """
        by_file: Dict[int, List[int]] = {}
        for file_index, sim_index in self._state.plotted_sims():
            by_file.setdefault(file_index, []).append(sim_index)
        return by_file

    def show_placeholder(self, message: str) -> None:
        for tile in self._tiles:
            tile.show_placeholder(message)

    def refresh(self) -> None:
        """Rebuild every tile's curves from the current state.

        One tile failing must not stop the others. A rebuild reaches a lot of
        drawing code, and when something in it raised -- a marker label
        pyqtgraph would not accept, in the case that prompted this -- the loop
        stopped at that tile: every tile after it kept whatever it was showing
        before, silently. That read as "the plot did not update", which is a
        much harder thing to diagnose than "one tile is empty and the log says
        why".
        """
        for tile in self._tiles:
            try:
                tile.rebuild()
            except Exception:
                _log.exception("tile %d failed to rebuild", tile.index)
                try:
                    tile.show_placeholder("this tile could not be drawn")
                except Exception:
                    pass

    def clear_inspect_dots(self, keep: int = -1) -> None:
        """Take the inspect mark off every tile but `keep`.

        Only one tile is being inspected at a time -- the panel reports one
        run at one instant -- so a dot left behind on the tile you clicked
        before is a claim about a reading that is no longer on screen.
        """
        for index, tile in enumerate(self._tiles):
            if index != keep:
                tile.clear_inspect_dot()

    def refresh_texts(self) -> None:
        """Repaint the captions only.

        A caption is furniture painted in viewport pixels; changing one touches
        no data, so it must not go anywhere near the reload-and-redraw path.
        Sending it there is what made editing a label flicker the whole grid.
        """
        for tile in self._tiles:
            tile.refresh_texts()

    def refresh_active(self) -> None:
        index = self._state.active_tile
        if 0 <= index < len(self._tiles):
            self._tiles[index].rebuild()

    def apply_mouse_mode(self) -> None:
        for tile in self._tiles:
            tile.apply_mouse_mode()

    def autoscale_all(self) -> None:
        for tile in self._tiles:
            tile.autoscale()

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        for tile in self._tiles:
            tile.apply_theme(theme)
