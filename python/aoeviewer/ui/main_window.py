# Author: Paulo R. A. Gilz
"""Main application window."""

from __future__ import annotations

import os
from typing import List, Optional

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QProgressBar,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..model import (
    MOUSE_MODE_HINTS,
    MOUSE_MODE_LABELS,
    AppState,
    MouseMode,
    load_workspace,
    save_workspace,
)
from ..session import CoreError, Session
from ..theme import apply_pyqtgraph_theme, stylesheet, theme_for
from . import icons
from .panels import (
    AliasPanel,
    AreaPanel,
    ComputePanel,
    DataPanel,
    InspectPanel,
    IsoPanel,
    MarkerPanel,
    ParameterPanel,
    Plot3DPanel,
    ScalePanel,
    StylePanel,
    TextPanel,
)
from .plot_grid import PlotGrid
from .plot_toolbar import PlotToolbar
from .sidebar import TabRail
from .tasks import TaskRunner

from ..logging_setup import get_logger

_log = get_logger("window")

FILE_FILTER = (
    "AOE result files (*.aoe *.dat *.txt *.res *.out);;"
    "All files (*)"
)


class MainWindow(QMainWindow):
    def __init__(self, initial_files: Optional[List[str]] = None):
        super().__init__()

        self.session = Session()
        self.state = AppState()

        # Background work. The core session is not internally synchronised, so
        # while a task runs the UI stops touching it entirely -- see tasks.py.
        self._stale_panels = set()
        self._focus_before_task = None
        # Which channels the last completed load put in memory. refresh_plots()
        # compares against it to tell a redraw that needs a read from one that
        # does not; None means "assume nothing".
        self._loaded_signature = None
        self._ui_enabled = True
        # (path, size, mtime) of files the user has said not to reload. Held
        # rather than acted on, so declining leaves the plots as they are --
        # see _check_for_changes().
        self._declined_reloads = set()
        self.tasks = TaskRunner(self)
        self.tasks.started.connect(self._on_task_started)
        self.tasks.finished.connect(self._on_task_finished)
        self.tasks.failed.connect(self._on_task_failed)

        # Cached sample data is capped so that opening study after study cannot
        # grow without bound. Evicted columns are re-read on demand, so the only
        # cost of being wrong here is time, never correctness.
        self.memory_limit_bytes = 2 * 1024 * 1024 * 1024

        self.setWindowTitle("aoeViewer {0}".format(__version__))
        self.resize(1500, 920)

        self._build_toolbar()
        self._build_central()
        self._build_status_bar()
        self.apply_theme()

        # Poll for files changing on disk. Cheap (a stat per file) and it means
        # a running solver's output can be watched live.
        self._watch_timer = QTimer(self)
        self._watch_timer.setInterval(3000)
        self._watch_timer.timeout.connect(self._check_for_changes)
        self._watch_timer.start()
        self._suppress_reload_prompt = False

        if initial_files:
            self.load_files(initial_files)

    # ------------------------------------------------------------- layout

    def _build_toolbar(self) -> None:
        """The one bar across the top.

        Two things it deliberately gets right, because the first version got
        both wrong. Alignment: every button is a labelled icon of the same
        height, groups are separated by one rule and equal padding rather than
        by ad-hoc spacer widgets, and the mouse modes sit as a unit on the
        right. Permanence: a QToolBar can normally be hidden from the context
        menu of the frame, and once hidden there is nothing left to click to
        bring it back -- so the context menu is suppressed and the toggle
        action is disabled.
        """
        self.toolbar = QToolBar("Main")
        self.toolbar.setObjectName("MainToolBar")
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setIconSize(QSize(18, 18))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        # Right-clicking the bar (or any empty part of the frame) would offer
        # "Main" as a checkbox and hide it for good; there is no menu bar to
        # get it back from.
        self.toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self.toolbar.toggleViewAction().setEnabled(False)
        self.toolbar.toggleViewAction().setVisible(False)
        self.addToolBar(self.toolbar)

        self._toolbar_actions = []   # (action, icon name) for re-theming

        def add(text: str, handler, icon_name: str = "",
                shortcut: Optional[str] = None, tip: str = ""):
            action = QAction(text, self)
            action.triggered.connect(handler)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
                tip = "{0}  ({1})".format(tip or text, shortcut)
            if tip:
                action.setToolTip(tip)
            self.toolbar.addAction(action)
            if icon_name:
                self._toolbar_actions.append((action, icon_name))
            return action

        # No "Open" action: Add Files in the Data tab and Load Workspace here
        # already cover it, and a third door onto the same thing is clutter.
        add("Load workspace", self.load_workspace_dialog, "open", "Ctrl+L",
            "Open a saved workspace, replacing what is loaded now")
        add("Save workspace", self.save_workspace, "save", "Ctrl+S",
            "Save the current files, pages, tiles and styles")

        self.toolbar.addSeparator()

        add("Image", self.export_image, "image",
            tip="Save the active page as a PNG")
        add("CSV", self.export_csv, "table",
            tip="Write the active tile's curves to a CSV file, at full resolution")

        self.toolbar.addSeparator()

        add("Fit", self.autoscale_all, "fit", "Ctrl+0",
            "Auto-scale every tile — double-click one plot to fit just that one")

        # Everything after this is pushed to the right edge, so the mouse modes
        # read as their own group instead of trailing the file actions.
        stretch = QWidget()
        stretch.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.toolbar.addWidget(stretch)

        mode_caption = QLabel("Mouse ")
        mode_caption.setObjectName("Hint")
        self.toolbar.addWidget(mode_caption)

        # Mouse modes. Exclusive by construction: an action group means the
        # other modes are not merely unbound but genuinely off, which is what
        # stops a drag panning a plot you only meant to click.
        self.mode_group = QActionGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_actions = {}
        self._mode_icon_names = {
            MouseMode.SELECT: "select",
            MouseMode.CROSSHAIR: "crosshair",
            MouseMode.PAN: "pan",
            MouseMode.ZOOM: "zoom",
        }
        for index, mode in enumerate(
            (MouseMode.SELECT, MouseMode.CROSSHAIR, MouseMode.PAN, MouseMode.ZOOM)
        ):
            action = QAction(MOUSE_MODE_LABELS[mode], self)
            action.setCheckable(True)
            shortcut = "Ctrl+{0}".format(index + 1)
            action.setShortcut(QKeySequence(shortcut))
            action.setToolTip("{0} — {1}  ({2})".format(
                MOUSE_MODE_LABELS[mode], MOUSE_MODE_HINTS[mode], shortcut))
            action.triggered.connect(lambda _c=False, m=mode: self.set_mouse_mode(m))
            self.mode_group.addAction(action)
            self.toolbar.addAction(action)
            self.mode_actions[mode] = action
        self.mode_actions[MouseMode.SELECT].setChecked(True)

        self._apply_toolbar_icons()

    def _apply_toolbar_icons(self) -> None:
        theme = theme_for(self.state.dark_mode)
        icons.clear_cache()
        for action, name in getattr(self, "_toolbar_actions", []):
            action.setIcon(icons.icon(name, theme.text, 20))
        for mode, name in getattr(self, "_mode_icon_names", {}).items():
            self.mode_actions[mode].setIcon(icons.icon(name, theme.text, 20))

    def createPopupMenu(self):
        """No toolbar/dock context menu.

        QMainWindow builds one automatically listing every toolbar as a
        checkable item. Ticking it off hides the only toolbar there is, and
        with no menu bar in this window there is then no way to get it back
        short of deleting the settings. Returning None removes that trap.
        """
        return None

    def _build_central(self) -> None:
        theme = theme_for(self.state.dark_mode)

        self.data_panel = DataPanel(self.session, self.state)
        self.param_panel = ParameterPanel(self.session, self.state)
        self.compute_panel = ComputePanel(self.session, self.state)
        self.iso_panel = IsoPanel(self.session, self.state)
        self.style_panel = StylePanel(self.session, self.state)
        self.scale_panel = ScalePanel(self.session, self.state)
        self.area_panel = AreaPanel(self.session, self.state)
        self.marker_panel = MarkerPanel(self.session, self.state)
        self.alias_panel = AliasPanel(self.session, self.state)
        self.inspect_panel = InspectPanel(self.session, self.state)
        self.plot3d_panel = Plot3DPanel(self.session, self.state)
        self.text_panel = TextPanel(self.session, self.state)

        self._panels = [
            ("DATA", self.data_panel),
            ("PARAMS", self.param_panel),
            ("COMPUTE", self.compute_panel),
            ("ALIAS", self.alias_panel),
            ("ISO", self.iso_panel),
            ("STYLE", self.style_panel),
            ("SCALE", self.scale_panel),
            ("3D", self.plot3d_panel),
            ("INSPECT", self.inspect_panel),
            ("AREA", self.area_panel),
            ("MARKERS", self.marker_panel),
            ("TEXT", self.text_panel),
        ]

        # A vertical rail rather than a QTabWidget: with six tabs a horizontal
        # tab bar either wraps or truncates, while rotated labels cost 28 px of
        # width no matter how many there are.
        self.rail = TabRail([name for name, _w in self._panels], theme)
        self.rail.tab_selected.connect(self._on_tab_selected)
        self.rail.theme_toggled.connect(self.toggle_theme)

        self.panel_stack = QStackedWidget()
        # 230, not 300. The window's minimum width is the sum of its parts, and
        # on a laptop screen that sum was wider than the screen -- which is why
        # the window could not be shrunk, and on some Linux window managers
        # could not be maximised either, since a maximise that cannot satisfy
        # the minimum is refused outright. The panels are narrower now that the
        # type is smaller, so this is the width they actually need rather than
        # the width they used to.
        self.panel_stack.setMinimumWidth(230)
        self.panel_stack.setMaximumWidth(600)
        for _name, widget in self._panels:
            self.panel_stack.addWidget(widget)

        self.data_panel.add_files_requested.connect(self.open_files)
        self.data_panel.selection_changed.connect(self._on_selection_changed)
        self.data_panel.files_changed.connect(self.refresh_all)
        self.param_panel.tile_changed.connect(self._on_tile_content_changed)
        self.param_panel.alias_requested.connect(self._on_alias_created)
        self.plot3d_panel.tile_changed.connect(self._on_tile_content_changed)
        self.compute_panel.parameters_changed.connect(self.refresh_all)
        self.iso_panel.filter_changed.connect(self._on_iso_filter)
        self.style_panel.style_changed.connect(self.refresh_plots)
        self.style_panel.legend_applied.connect(self._on_legend_applied)
        self.scale_panel.scale_changed.connect(self.refresh_plots)
        self.area_panel.changed.connect(self.refresh_plots)
        self.marker_panel.changed.connect(self.refresh_plots)

        self.grid = PlotGrid(self.session, self.state, theme)
        # Captions repaint on their own, without reloading a single sample.
        # Connected through a method of this window rather than straight to
        # grid.refresh_texts: a bound method has to be looked up at connect
        # time, so wiring it up here in the panel block -- which runs before
        # the grid exists -- failed at construction and took the window down.
        self.text_panel.changed.connect(self._on_texts_changed)
        self.grid.tile_activated.connect(self._on_tile_activated)
        self.grid.cursor_moved.connect(self._on_cursor_moved)
        self.grid.view_changed.connect(self.scale_panel.follow_view)
        self.grid.view_changed.connect(self._on_view_changed)
        self.grid.point_inspected.connect(self._on_point_inspected)
        self.grid.slope_measured.connect(self._on_slope_measured)
        self.grid.annotation_moved.connect(self._on_annotation_moved)
        self.grid.legend_moved.connect(self._on_legend_moved)
        self.grid.text_moved.connect(self._on_text_moved)
        self.inspect_panel.load_requested.connect(self._load_inspect)
        self.alias_panel.aliases_changed.connect(self.refresh_all)

        self.plot_toolbar = PlotToolbar(self.state, theme)
        self.plot_toolbar.pages_changed.connect(self._on_pages_changed)
        self.plot_toolbar.tiles_changed.connect(self._on_tiles_rearranged)
        self.plot_toolbar.active_tile_changed.connect(self._on_tile_activated)

        plot_area = QWidget()
        plot_layout = QVBoxLayout(plot_area)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)
        plot_layout.addWidget(self.plot_toolbar)
        plot_layout.addWidget(self.grid, 1)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.panel_stack)
        self.splitter.addWidget(plot_area)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([340, 1160])

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.rail)
        layout.addWidget(self.splitter, 1)
        self.setCentralWidget(container)

        self._sync_page_combo()

    @staticmethod
    def _refresh_now(widget) -> bool:
        """Refresh one panel, and survive it failing.

        One panel is a list of things; the plots are the work. A refresh that
        raised used to take the whole handler with it, and the damage was never
        confined to the panel: dragging a shaded band ran
        _apply_annotation_move(), which refreshed the marker and area panels
        *before* redrawing the grid, so when AreaPanel.refresh() raised on a
        field that had been removed from the model, the grid refresh never
        happened at all. The band moved on the tile under the cursor -- pyqtgraph
        having already dragged that item -- and stayed where it was on every
        other tile. Two bugs from one exception, and only one of them looked
        like an exception.

        Logged rather than swallowed, for the same reason PlotGrid.refresh()
        logs a tile that will not draw: a panel that is visibly out of date and
        a line in the log is a far easier thing to diagnose than a window that
        silently stops keeping up.
        """
        if widget is None or not hasattr(widget, "refresh"):
            return False
        try:
            widget.refresh()
            return True
        except Exception:
            _log.exception("panel %s failed to refresh",
                           type(widget).__name__)
            return False

    def _on_tab_selected(self, index: int) -> None:
        self.panel_stack.setCurrentIndex(index)
        widget = self._panels[index][1]
        # Always refresh on show: a panel becomes visible rarely, and doing it
        # unconditionally avoids having to track staleness perfectly.
        self._stale_panels.discard(widget)
        self._refresh_now(widget)

    def _refresh_panel(self, widget) -> None:
        """Refresh one panel now, or mark it for when it is next shown.

        Panels enumerate every plotted curve, so refreshing all ten on every
        page change is most of what made page switching feel slow. Only the
        visible one is done immediately; the rest are refreshed lazily the
        moment they are selected.
        """
        if widget is None or not hasattr(widget, "refresh"):
            return
        if widget is self.panel_stack.currentWidget():
            self._stale_panels.discard(widget)
            self._refresh_now(widget)
        else:
            self._stale_panels.add(widget)

    def _invalidate_panels(self) -> None:
        """Mark every panel stale, refreshing only the one on screen."""
        current = self.panel_stack.currentWidget()
        self._stale_panels = {panel for _name, panel in self._panels if panel is not current}
        self._refresh_now(current)

    def _build_status_bar(self) -> None:
        self.status_label = QLabel("Ready")
        self.readout_label = QLabel("")
        self.readout_label.setObjectName("Hint")

        # Indeterminate: the core does not report progress from inside a parse,
        # and a fake percentage would be worse than an honest "working".
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedWidth(120)
        self.progress.setTextVisible(False)
        self.progress.hide()

        self.memory_label = QLabel("")
        self.memory_label.setObjectName("Hint")
        self.memory_label.setToolTip(
            "Sample data cached by the core. Columns are dropped automatically "
            "when this exceeds the limit, and re-read when needed again."
        )

        bar = self.statusBar()
        bar.addWidget(self.status_label, 1)
        bar.addWidget(self.progress)
        bar.addPermanentWidget(self.readout_label)
        bar.addPermanentWidget(self.memory_label)

    # ------------------------------------------------------- background

    def _set_ui_enabled(self, enabled: bool) -> None:
        """Disable the interactive surface while a task owns the core session.

        The plot grid belongs in here as much as the panels do. Panning a tile
        emits a range change, and with the Inspect tab showing that reads a
        channel on the GUI thread -- which can start a column load, i.e. a
        second thread inside a core that is not internally synchronised, while
        a worker is already in it. The panels were guarded and the grid, the
        one surface a user is most likely to be touching, was not.

        Disabling a widget takes keyboard focus away from anything inside it,
        and Qt does not give it back. That is invisible for a button, and very
        visible for a text field: typing a caption fired a redraw per keystroke,
        each redraw disabled the panel, and the caret was gone after every
        single character. So the focused widget is put back afterwards.

        Redundant calls are dropped. Disabling a widget changes how it is drawn,
        so every plot in the grid repaints on the way down and again on the way
        back up -- and a chain of queued tasks emits started/finished for each
        one, asking for a state the UI is already in.
        """
        enabled = bool(enabled)
        if enabled == self._ui_enabled:
            return
        self._ui_enabled = enabled

        if not enabled:
            focused = QApplication.focusWidget()
            self._focus_before_task = (
                focused if focused is not None
                and self.panel_stack.isAncestorOf(focused) else None
            )

        self.panel_stack.setEnabled(enabled)
        self.plot_toolbar.setEnabled(enabled)
        self.rail.setEnabled(enabled)
        if hasattr(self, "grid"):
            self.grid.setEnabled(enabled)

        if enabled and self._focus_before_task is not None:
            widget = self._focus_before_task
            self._focus_before_task = None
            # It may have been destroyed by a panel rebuild in the meantime.
            try:
                if widget.isVisible() and widget.isEnabled():
                    widget.setFocus(Qt.FocusReason.OtherFocusReason)
            except RuntimeError:
                pass

    def _on_task_started(self, description: str) -> None:
        self.status_label.setText(description)
        self.progress.show()
        self._set_ui_enabled(False)

    def _on_task_finished(self) -> None:
        self.progress.hide()
        self._set_ui_enabled(True)
        self._update_status()

    def _on_task_failed(self, message: str) -> None:
        self.progress.hide()
        self._set_ui_enabled(True)
        QMessageBox.warning(self, "Background task failed", message)

    def _update_memory_label(self) -> None:
        resident = self.session.resident_bytes
        self.memory_label.setText("cache {0:.0f} MB".format(resident / (1024 * 1024)))

    def _enforce_memory_limit(self) -> bool:
        """Trim resident columns back under the cap. True if anything was let go.

        The answer matters to refresh_plots(): what it remembers having loaded
        is only true until the cap evicts some of it.
        """
        if self.session.resident_bytes > self.memory_limit_bytes:
            self.session.trim_memory(int(self.memory_limit_bytes * 0.8))
            return True
        return False

    # -------------------------------------------------------------- theme

    def apply_theme(self) -> None:
        theme = theme_for(self.state.dark_mode)
        apply_pyqtgraph_theme(theme)
        self.setStyleSheet(stylesheet(theme))
        if hasattr(self, "rail"):
            self.rail.set_theme(theme)
        if hasattr(self, "plot_toolbar"):
            self.plot_toolbar.set_theme(theme)
        if hasattr(self, "grid"):
            self.grid.apply_theme(theme)
        if hasattr(self, "toolbar"):
            self._apply_toolbar_icons()

    def toggle_theme(self) -> None:
        self.state.dark_mode = not self.state.dark_mode
        self.apply_theme()

    # -------------------------------------------------------------- files

    def open_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add AOE result files", "", FILE_FILTER
        )
        if paths:
            self.load_files(paths)

    def load_files(self, paths: List[str]) -> None:
        """Open files on a worker thread; a 500 MB index scan is not instant."""

        def work():
            loaded = []
            failures = []
            cached = 0
            for path in paths:
                try:
                    handle = self.session.add_file(path)
                except CoreError as exc:
                    failures.append("{0}: {1}".format(os.path.basename(path), exc))
                    continue
                loaded.append((handle.index, handle.simulation_count))
                if self._build_cache_for(path, handle.index):
                    cached += 1
            return loaded, failures, cached

        def done(result):
            loaded, failures, cached = result
            for index, sim_count in loaded:
                # Select every run of the first file, so something is on screen
                # immediately rather than an empty grid.
                if index == 0:
                    for sim in range(sim_count):
                        self.state.set_selected(index, sim, True)

            self.refresh_all()
            if failures:
                QMessageBox.warning(
                    self, "Some files could not be read", "\n".join(failures[:10])
                )
            message = "{0} file(s) loaded, {1} total".format(
                len(loaded), self.session.file_count)
            if cached:
                message += "  (cached {0} for faster reopening)".format(cached)
            self.status_label.setText(message)

        # Say what is actually about to happen. The first open of a large file
        # also writes its sidecar, which is a second full parse -- on three
        # 550 MB files that is most of the wait, and a progress message reading
        # "Opening 3 file(s)..." for the whole of it makes a one-off cost look
        # like the normal speed of opening a file.
        pending = sum(1 for path in paths if self._cache_would_be_built(path))
        if pending:
            description = ("Opening {0} file(s) and building {1} cache(s) — "
                           "this happens once; later opens are much faster..."
                           ).format(len(paths), pending)
        else:
            description = "Opening {0} file(s)...".format(len(paths))
        self.tasks.submit(description, work, done)

    def _cache_would_be_built(self, path: str) -> bool:
        """Whether opening `path` will also write a sidecar, near enough.

        A guess, and only ever used to word a progress message. It cannot see a
        sidecar that lives in the per-user fallback directory, so it errs
        towards announcing work that turns out to be already done -- which is
        the harmless direction.
        """
        if not self.session.cache_enabled:
            return False
        try:
            if os.path.getsize(path) < self.CACHE_THRESHOLD_BYTES:
                return False
        except OSError:
            return False
        return not os.path.exists(path + ".aoec")

    # Below this, parsing the file is quick enough that a sidecar would save
    # nothing measurable and would just leave an extra file beside the results.
    CACHE_THRESHOLD_BYTES = 8 * 1024 * 1024

    def _build_cache_for(self, path: str, index: int) -> bool:
        """Write a `.aoec` sidecar for a freshly opened file, if it is worth it.

        Runs on the worker thread, inside the same task that opened the file,
        which is deliberate: the user is already waiting, and doing it here
        costs one extra parse *once* rather than locking the window later for a
        file they may be in the middle of using.

        Every reason to skip is silent. A sidecar is an optimisation the user
        did not ask for, so failing to write one -- results on a read-only
        share, a full disk, caching switched off -- is not a problem to report;
        the file has already opened and works.
        """
        if not self.session.cache_enabled:
            return False
        # Already served from a cache: this open was the fast path.
        if self.session.cache_active(index):
            return False
        try:
            if os.path.getsize(path) < self.CACHE_THRESHOLD_BYTES:
                return False
        except OSError:
            return False
        try:
            return bool(self.session.build_cache(index))
        except CoreError:
            return False

    def _file_stamp(self, index: int):
        """(path, size, mtime) of one open file as it is on disk right now."""
        handle = self.session.file(index)
        if handle is None:
            return None
        try:
            info = os.stat(handle.path)
        except OSError:
            return None
        return (handle.path, info.st_size, info.st_mtime)

    def _check_for_changes(self) -> None:
        if self._suppress_reload_prompt:
            return
        # The core session is not internally synchronised, and this runs on the
        # GUI thread from a timer. Walking the file list while a worker is
        # adding to it reads a vector that may be reallocating underneath.
        if self.tasks.busy:
            return
        stale = self.session.stale_files()
        # Drop the ones already declined, unless they have changed again since.
        stale = [index for index in stale
                 if self._file_stamp(index) not in self._declined_reloads]
        if not stale:
            return

        self._suppress_reload_prompt = True
        try:
            names = ", ".join(
                self.session.file(i).name for i in stale if self.session.file(i)
            )
            answer = QMessageBox.question(
                self,
                "Files changed on disk",
                "These files changed:\n\n{0}\n\nReload them?".format(names),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                for index in stale:
                    handle = self.session.file(index)
                    if handle:
                        handle.reload()
                # Reloaded, so nothing about these is declined any more.
                self._declined_reloads.clear()
                self.refresh_all()
            else:
                # No means no. This used to reload anyway -- the only way to
                # stop the timer asking again was to re-read the timestamps,
                # and re-reading them is a reload -- so declining swapped the
                # data under the plots, which is the one thing the question was
                # asked to prevent.
                #
                # What the timer actually needs is to recognise the change it
                # has already been told about, and that is a fact about the
                # file on disk rather than about the session. Remembering the
                # size and mtime that were declined costs a stat per tick and
                # leaves the loaded data alone; if the file changes again, the
                # stamp no longer matches and the question comes back.
                for index in stale:
                    stamp = self._file_stamp(index)
                    if stamp is not None:
                        self._declined_reloads.add(stamp)
        finally:
            self._suppress_reload_prompt = False

    # ----------------------------------------------------------- refresh

    def refresh_all(self) -> None:
        # Everything that routes through here -- a file added, removed or
        # reloaded, a computed channel changed, an alias edited, a workspace
        # loaded -- can have moved the sample buffers, so what was resident
        # before says nothing about what is resident now.
        self._loaded_signature = None
        # Settle tiles that name channels the open files do not have, before
        # anything looks at them. This is the deliberate, once-per-file-change
        # moment for it, and the panels no longer do it as a side effect of
        # being displayed.
        notes = self.state.resolve_missing_parameters(self.session.parameters)
        if notes:
            _log.info("re-pointed missing parameters: %s", "; ".join(notes))
            self.status_label.setText(
                "{0} tile axis/axes re-pointed: {1}".format(
                    len(notes), notes[0] + ("..." if len(notes) > 1 else ""))
            )

        # Which runs the ISO filters admit is a fact about the loaded files, so
        # it is recomputed here and only here -- this is the one moment those
        # can have changed. Doing it on every redraw would put an ISO column
        # read per file into the restyle path for an answer that never moves;
        # not doing it at all would leave a newly loaded file's runs outside an
        # answer computed before it arrived, and so silently filtered out.
        self._resolve_iso_filters()

        # One panel failing must not stop the others, and must not abort the
        # task callback this runs inside -- that is what leaves the UI disabled.
        for name, panel in self._panels:
            try:
                panel.refresh()
            except Exception:
                _log.exception("panel %s failed to refresh", name)
        self._sync_page_combo()
        self.refresh_plots()

    def _resolve_iso_filters(self) -> None:
        """Re-derive the ISO half of what gets drawn."""
        self.state.resolve_iso_filters(
            [(handle.index, handle.simulation_count)
             for handle in self.session.files],
            self.session.iso_values,
        )

    def _on_selection_changed(self) -> None:
        # Which curves exist changed, so every panel listing them is stale.
        self._invalidate_panels()
        self.refresh_plots()

    def _on_tile_content_changed(self) -> None:
        # A parameter arriving on a tile is exactly when a limit declared for
        # that parameter has to take effect -- otherwise a tile filled in after
        # the limit was set quietly ignores it.
        self.state.apply_global_limits(self.state.tile)
        self._invalidate_panels()
        self.refresh_active_plot()

    def _on_pages_changed(self) -> None:
        # rebuild_layout() reuses the tile widgets when the grid shape has not
        # changed, and only the visible panel is refreshed; the rest catch up
        # when selected. Together those are what make switching page quick.
        self.grid.rebuild_layout()
        self._invalidate_panels()
        self.refresh_plots()

    def _on_tiles_rearranged(self) -> None:
        self._invalidate_panels()
        self.refresh_plots()

    def refresh_plots(self) -> None:
        """Load whatever the visible tiles need, then draw them.

        The load is hoisted out of the individual tiles into one batched call so
        it can run on a worker thread; drawing then happens on the GUI thread
        with everything already resident, which is fast.
        """
        # Give a tile that is plotting something but has never been given an
        # x axis the default one. Silent and idempotent, and it has to be here
        # rather than in a panel: panels refresh only when their tab is shown,
        # so doing it there made the result depend on where you had clicked.
        self.state.apply_parameter_defaults(self.session.parameters)

        # Covers every route by which a tile can acquire parameters -- paste,
        # reorder, a new page, a loaded workspace -- not just the edit path.
        self.state.apply_global_limits_everywhere()
        # Every route that redraws the grid also settles which tile is active,
        # so this is the one place the highlight has to be kept honest.
        self._refresh_active_border()

        requirements = self.grid.required_channels()
        if not requirements:
            self.grid.refresh()
            self._update_status()
            return

        sims_by_file = self.grid.plotted_sims_by_file()
        signature = (
            tuple(sorted((index, tuple(names))
                         for index, names in requirements.items())),
            tuple(sorted((index, tuple(sims))
                         for index, sims in sims_by_file.items())),
        )

        # Nothing new to read: draw now, on this thread, and skip the worker
        # entirely. Most redraws are of this kind -- restyling a curve, toggling
        # the grid or the legend, editing a limit -- and routing them through a
        # background task was costing far more than the read it was there to
        # cover. Starting a task disables the panels, the rail and the whole
        # grid, and re-enables them a moment later; every plot widget in the
        # page repaints on each of those, so a style tweak was paying for two
        # full-grid repaints, a thread, and two trips round the event loop
        # before a single curve was redrawn.
        #
        # `busy` still forces the slow path: while a worker is inside the core
        # session, nothing on this thread may touch it (see tasks.py).
        if signature == self._loaded_signature and not self.tasks.busy:
            self.grid.refresh()
            self._update_status()
            return

        def work():
            for file_index, names in requirements.items():
                self.session.preload(file_index, sims_by_file.get(file_index, []), names)
            return self._enforce_memory_limit()

        def done(trimmed):
            # Only claim the data is resident if the cap did not just evict
            # part of it; otherwise the next redraw would take the fast path
            # and read columns back in on the GUI thread.
            self._loaded_signature = None if trimmed else signature
            self.grid.refresh()
            self._update_status()
            self._update_memory_label()

        if self.tasks.busy:
            self.grid.show_placeholder("loading...")
        self.tasks.submit("Reading channels...", work, done)

    def refresh_active_plot(self) -> None:
        # One tile still needs its channels resident, so it goes through the
        # same path rather than risking a synchronous load on the GUI thread.
        self.refresh_plots()

    def _update_status(self) -> None:
        # Both numbers when they differ. The DATA tab and the ISO filters are
        # independent constraints, so a run can be ticked and still not drawn,
        # and "12 run(s) plotted" over a DATA tab showing twenty ticks is the
        # kind of thing that reads as a bug. Saying how many the filters are
        # holding back is what makes it read as a filter.
        plotted = len(self.state.plotted_sims())
        held = self.state.excluded_by_iso()
        runs = ("{0} run(s) plotted".format(plotted) if not held
                else "{0} of {1} run(s) plotted, {2} held back by the ISO "
                     "filter(s)".format(plotted, plotted + held, held))
        self.status_label.setText(
            "{0} file(s), {1}, {2} parameter(s)".format(
                self.session.file_count, runs, len(self.session.parameters),
            )
        )

    # -------------------------------------------------------- mouse modes

    def set_mouse_mode(self, mode: MouseMode) -> None:
        self.state.mouse_mode = mode
        action = self.mode_actions.get(mode)
        if action is not None:
            action.setChecked(True)
        self.grid.apply_mouse_mode()
        self.status_label.setText(MOUSE_MODE_HINTS[mode])

        # Inspect mode is only useful next to its panel, so show it.
        if mode == MouseMode.CROSSHAIR:
            self._show_panel(self.inspect_panel)

    def autoscale_all(self) -> None:
        self.grid.autoscale_all()

    # ------------------------------------------------------------ inspect

    def _show_panel(self, panel) -> None:
        """Bring one panel to the front of the rail."""
        for index, (_name, widget) in enumerate(self._panels):
            if widget is panel:
                self.rail.set_current(index)
                self.panel_stack.setCurrentIndex(index)
                return

    def _on_point_inspected(self, file_index: int, sim_index: int, x_value: float) -> None:
        # Switching to Inspect mode already shows the panel, but the user is
        # free to go and look at something else afterwards -- and then clicking
        # a curve read a run, filled in a table, and left it on a tab nobody
        # was looking at. Clicking a curve in Inspect mode is an unambiguous
        # request to see those numbers, so the panel comes forward every time.
        self._show_panel(self.inspect_panel)
        self.inspect_panel.request(file_index, sim_index, x_value)

    def _load_inspect(self, file_index: int, sim_index: int, x_value: float) -> None:
        """Read every channel of one run at the clicked point.

        This loads all of that run's columns, which on a large file is real
        work, so it goes through the background runner. Afterwards they are
        cached, so clicking around the same run costs nothing.
        """
        handle = self.session.file(file_index)
        if handle is None:
            return
        x_param = self.state.tile.x_param or (handle.headers[0] if handle.headers else "")
        channels = list(handle.headers)
        # Computed parameters are worth including: they are usually the derived
        # quantity the engineer actually cares about.
        channels += [
            name for name in self.session.parameters
            if self.session.is_computed(name)
        ]

        def work():
            self.session.preload(file_index, [sim_index], channels)
            reference = self.session.channel(file_index, sim_index, x_param)
            if reference.size == 0:
                return x_param, []
            index = Session.nearest_index(reference, x_value)

            rows = []
            for name in channels:
                values = self.session.channel(file_index, sim_index, name)
                if values.size == 0:
                    continue
                position = index if index < values.size else values.size - 1
                rows.append((name, float(values[position])))
            return x_param, rows, self._enforce_memory_limit()

        def done(result):
            axis, rows, trimmed = result
            # Reading every channel of a run is the most likely thing to push
            # the cache over its cap, and what it evicts may be exactly the
            # columns the plots are drawn from.
            if trimmed:
                self._loaded_signature = None
            self.inspect_panel.show_result(file_index, sim_index, x_value, axis, rows)
            self._update_memory_label()

        self.tasks.submit("Reading every channel of the clicked run...", work, done)

    def _on_slope_measured(self, x0: float, y0: float, x1: float, y1: float) -> None:
        # Same reasoning as a click: dragging out a slope is a request to read
        # the gradient, and the gradient is written in the Inspect panel.
        self._show_panel(self.inspect_panel)
        self.inspect_panel.show_slope(x0, y0, x1, y1)
        for index, (_name, panel) in enumerate(self._panels):
            if panel is self.inspect_panel:
                self.rail.set_current(index)
                self.panel_stack.setCurrentIndex(index)
                break

    def _on_annotation_moved(self) -> None:
        """A marker or area was dragged on one tile.

        Both are global, so every other tile showing them has to catch up, and
        the panel that lists them is now out of date. Deferred by a zero-delay
        timer because this arrives from inside the dragged item's own signal --
        rebuilding the tile immediately would delete that item while it is
        still handling the event.
        """
        QTimer.singleShot(0, self._apply_annotation_move)

    def _apply_annotation_move(self) -> None:
        # The plots first. They are the thing the drag was for, and they are
        # global: the same band belongs on every tile drawn against its
        # parameter, so a drag that reaches only the tile under the cursor is
        # not a cosmetic failure but a wrong picture. Refreshing the lists
        # ahead of them put two panels between the user's gesture and its
        # result for no reason at all.
        self.grid.refresh()
        self._refresh_panel(self.marker_panel)
        self._refresh_panel(self.area_panel)

    def _on_legend_applied(self, count: int) -> None:
        """The legend was copied to every tile. Say how far it reached.

        Most of those tiles are on other pages or in another part of the grid,
        so without this the button looks like it did nothing.
        """
        if not count:
            self.status_label.setText(
                "Every tile already had this legend."
            )
            return
        self.status_label.setText(
            "Legend applied to {0} other tile{1}.".format(
                count, "" if count == 1 else "s")
        )

    def _on_legend_moved(self) -> None:
        """The legend was dragged. Only the Style panel needs to know."""
        self._refresh_panel(self.style_panel)
        self.status_label.setText(
            "Legend position saved. Pick a corner in STYLE to snap it back."
        )

    def _on_texts_changed(self) -> None:
        """A caption was added, removed or edited in the panel."""
        if hasattr(self, "grid"):
            self.grid.refresh_texts()

    def _on_text_moved(self) -> None:
        """A caption was dragged. Only the Text panel's boxes need to catch up;
        the overlay has already moved the note and repainted itself, so there is
        nothing here to redraw."""
        self._refresh_panel(self.text_panel)

    def _on_alias_created(self, alias: str) -> None:
        self.status_label.setText("Alias '{0}' created.".format(alias))
        self.refresh_all()

    def _on_cursor_moved(self, text: str) -> None:
        self.readout_label.setText(text)

    def _on_view_changed(self) -> None:
        # Statistics are of the visible window, so they have to follow pan and
        # zoom. They live in the Inspect tab now; previously this checked for a
        # panel that is no longer the one showing them, which is why the
        # figures sat still while the plot moved.
        if self.panel_stack.currentWidget() is self.inspect_panel:
            self.inspect_panel.refresh_stats()

    def _on_iso_filter(self) -> None:
        # Re-derive which runs the filters admit. The ISO panel declares the
        # filter; this is where it becomes a set of runs, because this is the
        # only place that knows which files are loaded.
        self._resolve_iso_filters()
        # The DATA tab is refreshed whether or not it is the visible panel, and
        # its ticks are untouched by any of this -- what changes is which of
        # its rows are dimmed as held back by the filter.
        self._refresh_now(self.data_panel)
        self.refresh_plots()

    def _refresh_active_border(self) -> None:
        """Repaint the active-tile highlight from the state, wherever we are.

        The one place that knows which tile the panels are editing. It used to
        be repainted only inside the branch that ran when the index *changed*,
        and three ordinary things never reach that branch: moving a tile updates
        the active index before the toolbar emits, so the indices already match;
        changing page resets it without emitting at all; and on a 1x1 grid the
        index is always 0. The border then pointed at the wrong tile, or at
        none -- which misreports what an edit is about to change.
        """
        if not hasattr(self, "grid"):
            return
        active = self.state.active_tile
        for position, tile in enumerate(self.grid.tiles):
            tile.set_active_border(position == active)

    def _on_tile_activated(self, index: int) -> None:
        if index == self.state.active_tile:
            # Clicking the already-active tile must not re-enumerate every
            # curve; with a hundred runs selected that is what made repeated
            # clicking feel sticky. The border is still repainted, because the
            # click may have arrived precisely because it was showing the wrong
            # tile.
            self._refresh_active_border()
            self.plot_toolbar.refresh()
            return
        self.state.active_tile = index
        # The Inspect panel reports one run at one instant, and that instant
        # belonged to the tile that was active. A dot left on it would point at
        # a reading no longer on screen.
        self.grid.clear_inspect_dots(keep=index)
        self._invalidate_panels()
        self.plot_toolbar.refresh()
        self._refresh_active_border()

    # -------------------------------------------------------------- pages

    def _sync_page_combo(self) -> None:
        if hasattr(self, "plot_toolbar"):
            self.plot_toolbar.refresh()



    # ---------------------------------------------------------- workspace

    def save_workspace(self) -> None:
        path = self.state.workspace_path
        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save workspace", "", "aoeViewer workspace (*.aoews);;All files (*)"
            )
            if not path:
                return
            if not path.lower().endswith(".aoews"):
                path += ".aoews"

        computed = {
            name: self.session.computed_source(name)
            for name in self.session.parameters
            if self.session.is_computed(name)
        }
        try:
            save_workspace(
                path,
                self.state,
                [f.path for f in self.session.files],
                computed,
                {},
            )
        except OSError as exc:
            QMessageBox.warning(self, "Could not save", str(exc))
            return

        self.state.workspace_path = path
        self.state.dirty = False
        self.status_label.setText("Workspace saved to {0}".format(path))

    def load_workspace_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load workspace", "", "aoeViewer workspace (*.aoews);;All files (*)"
        )
        if not path:
            return

        try:
            data = load_workspace(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not load", str(exc))
            return

        # Everything that touches the core goes to a worker, like opening files
        # already did. Done inline it was both the one heavy core operation left
        # on the GUI thread -- a workspace naming three 500 MB files froze the
        # window for seconds -- and unguarded against a preload still running,
        # whose datasets session.clear() would have pulled out from under it.
        def work():
            self.session.clear()
            missing = []
            for file_path in data.get("files", []):
                try:
                    handle = self.session.add_file(file_path)
                except CoreError:
                    missing.append(file_path)
                    continue
                self._build_cache_for(file_path, handle.index)

            for name, expression in data.get("computed", {}).items():
                try:
                    self.session.set_computed(name, expression)
                except CoreError:
                    pass
            for alias, target in data.get("aliases", {}).items():
                self.session.set_alias(alias, target)
            return missing

        def done(missing):
            # Replacing the state is a GUI-side operation and belongs here, not
            # in the worker: the panels and tiles must be rebound to the new one
            # in the same turn of the event loop that starts using it.
            self.state = AppState()
            self.state.load_dict(data)
            self.state.workspace_path = path

            self._rebind_state()
            self.refresh_all()

            if missing:
                QMessageBox.information(
                    self,
                    "Some files are missing",
                    "These files from the workspace could not be opened:\n\n"
                    + "\n".join(missing[:10]),
                )

        self.tasks.submit("Loading workspace...", work, done)

    def _rebind_state(self) -> None:
        for _name, panel in self._panels:
            panel._state = self.state
        self.plot_toolbar._state = self.state
        self.grid._state = self.state
        self._sync_page_combo()
        # force=True is load-bearing: rebuild_layout() reuses the existing tile
        # widgets when the grid shape is unchanged, and each of those holds its
        # own reference to the *old* AppState. Loading a 1x1 workspace over a
        # 1x1 default therefore left every tile reading and writing an object
        # nobody saves.
        self.grid.rebuild_layout(force=True)
        self.apply_theme()

    # ------------------------------------------------------------- export

    def export_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export page as image", "", "PNG image (*.png)"
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"

        pixmap = self.grid.grab()
        if pixmap.save(path, "PNG"):
            self.status_label.setText("Exported {0}".format(path))
        else:
            QMessageBox.warning(self, "Export failed", "Could not write {0}".format(path))

    def export_csv(self) -> None:
        """Write the active tile's curves to CSV.

        Exports the samples as they are, not the decimated version that is
        drawn: the plot is a picture, the export is data, and silently handing
        someone 1200 of their 500,000 points would be a trap.
        """
        tile = self.state.tile
        if tile.is_empty() or not self.state.plotted_sims():
            QMessageBox.information(self, "Nothing to export",
                                    "The active tile has no curves.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export the active tile as CSV", "", "CSV file (*.csv)"
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"

        def work():
            import csv

            written = 0
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["file", "run", "parameter", tile.x_param, "value"])

                for file_index, sim_index in self.state.plotted_sims():
                    file_handle = self.session.file(file_index)
                    if file_handle is None or sim_index >= file_handle.simulation_count:
                        continue
                    x = self.session.channel(file_index, sim_index, tile.x_param)
                    if x.size == 0:
                        continue
                    sim_id = file_handle.simulations[sim_index].sim_id
                    name = file_handle.name

                    for param in tile.y_params:
                        y = self.session.channel(file_index, sim_index, param)
                        if y.size == 0:
                            continue
                        count = min(x.size, y.size)
                        for i in range(count):
                            writer.writerow(
                                [name, sim_id, param,
                                 "{0:.9g}".format(float(x[i])),
                                 "{0:.9g}".format(float(y[i]))]
                            )
                        written += count
            return written

        def done(written):
            self.status_label.setText(
                "Exported {0:,} samples to {1}".format(written, path)
            )

        self.tasks.submit("Writing CSV...", work, done)

    # ------------------------------------------------------------- close

    def closeEvent(self, event) -> None:
        # `dirty` alone, not `dirty and workspace_path`. Requiring a path meant
        # the prompt appeared only for work that had *already* been saved once
        # -- so an afternoon of pages, tiles and styles built from scratch, the
        # case with the most to lose, was the one case that closed without
        # asking. save_workspace() falls back to a Save As when there is no
        # path, so answering Yes still works from here.
        if self.state.dirty:
            answer = QMessageBox.question(
                self,
                "Unsaved changes",
                "Save the workspace before closing?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.StandardButton.Yes:
                self.save_workspace()

        # A worker thread inside the core must finish before the session is
        # destroyed, or it writes into freed memory.
        self.tasks.wait_for_idle()
        self.session.close()
        super().closeEvent(event)
