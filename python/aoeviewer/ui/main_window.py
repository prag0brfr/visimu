# Author: Paulo R. A. Gilz
"""Main application window."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QProgressBar,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
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
from ..theme import (
    UI_SCALE_MAX,
    UI_SCALE_MIN,
    UI_SCALE_STEP,
    apply_pyqtgraph_theme,
    px,
    set_ui_scale,
    stylesheet,
    theme_for,
    ui_scale,
)
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
    SyncPanel,
    TextPanel,
)
from .plot_grid import PlotGrid
from .plot_toolbar import PlotToolbar
from .sidebar import TabRail
from .tasks import TaskRunner

from .. import hdf5
from ..logging_setup import get_logger

_log = get_logger("window")

FILE_FILTER = (
    "Result files (*.aoe *.dat *.txt *.res *.out *.h5 *.hdf5 *.he5 *.hdf);;"
    "AOE result files (*.aoe *.dat *.txt *.res *.out);;"
    "HDF5 result files (*.h5 *.hdf5 *.he5 *.hdf);;"
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
        self.tasks = TaskRunner(self)
        self.tasks.started.connect(self._on_task_started)
        self.tasks.finished.connect(self._on_task_finished)
        self.tasks.failed.connect(self._on_task_failed)

        # Cached sample data is capped so that opening study after study cannot
        # grow without bound. Evicted columns are re-read on demand, so the only
        # cost of being wrong here is time, never correctness.
        self.memory_limit_bytes = 2 * 1024 * 1024 * 1024

        self.resize(1500, 920)

        self._build_toolbar()
        self._build_central()
        self._build_status_bar()
        self._update_window_title()
        self.apply_theme()

        # Nothing polls the disk. Re-reading a file is something the user asks
        # for, with Reload in the DATA tab -- see reload_files().

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
        # Save and Save As are two buttons rather than one with a modifier,
        # because which file is being written is the thing people get wrong.
        # Both tooltips name the target explicitly, and they are refreshed
        # whenever the current workspace changes.
        self.save_action = add("Save workspace", self.save_workspace, "save", "Ctrl+S")
        self.save_as_action = add("Save workspace as...", self.save_workspace_as,
                                  "save-as", "Ctrl+Shift+S")
        self._update_workspace_actions()

        self.toolbar.addSeparator()

        add("Image", self.export_image, "image",
            tip="Save the active page as a PNG")
        add("CSV", self.export_csv, "table",
            tip="Write the active tile's curves to a CSV file, at full resolution")

        # Fit is not here. It acts on the tiles, so it sits in the page and
        # tile bar with the rest of what does -- but the shortcut has to stay
        # reachable from anywhere in the window, so the action lives on, added
        # to the window rather than to a bar. An action that is on no widget
        # has no shortcut at all.
        self.fit_action = QAction("Fit", self)
        self.fit_action.setShortcut(QKeySequence("Ctrl+0"))
        self.fit_action.setToolTip(
            "Auto-scale every tile — double-click one plot to fit just that one"
            "  (Ctrl+0)")
        self.fit_action.triggered.connect(self.autoscale_all)
        self.addAction(self.fit_action)

        # Interface. Its own group behind its own rule, and labelled, because
        # these are the ones here that change nothing about the data or the
        # view: what is left of the rule acts on the workspace or exports it,
        # and these act on the window itself.
        self.toolbar.addSeparator()
        size_caption = QLabel(" Interface ")
        size_caption.setObjectName("Hint")
        size_caption.setToolTip(
            "The size of the text and the spacing around it, everywhere:\n"
            "the tab rail, the panels, this bar, and the titles, axes and\n"
            "legends on the plots")
        self.toolbar.addWidget(size_caption)
        self.shrink_action = add(
            "Smaller", lambda: self.step_ui_scale(-1), "text-smaller",
            "Ctrl+-", "Smaller text and tighter spacing everywhere")
        self.grow_action = add(
            "Larger", lambda: self.step_ui_scale(+1), "text-larger",
            "Ctrl+=", "Larger text and roomier spacing everywhere")
        self._update_scale_actions()

        # The theme belongs with these rather than at the foot of the tab
        # rail, where it used to be: it selects no panel, and the rail's
        # height is the one thing a short screen is short of. Its icon shows
        # what a press *gives* -- a sun while the dark theme is on -- which is
        # the convention everywhere else this control appears.
        self.theme_action = add("Theme", self.toggle_theme, tip="")
        self._update_theme_action()

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

    def step_ui_scale(self, direction: int) -> None:
        """One step larger or smaller, and repaint everything that measures.

        The scale reaches the interface through three routes and all of them
        have to be taken: the stylesheet, which carries the font sizes and
        every padding; the painted widgets, which measure their own text --
        the tab rail's width is the height of its rotated labels; and the
        plots, whose titles, axes and legends are sized in points handed to
        pyqtgraph. apply_theme() already walks all three, so this sets the
        number and asks for a re-theme.
        """
        before = ui_scale()
        after = set_ui_scale(before + direction * UI_SCALE_STEP)
        if after == before:
            return
        self.state.ui_scale = after
        self.state.dirty = True
        self.apply_theme()
        self._update_scale_actions()
        self.status_label.setText(
            "Interface size {0:.0f}%".format(after * 100.0))

    def _update_scale_actions(self) -> None:
        """Disable whichever end has been reached.

        A button that silently does nothing is worse than one that says it
        cannot: the scale clamps, so without this the last press at either end
        looks like the window failing to respond.
        """
        shrink = getattr(self, "shrink_action", None)
        grow = getattr(self, "grow_action", None)
        if shrink is None or grow is None:
            return
        current = ui_scale()
        shrink.setEnabled(current > UI_SCALE_MIN + 1e-6)
        grow.setEnabled(current < UI_SCALE_MAX - 1e-6)

    def _update_theme_action(self) -> None:
        """Say what the button gives, not what is already on screen.

        A toggle labelled with its current state reads as a description; one
        labelled with its result reads as an offer, which is what a button is.
        """
        action = getattr(self, "theme_action", None)
        if action is None:
            return
        dark = self.state.dark_mode
        action.setText("Light" if dark else "Dark")
        action.setToolTip(
            "Switch to the {0} theme".format("light" if dark else "dark"))
        theme = theme_for(dark)
        action.setIcon(icons.icon("sun" if dark else "moon", theme.text, px(20)))

    def _apply_toolbar_icons(self) -> None:
        theme = theme_for(self.state.dark_mode)
        icons.clear_cache()
        for action, name in getattr(self, "_toolbar_actions", []):
            action.setIcon(icons.icon(name, theme.text, px(20)))
        for mode, name in getattr(self, "_mode_icon_names", {}).items():
            self.mode_actions[mode].setIcon(icons.icon(name, theme.text, px(20)))
        # Not in _toolbar_actions: which icon it wants depends on the theme
        # rather than only on its colour.
        self._update_theme_action()

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
        self.sync_panel = SyncPanel(self.session, self.state)
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
            ("SYNC", self.sync_panel),
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
        # Each panel goes in through a scroll area, and the reason is the same
        # failure the width comment above describes, in the other direction.
        #
        # A plain widget's minimum height is the sum of its children's, and
        # STYLE is a tall panel: four switches, two encodings with their combos
        # and hints, two scrolling tables, and a legend group. Added to the
        # toolbar, the page bar and the status bar, the window's minimum height
        # came out around 950 px -- fine on a 1080-high screen, and taller than
        # the usable area of a 1050-high one once a desktop panel takes its
        # share. The window could then neither be shrunk nor maximised, because
        # a maximise that cannot satisfy the minimum is refused.
        #
        # A scroll area does not propagate its child's minimum height, so the
        # window's minimum stops depending on the tallest panel. Nothing is
        # hidden: the panel keeps its full height and scrolls when the pane is
        # shorter than it, which on any screen that fitted before never
        # happens.
        self._panel_scrollers = []
        for _name, widget in self._panels:
            scroller = QScrollArea()
            scroller.setObjectName("Panel")
            # The viewport is a plain child widget, so the panel background
            # rule keyed on the name has to reach it too or the pane shows the
            # window colour behind a panel that does not fill it.
            scroller.viewport().setObjectName("Panel")
            scroller.setWidget(widget)
            # The panel fills the pane's width rather than keeping its own
            # size hint, so nothing reflows when the splitter moves.
            scroller.setWidgetResizable(True)
            scroller.setFrameShape(QFrame.Shape.NoFrame)
            scroller.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroller.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.panel_stack.addWidget(scroller)
            self._panel_scrollers.append(scroller)

        self.data_panel.add_files_requested.connect(self.open_files)
        self.data_panel.reload_requested.connect(self.reload_files)
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
        # A full redraw, not a restyle: the alignment changes where every
        # sample is drawn, and the curves have to be re-decimated against the
        # new x values.
        self.sync_panel.sync_changed.connect(self._on_sync_changed)
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
        self.grid.tile_params_requested.connect(self._on_tile_params_requested)
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
        self.plot_toolbar.fit_requested.connect(self.autoscale_all)

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
        if widget is self._current_panel():
            self._stale_panels.discard(widget)
            self._refresh_now(widget)
        else:
            self._stale_panels.add(widget)

    def _current_panel(self):
        """The panel on screen.

        Not panel_stack.currentWidget(), which is the scroll area the panel
        sits in. One indirection, in one place, so that everything asking
        "which panel is showing" keeps getting a panel back.
        """
        index = self.panel_stack.currentIndex()
        if 0 <= index < len(self._panels):
            return self._panels[index][1]
        return None

    def _invalidate_panels(self) -> None:
        """Mark every panel stale, refreshing only the one on screen."""
        current = self._current_panel()
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
        if hasattr(self, "toolbar"):
            # The bar's icons are painted at a fixed pixel size, so they are
            # the one part of it the stylesheet cannot reach.
            self.toolbar.setIconSize(QSize(px(18), px(18)))
        if hasattr(self, "rail"):
            self.rail.set_theme(theme)
            # Painted widgets measure their own text, so the stylesheet's new
            # sizes do not reach them: the rail has to re-read the scale and
            # re-measure, or it keeps the width its old labels needed.
            self.rail.refresh_metrics()
        if hasattr(self, "plot_toolbar"):
            self.plot_toolbar.set_theme(theme)
        if hasattr(self, "grid"):
            self.grid.apply_theme(theme)
        if hasattr(self, "toolbar"):
            self._apply_toolbar_icons()

    def toggle_theme(self) -> None:
        self.state.dark_mode = not self.state.dark_mode
        # Saved with the workspace, so changing it is an unsaved change --
        # same as the interface size, and the title's asterisk should say so.
        self.state.dirty = True
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
            # After refresh_all, so the file is drawn before its samples are
            # fetched for the sidecar.
            self._build_pushed_caches([index for index, _count in loaded])
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
        # A file read outside the core has no parse to rewrite, so no sidecar
        # is ever written for one and none should be announced.
        if hdf5.looks_like_hdf5(path):
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
        # A pushed file is not cached here. Its columns are fetched as they are
        # wanted, so writing a sidecar means fetching all of them first -- and
        # doing that inside the open would reinstate exactly the wait the lazy
        # reader removes. It happens afterwards instead, with the file already
        # on screen: see _build_pushed_caches.
        if self.session.is_pushed_index(index):
            return False
        try:
            return bool(self.session.build_cache(index))
        except CoreError:
            return False

    def _build_pushed_caches(self, indices) -> None:
        """Write sidecars for files read outside the core, in the background.

        The point of the whole arrangement: a large HDF5 file opens at once
        because nothing is read until it is drawn, and a moment later -- while
        it is already usable -- its samples are written out in the layout this
        program reads fastest. Every later open of that file then takes the
        cached path and needs the reader for nothing.

        Everything here is best-effort and silent on failure. A sidecar is an
        optimisation the user did not ask for, and the file is already open and
        working without it.
        """
        if not self.session.cache_enabled:
            return
        wanted = []
        for index in indices:
            if not self.session.is_pushed_index(index):
                continue
            if self.session.cache_active(index):
                continue
            try:
                if os.path.getsize(self.session.file(index).path) \
                        < self.CACHE_THRESHOLD_BYTES:
                    continue
            except (OSError, AttributeError):
                continue
            wanted.append(index)
        if not wanted:
            return

        def work():
            done = 0
            for index in wanted:
                try:
                    if not self.session.fill_pushed_completely(index):
                        continue
                    if self.session.build_cache(index):
                        done += 1
                except (CoreError, OSError, ValueError):
                    continue
            return done

        def finished(count):
            if count:
                self.status_label.setText(
                    "Cached {0} file(s) for faster reopening".format(count))
                # Filling every column to write the sidecar left the whole file
                # resident, which for a large one is most of the memory budget
                # spent on samples nothing is drawing. They are on disk now and
                # come back from there in milliseconds, so the cap is applied
                # straight away rather than at the next redraw.
                self._enforce_memory_limit()
                self._update_memory_label()

        self.tasks.submit(
            "Caching {0} file(s) for faster reopening — the file is already "
            "usable...".format(len(wanted)), work, finished)

    def reload_files(self, indices=None) -> None:
        """Re-read files from disk, because the user asked.

        Nothing watches the disk. Until 2.1.1 a timer stat-ed every open file
        every three seconds and put up a dialog when one had changed, which
        was wrong in both directions: it interrupted work that had nothing to
        do with the file, and it could not be trusted anyway, since a solver
        halfway through rewriting its output is a file that has changed and
        must not be read. Reloading is a decision about whether the data on
        disk is finished, and only the person running the solver has it.

        `indices` names the files; an empty list or None means every open one.
        """
        if self.tasks.busy:
            self.status_label.setText(
                "Busy — try Reload again when the current task finishes.")
            return

        targets = [index for index in (indices or range(self.session.file_count))
                   if self.session.file(index) is not None]
        if not targets:
            self.status_label.setText("Nothing to reload — no files are open.")
            return

        # Run names are the user's and the file knows nothing about them, so
        # they are carried across the reload by hand. They travel on each
        # run's own key, which survives a reload now precisely so that this
        # and the styles can be put back on the right runs.
        names = self._collect_display_names()
        paths = {index: self.session.file(index).path for index in targets}

        def work():
            reloaded, failed = [], []
            for index in targets:
                handle = self.session.file(index)
                if handle is None:
                    continue
                try:
                    ok = handle.reload()
                except CoreError as exc:
                    failed.append("{0}: {1}".format(
                        os.path.basename(paths.get(index, "")), exc))
                    continue
                if not ok:
                    failed.append("{0}: could not be re-read".format(
                        os.path.basename(paths.get(index, ""))))
                    continue
                reloaded.append(index)
                # The file on disk is not the one the old sidecar describes,
                # so the reload will have fallen back to parsing the text.
                # Writing the new sidecar here costs the parse the user is
                # already waiting for, rather than the next open.
                self._build_cache_for(paths.get(index, ""), index)
            # Put the run names back, on whatever runs still answer to the
            # keys they were given to.
            self._apply_display_names({
                "file_names": names.get("files", {}),
                "sim_names": names.get("simulations", {}),
            })
            return reloaded, failed

        def done(result):
            reloaded, failed = result
            # Every sample pointer the plots were holding belonged to the old
            # parse. refresh_all() drops the resident signature, re-derives
            # the ISO answer and repaints from scratch, which is exactly what
            # a reload needs.
            self.refresh_all()
            if failed:
                QMessageBox.warning(
                    self, "Some files could not be reloaded",
                    "\n".join(failed[:10]))
            self.status_label.setText(
                "{0} file(s) reloaded from disk.".format(len(reloaded))
                if reloaded else "No file was reloaded.")

        self.tasks.submit(
            "Re-reading {0} file(s) from disk...".format(len(targets)),
            work, done)

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
        #
        # plottable_parameters, not parameters: closing a study leaves its
        # formulas and aliases defined, so the plain list is still non-empty
        # with no file open at all -- and this then "repaired" every tile by
        # pointing its x axis at whichever alias sorted first. Swapping one
        # study for another, which is remove-then-add, lost the x axis of
        # every tile on the way through zero files.
        notes = self.state.resolve_missing_parameters(
            self.session.plottable_parameters)
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
        self.state.apply_parameter_defaults(self.session.plottable_parameters)

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
        # The title carries the workspace name and the unsaved marker, and this
        # is the one place called after everything that can change either.
        self._update_window_title()

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

    def _on_tile_params_requested(self, index: int) -> None:
        """A tile was double-clicked: it is fitted already, now show PARAMS.

        The tile has already been activated by the same gesture, so the panel
        that comes forward is describing the tile just clicked.
        """
        if index != self.state.active_tile:
            self._on_tile_activated(index)
        self._show_panel(self.param_panel)

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

    def _on_sync_changed(self) -> None:
        """The alignment was edited, so the tiles have to be drawn again.

        Through refresh_plots rather than refresh_active_plot: Apply to every
        tile touches the whole page, and the title of each tile changes too.
        """
        self.refresh_plots()
        self._invalidate_panels()

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
        if self._current_panel() is self.inspect_panel:
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

    def _workspace_label(self) -> str:
        """The current workspace as a short name, or an explicit 'unsaved'."""
        path = self.state.workspace_path
        if not path:
            return ""
        return os.path.basename(path)

    def _update_window_title(self) -> None:
        """Name the workspace in the title bar.

        The title is the one place a user can always look to answer "which file
        am I working in", and Save As only makes sense if that question has a
        visible answer. An asterisk marks unsaved changes, by the usual
        convention.
        """
        name = self._workspace_label()
        marker = "*" if self.state.dirty else ""
        if name:
            self.setWindowTitle("{0}{1} — aoeViewer {2}".format(
                marker, name, __version__))
        else:
            self.setWindowTitle("{0}Untitled workspace — aoeViewer {1}".format(
                marker, __version__))

    def _update_workspace_actions(self) -> None:
        """Keep both Save tooltips naming the file they would write."""
        save = getattr(self, "save_action", None)
        save_as = getattr(self, "save_as_action", None)
        if save is None or save_as is None:
            return
        path = self.state.workspace_path
        if path:
            save.setText("Save workspace")
            save.setToolTip(
                "Overwrite {0}\nwith the current files, pages, tiles, styles, "
                "formulas, aliases and ISO filters  (Ctrl+S)".format(path))
        else:
            save.setText("Save workspace")
            save.setToolTip(
                "No workspace file yet — this will ask where to save  (Ctrl+S)")
        save_as.setToolTip(
            "Save to a new workspace file and continue in that one, leaving "
            "{0} as it is  (Ctrl+Shift+S)".format(path or "the current session"))

    def _ask_workspace_path(self, title: str) -> str:
        """Ask for a path to write, defaulting to the current workspace.

        Offering the current path as the starting point matters for Save As:
        the common use is "keep this one and start a variant", and typing the
        whole directory again to do that is busywork.
        """
        suggestion = self.state.workspace_path or ""
        path, _ = QFileDialog.getSaveFileName(
            self, title, suggestion,
            "aoeViewer workspace (*.aoews);;All files (*)"
        )
        if not path:
            return ""
        if not path.lower().endswith(".aoews"):
            path += ".aoews"
        return path

    def _collect_definitions(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """The computed parameters and the aliases, ready to be written.

        Both are asked of the core rather than tracked alongside it, so what is
        saved is what is actually defined. A name resolving to something other
        than itself is an alias; a name the core calls computed is a formula.
        The two are disjoint, and a computed channel is never reported as an
        alias of itself.
        """
        computed: Dict[str, str] = {}
        aliases: Dict[str, str] = {}
        for name in self.session.parameters:
            if self.session.is_computed(name):
                computed[name] = self.session.computed_source(name)
                continue
            target = self.session.resolve(name)
            if target and target != name:
                aliases[name] = target
        return computed, aliases

    def _collect_display_names(self) -> Dict[str, dict]:
        """The labels the DATA tab shows, keyed by things that survive a reload.

        A file is keyed by its path and a run by the core's stable key, never
        by index: a workspace may be opened against files that have grown a run
        or lost one, and a name pinned to position 7 would then land on
        whichever run happens to sit there.

        Runs are collected only where the name differs from the identifier the
        file itself carries: a run keeps that identifier until someone renames
        it, so storing the untouched ones would put a thousand redundant
        entries in the workspace of a thousand-run study for no gain. A file's
        name is always collected, because the automatic one already depends on
        what else is open -- two files of the same basename make the second
        "out.aoe (2)" -- and writing it down is cheaper than reasoning about
        which of them a reader would regenerate.
        """
        files: Dict[str, str] = {}
        simulations: Dict[str, Dict[str, str]] = {}
        for handle in self.session.files:
            key = handle.path
            if not key:
                continue
            if handle.name:
                files[key] = handle.name
            renamed = {
                sim.key: sim.name
                for sim in handle.simulations
                if sim.name and sim.name != sim.sim_id
            }
            if renamed:
                simulations[key] = renamed
        return {"files": files, "simulations": simulations}

    def _apply_display_names(self, data: dict) -> None:
        """Put saved file and run labels back on the freshly opened files.

        Runs on the worker thread, right after the files are added and before
        anything reads their names, so the first paint already shows them.
        Silently skips whatever no longer matches: a workspace naming a run
        that this data does not have is not an error worth a dialog, it is a
        label with nothing to sit on.
        """
        file_names = data.get("file_names")
        sim_names = data.get("sim_names")
        if not isinstance(file_names, dict) and not isinstance(sim_names, dict):
            return
        for handle in self.session.files:
            key = handle.path
            if not key:
                continue
            if isinstance(file_names, dict):
                name = file_names.get(key)
                if isinstance(name, str) and name.strip():
                    handle.rename(name)
            if isinstance(sim_names, dict):
                per_sim = sim_names.get(key)
                if isinstance(per_sim, dict) and per_sim:
                    for sim in handle.simulations:
                        name = per_sim.get(sim.key)
                        if isinstance(name, str) and name.strip():
                            sim.rename(name)

    def _write_workspace(self, path: str) -> bool:
        computed, aliases = self._collect_definitions()
        names = self._collect_display_names()
        try:
            save_workspace(
                path,
                self.state,
                [f.path for f in self.session.files],
                computed,
                aliases,
                names,
            )
        except OSError as exc:
            QMessageBox.warning(self, "Could not save", str(exc))
            return False

        self.state.workspace_path = path
        self.state.dirty = False
        self._update_window_title()
        self._update_workspace_actions()
        counts = []
        if computed:
            counts.append("{0} formula(s)".format(len(computed)))
        if aliases:
            counts.append("{0} alias(es)".format(len(aliases)))
        renamed = sum(len(per_sim) for per_sim in names["simulations"].values())
        if renamed:
            counts.append("{0} renamed run(s)".format(renamed))
        if self.state.iso_states:
            counts.append("{0} ISO filter(s)".format(len(self.state.iso_states)))
        detail = " (including {0})".format(", ".join(counts)) if counts else ""
        self.status_label.setText(
            "Workspace saved to {0}{1}".format(path, detail))
        return True

    def save_workspace(self) -> None:
        """Save to the current workspace file, asking for one if there is none."""
        path = self.state.workspace_path or self._ask_workspace_path("Save workspace")
        if not path:
            return
        self._write_workspace(path)

    def save_workspace_as(self) -> None:
        """Always ask for a path, and adopt it as the current workspace.

        This is how a variant is made: the file that was open is left as it was
        on disk, and everything from here on saves to the new one.
        """
        path = self._ask_workspace_path("Save workspace as")
        if not path:
            return
        self._write_workspace(path)

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

            # Before anything else reads a name: the panels and the legend all
            # ask the session what a file is called.
            self._apply_display_names(data)

            # Aliases first: a formula may be written in terms of one.
            unresolved_aliases = []
            aliases = data.get("aliases", {})
            if isinstance(aliases, dict):
                known = set(self.session.parameters)
                for alias, target in aliases.items():
                    if not isinstance(alias, str) or not isinstance(target, str):
                        continue
                    if not alias.strip() or not target.strip() or alias == target:
                        continue
                    self.session.set_alias(alias, target)
                    if target not in known:
                        # Kept rather than dropped: the file it belongs to may
                        # be opened next, and an alias costs nothing until it
                        # resolves. Reported so an axis that plots nothing has
                        # an explanation.
                        unresolved_aliases.append("{0} -> {1}".format(alias, target))

            # Formulas, in as many passes as it takes. A workspace is a JSON
            # object and object order is not dependency order, so a single pass
            # drops any formula defined in terms of one that has not been
            # installed yet -- silently, since the core rightly refuses an
            # unknown name. Repeating until a pass installs nothing new settles
            # any acyclic chain, and what is left over genuinely cannot be
            # defined against this data.
            failed_formulas = []
            pending = {
                name: expression
                for name, expression in (data.get("computed", {}) or {}).items()
                if isinstance(name, str) and isinstance(expression, str)
            }
            while pending:
                installed = []
                errors = {}
                for name, expression in pending.items():
                    try:
                        self.session.set_computed(name, expression)
                        installed.append(name)
                    except CoreError as exc:
                        errors[name] = str(exc)
                if not installed:
                    failed_formulas = [
                        "{0} = {1}   ({2})".format(name, pending[name],
                                                   errors.get(name, "refused"))
                        for name in sorted(pending)
                    ]
                    break
                for name in installed:
                    pending.pop(name, None)

            return missing, failed_formulas, unresolved_aliases

        def done(report):
            missing, failed_formulas, unresolved_aliases = report
            # Replacing the state is a GUI-side operation and belongs here, not
            # in the worker: the panels and tiles must be rebound to the new one
            # in the same turn of the event loop that starts using it.
            self.state = AppState()
            self.state.load_dict(data)
            self.state.workspace_path = path

            # Before _rebind_state(), which re-themes: the scale has to be in
            # place by the time anything measures itself against it.
            set_ui_scale(self.state.ui_scale)
            self._update_scale_actions()

            self._rebind_state()
            self.refresh_all()
            self._update_window_title()
            self._update_workspace_actions()

            # One dialog for everything the workspace asked for and this data
            # cannot give, rather than three in a row. Nothing here stops the
            # workspace loading: the layout, the styles and whatever else does
            # resolve are all in place by the time this appears.
            notes = []
            if missing:
                notes.append("Files that could not be opened:\n  "
                             + "\n  ".join(missing[:10]))
            if failed_formulas:
                notes.append("Computed parameters that could not be defined "
                             "against this data:\n  "
                             + "\n  ".join(failed_formulas[:10]))
            if unresolved_aliases:
                notes.append("Aliases whose channel is not in the open files "
                             "(kept, in case it returns):\n  "
                             + "\n  ".join(unresolved_aliases[:10]))

            inapplicable = sorted(self.state.iso_inapplicable_params())
            if inapplicable:
                notes.append("ISO filters on parameters none of these files "
                             "carry (kept, and filtering nothing):\n  "
                             + "\n  ".join(inapplicable[:10]))
            empty = sorted(self.state.iso_empty_params())
            if empty:
                notes.append("ISO filters that no run satisfies, so nothing "
                             "passes them:\n  " + "\n  ".join(empty[:10]))

            if notes:
                QMessageBox.information(
                    self,
                    "Workspace loaded with differences",
                    "The workspace loaded, but this data does not match it "
                    "everywhere.\n\n" + "\n\n".join(notes),
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
