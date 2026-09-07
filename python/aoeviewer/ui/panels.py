# Author: Paulo R. A. Gilz
"""Side panels: data tree, parameter picker, computed parameters, ISO filtering.

Each panel is an independent widget that reads the shared AppState and emits a
signal when something changed; the main window decides what to redraw. Keeping
them decoupled is what makes it cheap to add the remaining tabs later.
"""

from __future__ import annotations

from contextlib import contextmanager

from typing import Dict, List, Optional

from PySide6.QtCore import QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..model import (
    DASH_STYLES,
    SYMBOL_LABELS,
    SYMBOLS,
    AppState,
    AreaBand,
    ColorMode,
    IsoState,
    Marker,
    TextNote,
    tile_title,
)
from ..session import CoreError, IsoGroup, Session
from ..theme import theme_for
from .plot_tile import LEGEND_POSITIONS


def _title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("SectionTitle")
    return label


class DataPanel(QWidget):
    """Tree of loaded files and their simulations, with plot checkboxes."""

    selection_changed = Signal()
    files_changed = Signal()
    add_files_requested = Signal()

    def __init__(self, session: Session, state: AppState, parent=None):
        super().__init__(parent)
        self._session = session
        self._state = state
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        buttons = QHBoxLayout()
        add = QPushButton("Add files")
        add.clicked.connect(self.add_files_requested.emit)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_selected)
        clear = QPushButton("Clear all")
        clear.clicked.connect(self._clear_all)
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addWidget(clear)
        layout.addLayout(buttons)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["File / run"])
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemDoubleClicked.connect(lambda *_: self._rename_selected())
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_menu)
        layout.addWidget(self.tree, 1)

        rename = QPushButton("Rename")
        rename.setToolTip("Rename the selected file or run (or double-click it)")
        rename.clicked.connect(self._rename_selected)
        layout.addWidget(rename)

        selection = QHBoxLayout()
        for text, handler in (("Plot all", lambda: self._set_all(True)),
                              ("Plot none", lambda: self._set_all(False))):
            button = QPushButton(text)
            button.clicked.connect(handler)
            selection.addWidget(button)
        layout.addLayout(selection)

    def refresh(self) -> None:
        """Rebuild the tree from the session, preserving check state."""
        self._updating = True
        try:
            # A study of a thousand runs is a thousand tree items. Without
            # these, Qt relayouts and repaints the widget on every single one.
            self.tree.setUpdatesEnabled(False)
            blocked = self.tree.blockSignals(True)
            self.tree.clear()
            for file_handle in self._session.files:
                parent = QTreeWidgetItem(self.tree, [file_handle.name])
                parent.setData(0, Qt.ItemDataRole.UserRole, ("file", file_handle.index))
                # The full path on hover: the displayed name is a basename, and
                # two studies routinely have identically named result files.
                parent.setToolTip(0, file_handle.path)
                parent.setExpanded(True)

                for sim in file_handle.simulations:
                    child = QTreeWidgetItem(parent, [sim.name])
                    child.setData(0, Qt.ItemDataRole.UserRole,
                                  ("sim", file_handle.index, sim.index))
                    tip = file_handle.path
                    if sim.name != sim.sim_id:
                        tip = "{0}\n(named {1} in the file)".format(tip, sim.sim_id)
                    child.setToolTip(0, tip)
                    child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    checked = self._state.is_selected(file_handle.index, sim.index)
                    child.setCheckState(
                        0,
                        Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked,
                    )
        finally:
            self.tree.blockSignals(blocked)
            self.tree.setUpdatesEnabled(True)
            self._updating = False

    def _on_item_changed(self, item: QTreeWidgetItem, _column: int) -> None:
        if self._updating:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data or data[0] != "sim":
            return
        _kind, file_index, sim_index = data
        self._state.set_selected(
            file_index, sim_index, item.checkState(0) == Qt.CheckState.Checked
        )
        self.selection_changed.emit()

    def _set_all(self, on: bool) -> None:
        self._updating = True
        try:
            for i in range(self.tree.topLevelItemCount()):
                parent = self.tree.topLevelItem(i)
                for j in range(parent.childCount()):
                    child = parent.child(j)
                    data = child.data(0, Qt.ItemDataRole.UserRole)
                    if not data or data[0] != "sim":
                        continue
                    child.setCheckState(
                        0,
                        Qt.CheckState.Checked if on else Qt.CheckState.Unchecked,
                    )
                    self._state.set_selected(data[1], data[2], on)
        finally:
            self._updating = False
        self.selection_changed.emit()

    def _remove_selected(self) -> None:
        """Unload the selected *files*.

        Only entries tagged "file" count. A run is tagged
        ("sim", file_index, sim_index), so taking element [1] of every selected
        payload treated a selected run as a request to unload the file it
        belongs to -- select one run of a fifty-run file, click Remove, and all
        fifty disappeared. Runs are shown or hidden with their tick box; there
        is no such thing as unloading one.
        """
        # Remove highest index first so the lower indices stay valid.
        targets = sorted(
            {
                payload[1]
                for payload in (
                    item.data(0, Qt.ItemDataRole.UserRole)
                    for item in self.tree.selectedItems()
                )
                if payload and payload[0] == "file"
            },
            reverse=True,
        )
        if not targets:
            return
        for file_index in targets:
            self._session.remove_file(file_index)
            self._state.drop_file(file_index)
        self.refresh()
        self.files_changed.emit()

    def _clear_all(self) -> None:
        self._session.clear()
        self._state.selected_sims.clear()
        self.refresh()
        self.files_changed.emit()

    # -------------------------------------------------------------- rename

    def _show_menu(self, position) -> None:
        item = self.tree.itemAt(position)
        if item is None:
            return
        menu = QMenu(self)
        menu.addAction("Rename...", self._rename_selected)
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data and data[0] == "file":
            menu.addAction("Copy full path", lambda: self._copy_path(data[1]))
        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _copy_path(self, file_index: int) -> None:
        handle = self._session.file(file_index)
        if handle is not None:
            QApplication.clipboard().setText(handle.path)

    def _rename_selected(self) -> None:
        """Rename a file or a run for display.

        Only the label changes. Identity keys stay as they were, so a renamed
        run keeps its colour and any saved workspace still matches it.
        """
        item = self.tree.currentItem()
        if item is None:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return

        if data[0] == "file":
            handle = self._session.file(data[1])
            if handle is None:
                return
            current = handle.name
            title = "Rename file"
            hint = "Display name for this file:\n{0}".format(handle.path)
        else:
            handle = self._session.file(data[1])
            if handle is None or data[2] >= handle.simulation_count:
                return
            simulation = handle.simulations[data[2]]
            current = simulation.name
            title = "Rename run"
            hint = "Display name for this run (named {0} in the file):".format(
                simulation.sim_id
            )

        name, accepted = QInputDialog.getText(self, title, hint, text=current)
        if not accepted or not name.strip():
            return

        if data[0] == "file":
            handle.rename(name.strip())
        else:
            handle.simulations[data[2]].rename(name.strip())

        self.refresh()
        self.files_changed.emit()


class ParameterPanel(QWidget):
    """Choose the x parameter and the y parameters of the active tile.

    Deliberately only that. Legend, grid, log scale and point markers used to
    live at the bottom of this panel, which meant deciding how a plot *looks*
    was split across two tabs with no principle behind the split. They are all
    in STYLE now, and this panel answers exactly one question: what is on this
    tile.
    """

    tile_changed = Signal()
    alias_requested = Signal(str)

    def __init__(self, session: Session, state: AppState, parent=None):
        super().__init__(parent)
        self._session = session
        self._state = state
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        layout.addWidget(_title("X axis"))
        self.x_combo = QComboBox()
        self.x_combo.currentTextChanged.connect(self._on_x_changed)
        layout.addWidget(self.x_combo)

        layout.addWidget(_title("Y axis"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter (wildcards: AL* )")
        self.filter_edit.textChanged.connect(self._apply_filter)
        layout.addWidget(self.filter_edit)

        self.y_list = QListWidget()
        self.y_list.itemChanged.connect(self._on_y_changed)
        self.y_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.y_list.customContextMenuRequested.connect(self._show_menu)
        layout.addWidget(self.y_list, 1)

        hint = QLabel(
            "Right-click a parameter to rename it or give it an alias. "
            "Legend, grid, log scale and markers are in the STYLE tab."
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

    def refresh(self) -> None:
        self._updating = True
        try:
            params = self._session.parameters
            tile = self._state.tile

            # Names are shown with their unit but carried as data, so the
            # unit is never mistaken for part of the parameter name -- which is
            # exactly the confusion the file format invites.
            current_x = tile.x_param
            self.x_combo.clear()
            for name in params:
                self.x_combo.addItem(self._session.label(name), name)

            index = self.x_combo.findData(current_x)
            if index >= 0:
                self.x_combo.setCurrentIndex(index)
            elif params:
                # Show a sensible default -- a time-like channel when there is
                # one, since that is the x axis of nearly every one of these
                # files -- but do not write it into the tile.
                #
                # A refresh is not a user action. Assigning here meant the same
                # click produced different tile state depending on which tab
                # happened to be open, because panels only refresh when shown:
                # a workspace whose tile plots a channel the open files lack was
                # silently re-pointed at TIME, immediately if PARAMS was
                # visible, or minutes later when you clicked it. Missing
                # parameters are resolved once, deliberately, when the files
                # change -- see AppState.resolve_missing_parameters.
                default = next((p for p in params if p.upper().startswith("TIME")), params[0])
                self.x_combo.setCurrentIndex(max(0, self.x_combo.findData(default)))

            self.y_list.clear()
            for name in params:
                item = QListWidgetItem(self._session.label(name))
                item.setData(Qt.ItemDataRole.UserRole, name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked if name in tile.y_params else Qt.CheckState.Unchecked
                )
                self.y_list.addItem(item)

        finally:
            self._updating = False
        self._apply_filter(self.filter_edit.text())

    # ------------------------------------------------------- context menu

    def _show_menu(self, position) -> None:
        item = self.y_list.itemAt(position)
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole) or item.text()

        menu = QMenu(self)
        alias_action = menu.addAction("Create alias for {0}...".format(name))
        menu.addSeparator()
        x_action = menu.addAction("Use as X axis")
        copy_action = menu.addAction("Copy name")

        chosen = menu.exec(self.y_list.mapToGlobal(position))
        if chosen is None:
            return
        if chosen is alias_action:
            self._create_alias(name)
        elif chosen is x_action:
            index = self.x_combo.findData(name)
            if index >= 0:
                self.x_combo.setCurrentIndex(index)
        elif chosen is copy_action:
            QApplication.clipboard().setText(name)

    def _create_alias(self, target: str) -> None:
        """A second name for a channel, created where the channel is.

        The ALIAS tab can do this too, but by the time you have found the
        parameter in a list of four hundred and decided its name is unhelpful,
        being sent to another tab to find it again is the wrong answer.
        """
        suggestion = target.replace("_", " ").title()
        alias, accepted = QInputDialog.getText(
            self, "Create alias",
            "Show \"{0}\" under this name as well:".format(target),
            text=suggestion,
        )
        if not accepted:
            return
        alias = alias.strip()
        if not alias or alias == target:
            return
        self._session.set_alias(alias, target)
        self.alias_requested.emit(alias)

    def _apply_filter(self, text: str) -> None:
        pattern = text.strip().upper()
        for i in range(self.y_list.count()):
            item = self.y_list.item(i)
            if not pattern:
                item.setHidden(False)
                continue
            # Match the parameter name, not the displayed label: filtering on
            # the label would make "DEG" match every angle in the file, which
            # is not what someone typing a channel name expects.
            name = (item.data(Qt.ItemDataRole.UserRole) or item.text()).upper()
            if "*" in pattern or "?" in pattern:
                import fnmatch
                item.setHidden(not fnmatch.fnmatch(name, pattern))
            else:
                item.setHidden(pattern not in name)

    def _on_x_changed(self, _text: str) -> None:
        if self._updating:
            return
        name = self.x_combo.currentData()
        if not name:
            return
        self._state.tile.x_param = name
        self._state.dirty = True
        self.tile_changed.emit()

    def _on_y_changed(self, _item: QListWidgetItem) -> None:
        if self._updating:
            return
        tile = self._state.tile
        chosen = []
        for i in range(self.y_list.count()):
            item = self.y_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                chosen.append(item.data(Qt.ItemDataRole.UserRole) or item.text())
        tile.y_params = chosen
        self._state.dirty = True
        self.tile_changed.emit()


FORMULA_HELP = """
<h3>Computed parameters</h3>
<p>Define a channel from the ones already loaded. It then behaves exactly like a
channel read from the file: plottable, filterable, usable inside other
formulas.</p>

<h4>Operators</h4>
<table cellpadding="3">
<tr><td><b>+ - * / %</b></td><td>arithmetic; <b>%</b> is remainder</td></tr>
<tr><td><b>^</b></td><td>power, right-associative, so 2^3^2 is 512</td></tr>
<tr><td><b>&lt; &lt;= &gt; &gt;= == !=</b></td><td>comparison, yielding 1.0 or 0.0</td></tr>
<tr><td><b>&amp;&amp; || !</b></td><td>logical and / or / not</td></tr>
</table>

<h4>Element-wise functions</h4>
<p>One sample at a time.</p>
<p><code>abs sqrt sin cos tan asin acos atan exp log log10 floor ceil round
sign deg rad</code><br>
<code>min(a,b) max(a,b) atan2(a,b) pow(a,b) mod(a,b) hypot(a,b)</code><br>
<code>clamp(x, lo, hi)</code> &nbsp; <code>if(condition, a, b)</code></p>
<p><code>min(a,b)</code> and <code>max(a,b)</code> compare two channels sample
by sample. Given a <i>single</i> argument the same names mean something else
&mdash; see reductions below.</p>

<h4>Series functions</h4>
<p>These look at the whole channel, not one sample at a time.</p>
<table cellpadding="3">
<tr><td><code>deriv(Y, X)</code></td>
    <td>dY/dX by central differences</td></tr>
<tr><td><code>integ(Y, X)</code></td>
    <td>cumulative integral of Y over X, starting at zero</td></tr>
<tr><td><code>lowpass(Y, X, fc)</code></td>
    <td>first-order low-pass, cutoff <i>fc</i> in Hz</td></tr>
<tr><td><code>highpass(Y, X, fc)</code></td>
    <td>first-order high-pass, cutoff <i>fc</i> in Hz</td></tr>
<tr><td><code>movavg(Y, N)</code></td>
    <td>centred moving average over N samples</td></tr>
<tr><td><code>delta(Y)</code></td>
    <td>Y minus its first sample</td></tr>
</table>
<p>The filters take the timestep from <i>X</i> at every sample rather than
assuming a fixed rate, so they behave correctly on unevenly sampled data. Pass
the time channel as <i>X</i>.</p>

<h4>Reductions</h4>
<p>These collapse a whole channel to one number, held flat across the run, so it
can be combined with per-sample channels by ordinary arithmetic. Non-finite
samples are ignored.</p>
<table cellpadding="3">
<tr><td><code>min(Y)</code></td><td>smallest sample of the run</td></tr>
<tr><td><code>max(Y)</code></td><td>largest sample of the run</td></tr>
<tr><td><code>mean(Y)</code></td><td>arithmetic mean of the run</td></tr>
</table>
<p>The count of arguments decides the meaning: <code>max(NZ)</code> is the peak
load factor of the run, while <code>max(NZ, NY)</code> is the larger of the two
at each instant. Each run is reduced separately, so <code>NZ / max(NZ)</code>
normalises every run against its own peak.</p>

<h4>Constants and lookup</h4>
<p><code>pi</code>, <code>e</code>, <code>nan</code></p>
<p><code>NAME[COND=value]</code> is the value of <i>NAME</i> at the sample where
<i>COND</i> is closest to <i>value</i> &mdash; a single number, broadcast over
the whole channel. <code>ALPHA[TIME=0.0]</code> is alpha at t=0.</p>
<p>In place of a number, <code>first</code> and <code>last</code> name the ends
of the run: <code>ALPHA[TIME=first]</code> and <code>ALPHA[TIME=last]</code>.
Use these when runs have different lengths and no single number would mean "the
end" for all of them. Leading and trailing gaps are skipped, so these return a
real sample rather than a NaN.</p>

<h4>Examples</h4>
<pre>
LOAD_FACTOR   = NZ * 9.80665
DRIFT         = HEADING - HEADING[TIME=first]
PITCH_RATE    = deriv(THETA, TIME)
DISTANCE      = integ(VTAS, TIME)
ALPHA_SMOOTH  = lowpass(ALPHA, TIME, 2.0)
ALPHA_NOISE   = highpass(ALPHA, TIME, 5.0)
HIGH_MACH     = if(MACH &gt; 0.8, ALPHA, nan)
ENERGY        = 0.5 * MASS * (VX^2 + VY^2)
NZ_NORM       = NZ / max(NZ)
NZ_ENVELOPE   = max(NZ, NY)
ALT_GAINED    = ALTITUDE[TIME=last] - ALTITUDE[TIME=first]
ALPHA_EXCESS  = ALPHA - mean(ALPHA)
</pre>

<h4>Notes</h4>
<ul>
<li>Division by zero gives <code>nan</code>, not infinity: a NaN breaks the line
where the data is bad, while an infinity would destroy the plot's scale.</li>
<li>A formula may reference other computed parameters. Circular definitions are
refused when you create them, not discovered later.</li>
<li>Removing a parameter also removes anything defined in terms of it, and says
which.</li>
</ul>
"""


class FormulaHelpDialog(QDialog):
    """The syntax reference, on demand rather than permanently on screen."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Formula reference")
        self.resize(640, 640)

        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setHtml(FORMULA_HELP)
        browser.setOpenExternalLinks(False)
        layout.addWidget(browser)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class ComputePanel(QWidget):
    """Create and edit computed parameters."""

    parameters_changed = Signal()

    def __init__(self, session: Session, state: AppState, parent=None):
        super().__init__(parent)
        self._session = session
        self._state = state
        self._editing: Optional[str] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.addWidget(_title("Parameter"))
        header.addStretch(1)
        help_button = QPushButton("Help")
        help_button.setToolTip("Operators, functions and examples")
        help_button.clicked.connect(self._show_help)
        header.addWidget(help_button)
        layout.addLayout(header)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Name, e.g. LOAD_FACTOR")
        layout.addWidget(self.name_edit)

        self.expr_edit = QLineEdit()
        self.expr_edit.setPlaceholderText("Expression, e.g. NZ * 9.80665")
        self.expr_edit.textChanged.connect(self._validate)
        self.expr_edit.returnPressed.connect(self._commit)
        layout.addWidget(self.expr_edit)

        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setObjectName("Hint")
        layout.addWidget(self.message)

        buttons = QHBoxLayout()
        self.commit_button = QPushButton("Create")
        self.commit_button.clicked.connect(self._commit)
        self.cancel_button = QPushButton("Cancel edit")
        self.cancel_button.clicked.connect(self._cancel_edit)
        self.cancel_button.setVisible(False)
        buttons.addWidget(self.commit_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)

        layout.addWidget(_title("Defined"))
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda _i: self._start_edit())
        layout.addWidget(self.list, 1)

        actions = QHBoxLayout()
        edit = QPushButton("Edit selected")
        edit.setToolTip("Load it into the fields above (or double-click it)")
        edit.clicked.connect(self._start_edit)
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self._remove)
        actions.addWidget(edit)
        actions.addWidget(remove)
        layout.addLayout(actions)

    # ------------------------------------------------------------- refresh

    def refresh(self) -> None:
        selected = None
        if self.list.currentItem() is not None:
            selected = self.list.currentItem().data(Qt.ItemDataRole.UserRole)

        self.list.clear()
        for name in sorted(self._session.parameters):
            if self._session.is_computed(name):
                item = QListWidgetItem(
                    "{0}  =  {1}".format(name, self._session.computed_source(name))
                )
                item.setData(Qt.ItemDataRole.UserRole, name)
                self.list.addItem(item)
                if name == selected:
                    self.list.setCurrentItem(item)

    def _show_help(self) -> None:
        FormulaHelpDialog(self).exec()

    def _set_message(self, text: str, error: bool = False) -> None:
        self.message.setText(text)
        self.message.setObjectName("Error" if error else "Hint")
        self.message.style().polish(self.message)

    def _validate(self, text: str) -> None:
        """Live feedback while typing, without evaluating any data."""
        text = text.strip()
        if not text:
            self._set_message("")
            return
        ok, error = self._session.validate_formula(text)
        self._set_message("Valid" if ok else error, error=not ok)

    # -------------------------------------------------------------- edit

    def _start_edit(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        self._editing = name
        self.name_edit.setText(name)
        self.name_edit.setEnabled(False)   # renaming would orphan every user
        self.expr_edit.setText(self._session.computed_source(name))
        self.expr_edit.setFocus()
        self.expr_edit.selectAll()
        self.commit_button.setText("Apply")
        self.cancel_button.setVisible(True)
        self._set_message("Editing {0}. Applying re-evaluates it everywhere.".format(name))

    def _cancel_edit(self) -> None:
        self._editing = None
        self.name_edit.clear()
        self.name_edit.setEnabled(True)
        self.expr_edit.clear()
        self.commit_button.setText("Create")
        self.cancel_button.setVisible(False)
        self._set_message("")

    def _commit(self) -> None:
        name = (self._editing or self.name_edit.text()).strip()
        expression = self.expr_edit.text().strip()
        if not name or not expression:
            self._set_message("A name and an expression are both required.", error=True)
            return

        previous = self._session.computed_source(name) if self._editing else ""

        try:
            # set_computed replaces an existing definition in place, so editing
            # keeps the parameter's identity: anything referencing it, and any
            # tile plotting it, follows the new definition without being touched.
            self._session.set_computed(name, expression)
        except CoreError as exc:
            self._set_message(str(exc), error=True)
            return

        was_editing = self._editing is not None
        self._cancel_edit()
        self._set_message(
            "Updated {0}.".format(name) if was_editing else "Created {0}.".format(name)
        )
        del previous
        self.refresh()
        self.parameters_changed.emit()

    def _remove(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        removed = self._session.remove_computed(name)
        if self._editing == name:
            self._cancel_edit()
        if len(removed) > 1:
            others = ", ".join(n for n in removed if n != name)
            self._set_message(
                "Removed {0}, and {1} which depended on it.".format(name, others)
            )
        else:
            self._set_message("Removed {0}.".format(name))
        self.refresh()
        self.parameters_changed.emit()


class IsoPanel(QWidget):
    """Filter runs by one or more constant (ISO) parameters.

    Several parameters can filter at once and they combine with AND: a run is
    shown only if it passes every active filter, which is how you narrow a sweep
    down to "Mach 0.75 and 180 t". Colouring is a separate choice made per tile
    in the STYLE tab, and keys on exactly one parameter.
    """

    filter_changed = Signal()

    def __init__(self, session: Session, state: AppState, parent=None):
        super().__init__(parent)
        self._session = session
        self._state = state
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        layout.addWidget(_title("Filter by"))

        add_row = QHBoxLayout()
        self.param_combo = QComboBox()
        add = QPushButton("Add filter")
        add.clicked.connect(self._add_filter)
        add_row.addWidget(self.param_combo, 1)
        add_row.addWidget(add)
        layout.addLayout(add_row)

        # Active filters. The checkbox turns one off without losing its value
        # selection, which matters when comparing "with and without this."
        self.filter_list = QListWidget()
        self.filter_list.setMaximumHeight(120)
        self.filter_list.currentRowChanged.connect(lambda _r: self._show_values())
        self.filter_list.itemChanged.connect(self._on_filter_toggled)
        layout.addWidget(self.filter_list)

        remove = QPushButton("Remove selected filter")
        remove.clicked.connect(self._remove_filter)
        layout.addWidget(remove)

        layout.addWidget(_title("Values"))

        group_row = QHBoxLayout()
        self.group_check = QCheckBox("Group into")
        self.group_check.toggled.connect(self._on_grouping_changed)
        self.group_spin = QSpinBox()
        self.group_spin.setRange(1, 20)
        self.group_spin.setValue(3)
        self.group_spin.valueChanged.connect(self._on_grouping_changed)
        group_row.addWidget(self.group_check)
        group_row.addWidget(self.group_spin)
        group_row.addWidget(QLabel("intervals"))
        group_row.addStretch(1)
        layout.addLayout(group_row)

        self.value_list = QListWidget()
        self.value_list.itemChanged.connect(self._on_value_toggled)
        layout.addWidget(self.value_list, 1)

        buttons = QHBoxLayout()
        for text, on in (("All", True), ("None", False)):
            button = QPushButton(text)
            button.clicked.connect(lambda _c=False, v=on: self._set_all(v))
            buttons.addWidget(button)
        layout.addLayout(buttons)

        self.summary = QLabel("")
        self.summary.setObjectName("Hint")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

    # ------------------------------------------------------------- refresh

    def refresh(self) -> None:
        self._updating = True
        try:
            current = self.param_combo.currentText()
            self.param_combo.clear()
            available = [
                name for name in self._session.parameters
                if name not in self._state.iso_states
            ]
            self.param_combo.addItems(available)
            if current in available:
                self.param_combo.setCurrentText(current)

            row = self.filter_list.currentRow()
            self.filter_list.clear()
            for name in sorted(self._state.iso_states):
                iso = self._state.iso_states[name]
                item = QListWidgetItem(name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked if iso.active else Qt.CheckState.Unchecked
                )
                item.setData(Qt.ItemDataRole.UserRole, name)
                self.filter_list.addItem(item)
            if 0 <= row < self.filter_list.count():
                self.filter_list.setCurrentRow(row)
            elif self.filter_list.count():
                self.filter_list.setCurrentRow(0)
        finally:
            self._updating = False
        self._show_values()

    def _current_param(self) -> Optional[str]:
        item = self.filter_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _show_values(self) -> None:
        self._updating = True
        try:
            self.value_list.clear()
            param = self._current_param()
            if param is None:
                self.summary.setText("Add a parameter to filter by.")
                self.group_check.setEnabled(False)
                self.group_spin.setEnabled(False)
                return

            self.group_check.setEnabled(True)
            self.group_spin.setEnabled(True)

            iso = self._state.iso_state(param)
            self.group_check.setChecked(iso.grouped)
            self.group_spin.setValue(iso.group_count)

            values = self._session.iso_distinct(param)
            if values.size == 0:
                self.summary.setText("No values found for {0}.".format(param))
                return

            if iso.grouped:
                if not iso.groups:
                    groups = self._session.iso_auto_group(values, iso.group_count)
                    iso.groups = [(g.name, g.low, g.high, True) for g in groups]
                for name, low, high, active in iso.groups:
                    item = QListWidgetItem(name)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(
                        Qt.CheckState.Checked if active else Qt.CheckState.Unchecked
                    )
                    item.setData(Qt.ItemDataRole.UserRole, ("group", low, high))
                    self.value_list.addItem(item)
                self.summary.setText(
                    "{0}: {1} values in {2} intervals.".format(
                        param, values.size, len(iso.groups))
                )
            else:
                for value in values:
                    text = self._session.format_value(float(value))
                    enabled = iso.value_enabled.get(float(value), True)
                    iso.value_enabled[float(value)] = enabled
                    item = QListWidgetItem(text)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(
                        Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked
                    )
                    item.setData(Qt.ItemDataRole.UserRole, ("value", float(value)))
                    self.value_list.addItem(item)
                self.summary.setText(
                    "{0}: {1} distinct values.".format(param, values.size)
                )
        finally:
            self._updating = False

    # ------------------------------------------------------------- actions

    def _add_filter(self) -> None:
        name = self.param_combo.currentText().strip()
        if not name:
            return
        iso = self._state.iso_state(name)
        iso.active = True
        self._state.dirty = True
        self.refresh()
        for row in range(self.filter_list.count()):
            if self.filter_list.item(row).data(Qt.ItemDataRole.UserRole) == name:
                self.filter_list.setCurrentRow(row)
                break
        self.apply_filter()

    def _remove_filter(self) -> None:
        param = self._current_param()
        if param is None:
            return
        self._state.iso_states.pop(param, None)
        self._state.dirty = True
        self.refresh()
        self.apply_filter()

    def _on_filter_toggled(self, item: QListWidgetItem) -> None:
        if self._updating:
            return
        param = item.data(Qt.ItemDataRole.UserRole)
        if param is None:
            return
        self._state.iso_state(param).active = (
            item.checkState() == Qt.CheckState.Checked
        )
        self._state.dirty = True
        self.apply_filter()

    def _on_grouping_changed(self, *_args) -> None:
        if self._updating:
            return
        param = self._current_param()
        if param is None:
            return
        iso = self._state.iso_state(param)
        iso.grouped = self.group_check.isChecked()
        iso.group_count = self.group_spin.value()
        # Regenerate: the interval count changed, so the old bounds are wrong.
        iso.groups = []
        self._state.dirty = True
        self._show_values()
        self.apply_filter()

    def _on_value_toggled(self, item: QListWidgetItem) -> None:
        if self._updating:
            return
        param = self._current_param()
        if param is None:
            return
        iso = self._state.iso_state(param)
        data = item.data(Qt.ItemDataRole.UserRole)
        checked = item.checkState() == Qt.CheckState.Checked
        if data[0] == "value":
            iso.value_enabled[data[1]] = checked
        else:
            _kind, low, high = data
            iso.groups = [
                (name, lo, hi, checked if (lo == low and hi == high) else active)
                for name, lo, hi, active in iso.groups
            ]
        iso.active = True
        self._state.dirty = True
        self.apply_filter()

    def _set_all(self, on: bool) -> None:
        param = self._current_param()
        if param is None:
            return
        iso = self._state.iso_state(param)
        self._updating = True
        try:
            for i in range(self.value_list.count()):
                self.value_list.item(i).setCheckState(
                    Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
                )
            iso.value_enabled = {k: on for k in iso.value_enabled}
            iso.groups = [(n, lo, hi, on) for n, lo, hi, _ in iso.groups]
            iso.active = True
        finally:
            self._updating = False
        self.apply_filter()

    # -------------------------------------------------------------- filter

    def _passes(self, iso: IsoState, value: float) -> bool:
        if value != value:      # NaN: the run has no value for this parameter
            return False
        if iso.grouped and iso.groups:
            index = self._session.iso_group_index(
                value, [IsoGroup(n, lo, hi, a) for n, lo, hi, a in iso.groups]
            )
            return 0 <= index < len(iso.groups) and iso.groups[index][3]
        return iso.value_enabled.get(value, True)

    def apply_filter(self) -> None:
        """Select the runs that satisfy every active filter.

        Combining with AND rather than OR is the useful default: adding a second
        filter should narrow the set, not widen it.
        """
        filters = self._state.active_iso_filters()
        if not filters:
            self.filter_changed.emit()
            return

        for file_handle in self._session.files:
            # One ISO scan per parameter per file; each reads only the first
            # record of every block, so this stays cheap even on large files.
            values = {
                iso.param: self._session.iso_values(file_handle.index, iso.param)
                for iso in filters
            }
            for sim_index in range(file_handle.simulation_count):
                keep = True
                for iso in filters:
                    column = values[iso.param]
                    if sim_index >= column.size:
                        keep = False
                        break
                    if not self._passes(iso, float(column[sim_index])):
                        keep = False
                        break
                self._state.set_selected(file_handle.index, sim_index, keep)

        self.filter_changed.emit()


# ==========================================================================
# Styling
# ==========================================================================


class ColorSwatch(QPushButton):
    """A small button showing a colour, opening a colour picker when clicked."""

    color_picked = Signal(str)

    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(28, 18)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click to change colour")
        self.clicked.connect(self._choose)
        self._apply()

    def _apply(self) -> None:
        self.setStyleSheet(
            "QPushButton {{ background: {0}; border: 1px solid #555; border-radius: 2px; }}"
            "QPushButton:hover {{ border: 1px solid #fff; }}".format(self._color)
        )

    def color(self) -> str:
        return self._color

    def set_color(self, color: str) -> None:
        self._color = color
        self._apply()

    def _choose(self) -> None:
        chosen = QColorDialog.getColor(QColor(self._color), self, "Curve colour")
        if chosen.isValid():
            self.set_color(chosen.name())
            self.color_picked.emit(chosen.name())


class StylePanel(QWidget):
    """Colours, dash styles, markers and widths for the plotted curves.

    A styling mode decides what a colour is
    attached to (a simulation, a parameter, or an ISO value), and the list below
    shows one row per thing in that mode. Because styles are keyed by name
    rather than by list position, they survive loading and removing files.
    """

    style_changed = Signal()
    # How many tiles "Apply legend to all tiles" actually changed. Emitted even
    # when that is zero: a button that does nothing visible has to say so, or it
    # reads as broken.
    legend_applied = Signal(int)

    def __init__(self, session: Session, state: AppState, parent=None):
        super().__init__(parent)
        self._session = session
        self._state = state
        self._updating = False
        # What the curve table currently shows, so a refresh that would produce
        # the same rows can return without touching a widget, and which key
        # each row edits, because the row widgets outlive their keys. See
        # _rebuild_table().
        self._table_signature = None
        self._row_keys: List[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # --- what the tile shows ---------------------------------------------
        # Moved here from PARAMS: these are appearance decisions, and having
        # them in the tab that picks parameters meant every "how does this look"
        # question was answered in two places.
        display_box = QGroupBox("Tile display")
        display_row = QHBoxLayout(display_box)
        self.legend_check = QCheckBox("Legend")
        self.grid_check = QCheckBox("Grid")
        self.logy_check = QCheckBox("Log Y")
        self.points_check = QCheckBox("Points")
        for widget in (self.legend_check, self.grid_check, self.logy_check,
                       self.points_check):
            widget.toggled.connect(self._on_display_changed)
            display_row.addWidget(widget)
        layout.addWidget(display_box)

        # --- mode -----------------------------------------------------------
        layout.addWidget(_title("Colour by"))
        mode_row = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.mode_buttons = []
        for index, label in enumerate(("Simulation", "Parameter", "ISO")):
            radio = QRadioButton(label)
            self.mode_group.addButton(radio, index)
            self.mode_buttons.append(radio)
            mode_row.addWidget(radio)
        self.mode_group.idClicked.connect(self._on_mode_changed)
        layout.addLayout(mode_row)

        self.iso_combo = QComboBox()
        self.iso_combo.currentTextChanged.connect(self._on_iso_param_changed)
        self.iso_row = QWidget()
        iso_layout = QHBoxLayout(self.iso_row)
        iso_layout.setContentsMargins(0, 0, 0, 0)
        iso_layout.addWidget(QLabel("ISO parameter"))
        iso_layout.addWidget(self.iso_combo, 1)
        layout.addWidget(self.iso_row)

        apply_all = QPushButton("Apply colour mode to all tiles")
        apply_all.clicked.connect(self._apply_mode_to_all)
        layout.addWidget(apply_all)

        # --- global sliders -------------------------------------------------
        globals_box = QGroupBox("Global")
        globals_form = QFormLayout(globals_box)

        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.1, 10.0)
        self.width_spin.setSingleStep(0.1)
        self.width_spin.setDecimals(2)
        self.width_spin.valueChanged.connect(self._on_global_width)
        globals_form.addRow("Line width", self.width_spin)

        self.symbol_spin = QDoubleSpinBox()
        self.symbol_spin.setRange(1.0, 20.0)
        self.symbol_spin.setSingleStep(0.5)
        self.symbol_spin.setDecimals(1)
        self.symbol_spin.valueChanged.connect(self._on_symbol_size)
        globals_form.addRow("Marker size", self.symbol_spin)

        layout.addWidget(globals_box)

        # --- per-curve table ------------------------------------------------
        layout.addWidget(_title("Curves"))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["", "Colour", "Line", "Marker"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        recolor = QPushButton("Re-assign palette")
        recolor.setToolTip("Give every curve a fresh colour from the palette")
        recolor.clicked.connect(self._reassign_palette)
        reset_width = QPushButton("Reset widths")
        reset_width.setToolTip("Drop per-curve widths so all follow the global setting")
        reset_width.clicked.connect(self._reset_widths)
        buttons.addWidget(recolor)
        buttons.addWidget(reset_width)
        layout.addLayout(buttons)

        # --- legend contents ------------------------------------------------
        legend_box = QGroupBox("Legend")
        legend_form = QVBoxLayout(legend_box)

        position_row = QHBoxLayout()
        position_row.addWidget(QLabel("Corner"))
        self.legend_position = QComboBox()
        self.legend_position.addItems(list(LEGEND_POSITIONS))
        self.legend_position.currentTextChanged.connect(self._on_legend_position)
        position_row.addWidget(self.legend_position, 1)
        legend_form.addLayout(position_row)

        legend_drag_hint = QLabel(
            "Drag the legend on the plot to put it anywhere; the corner "
            "switches to \"custom\" and the position is kept, including in "
            "saved workspaces. Pick a corner here to snap it back."
        )
        legend_drag_hint.setObjectName("Hint")
        legend_drag_hint.setWordWrap(True)
        legend_form.addWidget(legend_drag_hint)

        # One row rather than four stacked. What goes into a legend entry is a
        # single decision and reads better side by side, and the three lines it
        # frees are what the Apply button below now occupies.
        #
        # "ISO" rather than "ISO value" so the row still fits the panel at its
        # narrowest (300 px): four checkboxes with the longer label came to just
        # over that, and a clipped tick box is worse than a short name. The word
        # is unambiguous here -- the ISO parameter is chosen a few rows above.
        parts_row = QHBoxLayout()
        parts_row.setSpacing(8)
        self.legend_checks = {}
        for field, label, tip in (
            ("legend_show_param", "Parameter", "The parameter's name"),
            ("legend_show_iso", "ISO", "The ISO parameter and this run's value"),
            ("legend_show_file", "File", "The file the run came from"),
            ("legend_show_sim", "Simulation", "The run's own identifier"),
        ):
            check = QCheckBox(label)
            check.setToolTip(tip)
            check.toggled.connect(self._on_legend_changed)
            parts_row.addWidget(check)
            self.legend_checks[field] = check
        parts_row.addStretch(1)
        legend_form.addLayout(parts_row)

        legend_hint = QLabel(
            "With File and Simulation off, identical entries are merged — so in "
            "ISO mode you get one legend row per ISO value rather than one per run."
        )
        legend_hint.setObjectName("Hint")
        legend_hint.setWordWrap(True)
        legend_form.addWidget(legend_hint)

        apply_legend = QPushButton("Apply legend to all tiles")
        apply_legend.setToolTip(
            "Give every tile, on every page, this tile's legend: whether it is "
            "shown, its corner or dragged position, and which parts make up a row"
        )
        apply_legend.clicked.connect(self._apply_legend_to_all)
        legend_form.addWidget(apply_legend)

        layout.addWidget(legend_box)

    # ------------------------------------------------------------- refresh

    def refresh(self) -> None:
        self._updating = True
        try:
            tile = self._state.tile
            styles = self._state.styles

            self.legend_check.setChecked(tile.show_legend)
            self.grid_check.setChecked(tile.show_grid)
            self.logy_check.setChecked(tile.log_y)
            self.points_check.setChecked(tile.show_points)

            button = self.mode_group.button(int(tile.color_mode))
            if button is not None:
                button.setChecked(True)
            # A bold label as well as the indicator. Two independent signals
            # for the same fact means it survives any platform style that
            # decides to draw the indicator its own way.
            for index, radio in enumerate(self.mode_buttons):
                font = radio.font()
                font.setBold(index == int(tile.color_mode))
                radio.setFont(font)
            self.iso_row.setVisible(tile.color_mode == ColorMode.ISO)

            params = self._session.parameters
            self.iso_combo.clear()
            self.iso_combo.addItems(params)
            if tile.color_iso_param in params:
                self.iso_combo.setCurrentText(tile.color_iso_param)
            elif params:
                # Displayed, not assigned: a refresh must not edit the tile.
                # Prefer a parameter the user has already set up as an ISO.
                configured = [p for p in self._state.iso_states if p in params]
                self.iso_combo.setCurrentText(configured[0] if configured else params[0])

            self.width_spin.setValue(styles.line_width)
            self.symbol_spin.setValue(styles.symbol_size)

            for field, check in self.legend_checks.items():
                check.setChecked(bool(getattr(tile, field)))

            if tile.legend_position in LEGEND_POSITIONS:
                self.legend_position.setCurrentText(tile.legend_position)

            self._rebuild_table()
        finally:
            self._updating = False

    def _style_entries(self):
        """(display label, style key) for everything styleable in this mode."""
        tile = self._state.tile
        entries = []

        if tile.color_mode == ColorMode.PARAMETER:
            for param in tile.y_params:
                entries.append((param, "param::" + param))

        elif tile.color_mode == ColorMode.ISO and tile.color_iso_param:
            iso = self._state.iso_states.get(tile.color_iso_param)
            if iso is not None and iso.grouped and iso.groups:
                for name, _low, _high, active in iso.groups:
                    if active:
                        entries.append((name, "iso::{0}::{1}".format(tile.color_iso_param, name)))
            else:
                for value in self._session.iso_distinct(tile.color_iso_param):
                    label = self._session.format_value(float(value))
                    if iso is not None and not iso.value_enabled.get(float(value), True):
                        continue
                    entries.append((label, "iso::{0}::{1}".format(tile.color_iso_param, label)))

        else:  # simulation
            for file_index, sim_index in sorted(self._state.selected_sims):
                handle = self._session.file(file_index)
                if handle is None or sim_index >= handle.simulation_count:
                    continue
                sim = handle.simulations[sim_index]
                entries.append(("{0} · {1}".format(sim.sim_id, handle.name), "sim::" + sim.key))

        return entries

    def _rebuild_table(self) -> None:
        """Fill the per-curve table, reusing the rows already in it.

        This runs on every refresh, and a refresh happens on every click that
        changes the active tile. Rebuilt from scratch it costs four widgets per
        curve -- a colour button and two populated combo boxes -- created and
        the previous four destroyed. In simulation mode there is one row per
        selected run, so a study of a few hundred runs meant creating and
        destroying a four-figure number of widgets to answer a click that, most
        of the time, changed nothing in this table at all.

        Two things fix that. The signature check returns immediately when the
        rows would come out identical, which is the common case; and when they
        would not, the existing widgets are refilled in place instead of being
        replaced. Binding the signals to the row rather than to the key is what
        makes reuse possible -- the widget outlives the key it was made for, so
        the key has to be looked up when the signal fires, from _row_keys.
        """
        entries = self._style_entries()
        styles = self._state.styles

        signature = tuple(
            (label, key, styles.color_for(key), styles.dash_for(key),
             styles.symbol_for(key))
            for label, key in entries
        )
        if signature == self._table_signature:
            return
        self._table_signature = signature
        self._row_keys = [key for _label, key in entries]

        # Without this Qt relayouts and repaints the table on every single cell.
        self.table.setUpdatesEnabled(False)
        try:
            if not entries:
                # Drop to zero first: that destroys the cell widgets of any rows
                # still there, which setRowCount(1) alone would leave behind
                # under the hint.
                self.table.setRowCount(0)
                self.table.setRowCount(1)
                hint = QTableWidgetItem("Nothing plotted yet")
                hint.setFlags(Qt.ItemFlag.NoItemFlags)
                self.table.setItem(0, 0, hint)
                return

            self.table.setRowCount(len(entries))
            for row, (label, key) in enumerate(entries):
                self._fill_row(row, label, key, styles)
        finally:
            self.table.setUpdatesEnabled(True)

    def _fill_row(self, row: int, label: str, key: str, styles) -> None:
        """Put one curve's styling into a table row, making widgets only if the
        row does not have them yet."""
        item = self.table.item(row, 0)
        if item is None:
            item = QTableWidgetItem()
            self.table.setItem(row, 0, item)
        # A row left over from the "Nothing plotted yet" hint is unselectable
        # until its flags are put back.
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        item.setText(label)
        item.setToolTip(label)

        swatch = self.table.cellWidget(row, 1)
        if swatch is None:
            swatch = ColorSwatch(styles.color_for(key))
            swatch.color_picked.connect(
                lambda color, r=row: self._set_color(self._key_for_row(r), color)
            )
            self.table.setCellWidget(row, 1, swatch)
        else:
            swatch.set_color(styles.color_for(key))

        dash = self.table.cellWidget(row, 2)
        if dash is None:
            dash = QComboBox()
            dash.addItems(list(DASH_STYLES))
            dash.currentTextChanged.connect(
                lambda value, r=row: self._set_dash(self._key_for_row(r), value)
            )
            self.table.setCellWidget(row, 2, dash)
        # _updating is already set by refresh(); setting it here as well keeps
        # this correct if the table is ever refilled from anywhere else, since
        # setCurrentText fires currentTextChanged like a user edit would.
        was_updating = self._updating
        self._updating = True
        try:
            dash.setCurrentText(styles.dash_for(key))

            symbol = self.table.cellWidget(row, 3)
            if symbol is None:
                symbol = QComboBox()
                for code in SYMBOLS:
                    symbol.addItem(SYMBOL_LABELS.get(code, code), code)
                symbol.currentIndexChanged.connect(
                    lambda _i, r=row, combo=symbol:
                    self._set_symbol(self._key_for_row(r), combo.currentData())
                )
                self.table.setCellWidget(row, 3, symbol)
            symbol.setCurrentIndex(max(0, symbol.findData(styles.symbol_for(key))))
        finally:
            self._updating = was_updating

    # ------------------------------------------------------------- actions

    def _on_display_changed(self, *_args) -> None:
        if self._updating:
            return
        tile = self._state.tile
        tile.show_legend = self.legend_check.isChecked()
        tile.show_grid = self.grid_check.isChecked()
        tile.log_y = self.logy_check.isChecked()
        tile.show_points = self.points_check.isChecked()
        self._state.dirty = True
        self.style_changed.emit()

    def _key_for_row(self, row: int) -> str:
        """The styling key a table row currently edits.

        Rows are reused across refreshes, so the widget in a row outlives the
        key it was first built for and the key has to be looked up when the
        signal fires. Out of range means the row has been dropped since; the
        setters below treat the empty string as "nothing to do".
        """
        if 0 <= row < len(self._row_keys):
            return self._row_keys[row]
        return ""

    def _set_color(self, key: str, color: str) -> None:
        if not key:
            return
        self._state.styles.set_color(key, color)
        self._state.dirty = True
        self.style_changed.emit()

    def _set_dash(self, key: str, dash: str) -> None:
        if self._updating or not key:
            return
        self._state.styles.set_dash(key, dash)
        self._state.dirty = True
        self.style_changed.emit()

    def _set_symbol(self, key: str, symbol) -> None:
        if self._updating or not key:
            return
        self._state.styles.set_symbol(key, symbol or "")
        self._state.dirty = True
        self.style_changed.emit()

    def _on_mode_changed(self, index: int) -> None:
        if self._updating:
            return
        self._state.tile.color_mode = ColorMode(index)
        self._state.dirty = True
        self.refresh()
        self.style_changed.emit()

    def _on_iso_param_changed(self, name: str) -> None:
        if self._updating:
            return
        self._state.tile.color_iso_param = name
        self._state.dirty = True
        self._rebuild_table()
        self.style_changed.emit()

    def _apply_mode_to_all(self) -> None:
        tile = self._state.tile
        for page in self._state.pages:
            for other in page.tiles:
                other.color_mode = tile.color_mode
                other.color_iso_param = tile.color_iso_param
        self._state.dirty = True
        self.style_changed.emit()

    def _on_global_width(self, value: float) -> None:
        if self._updating:
            return
        self._state.styles.line_width = float(value)
        self._state.dirty = True
        self.style_changed.emit()

    def _on_symbol_size(self, value: float) -> None:
        if self._updating:
            return
        self._state.styles.symbol_size = float(value)
        self._state.dirty = True
        self.style_changed.emit()

    def _reassign_palette(self) -> None:
        for _label, key in self._style_entries():
            self._state.styles.reset(key)
        self._rebuild_table()
        self._state.dirty = True
        self.style_changed.emit()

    def _reset_widths(self) -> None:
        self._state.styles.clear_width_overrides()
        self._state.dirty = True
        self.style_changed.emit()

    def _on_legend_position(self, value: str) -> None:
        if self._updating:
            return
        if value == "custom":
            # Choosing "custom" by hand means "leave it where it is"; there is
            # nothing sensible to move it to, and resetting to a corner would
            # be the opposite of what was asked.
            return
        self._state.tile.legend_position = value
        self._state.dirty = True
        self.style_changed.emit()

    def _on_legend_changed(self, _checked: bool) -> None:
        if self._updating:
            return
        tile = self._state.tile
        for field, check in self.legend_checks.items():
            setattr(tile, field, check.isChecked())
        self._state.dirty = True
        self.style_changed.emit()

    def _apply_legend_to_all(self) -> None:
        """Copy this tile's legend onto every other tile, on every page.

        The copying itself is in AppState.apply_legend_everywhere(), not here:
        which fields make up "the legend" is a fact about the model, it must not
        drift from what a workspace saves, and it is worth testing without a
        running Qt.
        """
        count = self._state.apply_legend_everywhere(self._state.tile)
        # Redraw first, report second, and the order is load-bearing: the
        # redraw ends by rewriting the status bar with the file and run counts,
        # so a message posted before it would be overwritten before it was read.
        if count:
            self.style_changed.emit()
        self.legend_applied.emit(count)


# ==========================================================================
# Axis scaling
# ==========================================================================


class ScalePanel(QWidget):
    """Fix the displayed range of either axis, per tile or across a variable.

    Unlocked, the four boxes track whatever the plot is currently showing, so
    locking is a one-click "freeze what I am looking at". Locked, they are the
    range. That live-follow behaviour is what makes the feature usable: you
    zoom to the region you care about,
    tick Lock, and every subsequent redraw keeps it.
    """

    scale_changed = Signal()

    def __init__(self, session: Session, state: AppState, parent=None):
        super().__init__(parent)
        self._session = session
        self._state = state
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.tile_label = QLabel("")
        self.tile_label.setObjectName("Hint")
        self.tile_label.setWordWrap(True)
        layout.addWidget(self.tile_label)

        # --- X --------------------------------------------------------------
        x_box = QGroupBox("X axis")
        x_form = QFormLayout(x_box)
        self.lock_x = QCheckBox("Lock X range")
        self.lock_x.toggled.connect(self._on_lock_changed)
        x_form.addRow(self.lock_x)

        self.x_min = self._make_spin()
        self.x_max = self._make_spin()
        x_form.addRow("X min", self.x_min)
        x_form.addRow("X max", self.x_max)
        layout.addWidget(x_box)

        apply_x = QPushButton("Apply X range to every tile with the same X axis")
        apply_x.setToolTip(
            "Gives every plot against this x parameter, on every page, the same "
            "horizontal scale — so they can be compared directly."
        )
        apply_x.clicked.connect(self._apply_x_everywhere)
        layout.addWidget(apply_x)

        # --- Y --------------------------------------------------------------
        y_box = QGroupBox("Y axis")
        y_form = QFormLayout(y_box)
        self.lock_y = QCheckBox("Lock Y range")
        self.lock_y.toggled.connect(self._on_lock_changed)
        y_form.addRow(self.lock_y)

        self.y_min = self._make_spin()
        self.y_max = self._make_spin()
        y_form.addRow("Y min", self.y_min)
        y_form.addRow("Y max", self.y_max)
        layout.addWidget(y_box)

        apply_y = QPushButton("Apply Y range to every tile sharing a parameter")
        apply_y.setToolTip(
            "Gives every plot showing any of this tile's y parameters the same "
            "vertical scale."
        )
        apply_y.clicked.connect(self._apply_y_everywhere)
        layout.addWidget(apply_y)

        reset = QPushButton("Unlock both and auto-fit")
        reset.setToolTip(
            "Unlocks this tile and releases any variable-wide constraint on "
            "its parameters"
        )
        reset.clicked.connect(self._reset)
        layout.addWidget(reset)

        # --- variable-wide constraints ---------------------------------------
        self.global_box = QGroupBox("Constrained variables")
        global_layout = QVBoxLayout(self.global_box)
        self.global_list = QListWidget()
        self.global_list.setMaximumHeight(110)
        global_layout.addWidget(self.global_list)
        clear_global = QPushButton("Clear all variable constraints")
        clear_global.clicked.connect(self._clear_global)
        global_layout.addWidget(clear_global)
        layout.addWidget(self.global_box)

        hint = QLabel(
            "While unlocked, these boxes follow the plot as you zoom and pan. "
            "Tick Lock to freeze the current view. The two Apply buttons attach "
            "the range to the variable itself, so a tile that gets that "
            "parameter later is scaled the same way."
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        layout.addStretch(1)

    def _make_spin(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        # Result files carry everything from 1e-4 rad to 1e5 ft, so the range
        # has to be wide and the precision generous.
        spin.setRange(-1e12, 1e12)
        spin.setDecimals(6)
        spin.setSingleStep(0.1)
        spin.setKeyboardTracking(False)
        spin.valueChanged.connect(self._on_value_changed)
        return spin

    # ------------------------------------------------------------- refresh

    def refresh(self) -> None:
        self._updating = True
        try:
            tile = self._state.tile
            self.tile_label.setText(
                "Tile {0}:  {1}".format(self._state.active_tile + 1, tile_title(tile))
            )

            self.lock_x.setChecked(tile.lock_x)
            self.lock_y.setChecked(tile.lock_y)

            self.x_min.setValue(tile.x_min)
            self.x_max.setValue(tile.x_max)
            self.y_min.setValue(tile.y_min)
            self.y_max.setValue(tile.y_max)

            for spin, locked in (
                (self.x_min, tile.lock_x), (self.x_max, tile.lock_x),
                (self.y_min, tile.lock_y), (self.y_max, tile.lock_y),
            ):
                spin.setEnabled(locked)

            self.global_list.clear()
            for axis, table in (("X", self._state.global_x_limits),
                                ("Y", self._state.global_y_limits)):
                for name in sorted(table):
                    low, high = table[name]
                    self.global_list.addItem(
                        "{0}  {1}: [{2:.6g}, {3:.6g}]".format(axis, name, low, high)
                    )
            if self.global_list.count() == 0:
                self.global_list.addItem("none")
        finally:
            self._updating = False

    def follow_view(self) -> None:
        """Cheap update while the user pans or zooms an unlocked axis."""
        if self._updating or not self.isVisible():
            return
        self._updating = True
        try:
            tile = self._state.tile
            if not tile.lock_x:
                self.x_min.setValue(tile.x_min)
                self.x_max.setValue(tile.x_max)
            if not tile.lock_y:
                self.y_min.setValue(tile.y_min)
                self.y_max.setValue(tile.y_max)
        finally:
            self._updating = False

    # ------------------------------------------------------------- actions

    def _on_lock_changed(self, _checked: bool) -> None:
        if self._updating:
            return
        tile = self._state.tile
        was_locked_x = tile.lock_x
        tile.lock_x = self.lock_x.isChecked()
        tile.lock_y = self.lock_y.isChecked()
        # Pinning x is a request to look at that window, so the free y axis is
        # refitted to what falls inside it rather than to the whole channel.
        if tile.lock_x and not was_locked_x and not tile.lock_y:
            tile.view_initialised = False
        self._state.dirty = True
        self.refresh()
        self.scale_changed.emit()

    def _on_value_changed(self, _value: float) -> None:
        if self._updating:
            return
        tile = self._state.tile
        # A min above its max would make the plot draw nothing at all, so the
        # pair is normalised rather than accepted as typed.
        x_low, x_high = self.x_min.value(), self.x_max.value()
        y_low, y_high = self.y_min.value(), self.y_max.value()
        changed_x = (tile.x_min, tile.x_max) != (min(x_low, x_high), max(x_low, x_high))
        tile.x_min, tile.x_max = min(x_low, x_high), max(x_low, x_high)
        tile.y_min, tile.y_max = min(y_low, y_high), max(y_low, y_high)
        # Typing a new x window is an explicit instruction about what to look
        # at, and it is one of the few things that may move the frame -- an
        # ordinary redraw no longer does.
        if changed_x and tile.lock_x and not tile.lock_y:
            tile.view_initialised = False
        self._state.dirty = True
        self.scale_changed.emit()

    def _apply_x_everywhere(self) -> None:
        """Constrain the variable, not just the tiles that exist right now.

        Recording it against the parameter is the whole point: the previous
        version walked the current tiles and stopped, so a tile you filled in
        afterwards -- which is the normal order of work, scale first then
        populate -- silently ignored the constraint.
        """
        tile = self._state.tile
        if not tile.x_param:
            return
        self._state.set_global_x_limit(tile.x_param, tile.x_min, tile.x_max)
        for page in self._state.pages:
            for other in page.tiles:
                if other.x_param == tile.x_param:
                    other.lock_x = True
                    other.x_min, other.x_max = tile.x_min, tile.x_max
                    # Refit each tile's y inside the new window.
                    if not other.lock_y:
                        other.view_initialised = False
        self._state.dirty = True
        self.refresh()
        self.scale_changed.emit()

    def _apply_y_everywhere(self) -> None:
        tile = self._state.tile
        if not tile.y_params:
            return
        for param in tile.y_params:
            self._state.set_global_y_limit(param, tile.y_min, tile.y_max)
        wanted = set(tile.y_params)
        for page in self._state.pages:
            for other in page.tiles:
                if wanted.intersection(other.y_params):
                    other.lock_y = True
                    other.y_min, other.y_max = tile.y_min, tile.y_max
        self._state.dirty = True
        self.refresh()
        self.scale_changed.emit()

    def _clear_global(self) -> None:
        """Drop every variable-wide constraint, leaving per-tile locks alone."""
        self._state.global_x_limits.clear()
        self._state.global_y_limits.clear()
        self._state.dirty = True
        self.refresh()
        self.scale_changed.emit()

    def _reset(self) -> None:
        tile = self._state.tile
        # A tile carrying a constrained variable would be re-locked on the next
        # redraw, so unlocking it has to release that variable too -- otherwise
        # the button appears to do nothing.
        self._state.global_x_limits.pop(tile.x_param, None)
        for param in tile.y_params:
            self._state.global_y_limits.pop(param, None)
        tile.lock_x = False
        tile.lock_y = False
        # Releasing every constraint is a request to see the data again, so this
        # is one of the deliberate moments the frame is allowed to move.
        tile.view_initialised = False
        self._state.dirty = True
        self.refresh()
        self.scale_changed.emit()


# ==========================================================================
# Areas and vertical lines
# ==========================================================================


class _AnnotationPanelBase(QWidget):
    """Shared scaffolding for the area and marker lists.

    Both are global: an annotation is attached to an x parameter, not to a tile,
    so it appears on every plot drawn against that parameter across every page.
    That is the useful behaviour — a takeoff
    marker at t = 12 s belongs on every time plot at once.
    """

    changed = Signal()

    def __init__(self, session: Session, state: AppState, title: str, parent=None):
        super().__init__(parent)
        self._session = session
        self._state = state
        self._updating = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        layout.addWidget(_title(title))

        add = QPushButton("Add")
        add.clicked.connect(self._add)
        layout.addWidget(add)

        visibility = QHBoxLayout()
        for text, on in (("Show all", True), ("Hide all", False)):
            button = QPushButton(text)
            button.clicked.connect(lambda _c=False, v=on: self._set_all_visible(v))
            visibility.addWidget(button)
        layout.addLayout(visibility)

        self.list = QListWidget()
        self.list.currentRowChanged.connect(lambda _r: self._refresh_editor())
        layout.addWidget(self.list, 1)

        self.editor = QGroupBox("Selected")
        self.form = QFormLayout(self.editor)
        self._build_editor(self.form)
        layout.addWidget(self.editor)

        remove = QPushButton("Remove selected")
        remove.clicked.connect(self._remove)
        layout.addWidget(remove)

        hint = QLabel(
            "Global: applies to every page. A marker appears as a line on tiles "
            "using its parameter as the x axis, and as a dot on every other tile."
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

    # -- subclass hooks
    def _items(self):
        raise NotImplementedError

    def _build_editor(self, form: QFormLayout) -> None:
        raise NotImplementedError

    def _add(self) -> None:
        raise NotImplementedError

    def _describe(self, item) -> str:
        raise NotImplementedError

    def _load_editor(self, item) -> None:
        raise NotImplementedError

    # -- shared behaviour
    def _current(self):
        row = self.list.currentRow()
        items = self._items()
        return items[row] if 0 <= row < len(items) else None

    def _default_x_param(self) -> str:
        tile = self._state.tile
        if tile.x_param:
            return tile.x_param
        params = self._session.parameters
        return params[0] if params else ""

    @contextmanager
    def _suspended(self):
        """Ignore widget signals for the duration.

        A counter, not a flag, and that is the whole point. As a flag this was
        a real bug: refresh() set it, then called setCurrentRow(), which fired
        currentRowChanged, which ran _refresh_editor(), whose own `finally`
        cleared the flag -- while refresh() was still going. The parameter
        combo was then repopulated *unguarded*, its currentTextChanged fired,
        and the selected marker or area quietly adopted whichever parameter
        happened to land first in the list.

        Which is exactly what you saw: click a tile plotted against something
        else, every panel refreshes, and the annotation vanishes from the plots
        it belonged to -- because it was no longer attached to the parameter
        you gave it.
        """
        self._updating += 1
        try:
            yield
        finally:
            self._updating -= 1

    def refresh(self) -> None:
        with self._suspended():
            row = self.list.currentRow()
            self.list.clear()
            for item in self._items():
                entry = QListWidgetItem(self._describe(item))
                if not item.visible:
                    entry.setForeground(Qt.GlobalColor.gray)
                self.list.addItem(entry)
            if 0 <= row < self.list.count():
                self.list.setCurrentRow(row)
            elif self.list.count():
                self.list.setCurrentRow(0)

            # Blocked as well as guarded: refilling a combo emits
            # currentTextChanged, and an annotation must never be re-pointed by
            # the act of listing the parameters that exist.
            self.x_param_combo.blockSignals(True)
            try:
                previous = self.x_param_combo.currentText()
                self.x_param_combo.clear()
                self.x_param_combo.addItems(self._session.parameters)
                index = self.x_param_combo.findText(previous)
                if index >= 0:
                    self.x_param_combo.setCurrentIndex(index)
            finally:
                self.x_param_combo.blockSignals(False)
        self._refresh_editor()

    def _refresh_editor(self) -> None:
        item = self._current()
        self.editor.setEnabled(item is not None)
        if item is None:
            return
        with self._suspended():
            self._load_editor(item)

    def _set_all_visible(self, visible: bool) -> None:
        for item in self._items():
            item.visible = visible
        self._state.dirty = True
        self.refresh()
        self.changed.emit()

    def _remove(self) -> None:
        row = self.list.currentRow()
        items = self._items()
        if 0 <= row < len(items):
            items.pop(row)
            self._state.dirty = True
            self.refresh()
            self.changed.emit()

    def _commit(self) -> None:
        if self._updating:
            return
        self._state.dirty = True
        self._update_current_label()
        self.changed.emit()

    def _update_current_label(self) -> None:
        item = self._current()
        row = self.list.currentRow()
        if item is not None and 0 <= row < self.list.count():
            entry = self.list.item(row)
            entry.setText(self._describe(item))
            entry.setForeground(
                Qt.GlobalColor.gray if not item.visible else Qt.GlobalColor.white
            )


class AreaPanel(_AnnotationPanelBase):
    """Shaded x bands, for marking phases of a manoeuvre."""

    def __init__(self, session: Session, state: AppState, parent=None):
        super().__init__(session, state, "Shaded areas", parent)

    def _items(self):
        return self._state.areas

    def _build_editor(self, form: QFormLayout) -> None:
        self.visible_check = QCheckBox("Visible")
        self.visible_check.toggled.connect(self._commit_visible)
        form.addRow(self.visible_check)

        self.x_param_combo = QComboBox()
        self.x_param_combo.currentTextChanged.connect(self._commit_x_param)
        form.addRow("X axis", self.x_param_combo)

        self.min_spin = self._span_spin()
        self.max_spin = self._span_spin()
        form.addRow("From", self.min_spin)
        form.addRow("To", self.max_spin)

        self.color_swatch = ColorSwatch("#3380ff")
        self.color_swatch.color_picked.connect(self._commit_color)
        form.addRow("Colour", self.color_swatch)

        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("Optional caption, e.g. rotation")
        # Deliberately not textChanged: every keystroke would redraw every
        # plot, and a caption is not worth a round trip through the core per
        # character. editingFinished fires on Enter or when focus leaves, and
        # the timer catches a pause in typing so the plot still follows along.
        self.label_edit.editingFinished.connect(self._commit_label)
        self._label_timer = QTimer(self)
        self._label_timer.setSingleShot(True)
        self._label_timer.setInterval(400)
        self._label_timer.timeout.connect(self._commit_label)
        self.label_edit.textEdited.connect(lambda _t: self._label_timer.start())
        form.addRow("Label", self.label_edit)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.02, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setDecimals(2)
        self.alpha_spin.setKeyboardTracking(False)
        self.alpha_spin.valueChanged.connect(self._commit_alpha)
        form.addRow("Opacity", self.alpha_spin)

    def _span_spin(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-1e12, 1e12)
        spin.setDecimals(6)
        spin.setKeyboardTracking(False)
        spin.valueChanged.connect(self._commit_span)
        return spin

    def _add(self) -> None:
        tile = self._state.tile
        # Default to the middle of what is currently shown, so a new area is
        # visible immediately rather than off-screen at 0..1.
        span = tile.x_max - tile.x_min
        low = tile.x_min + span * 0.4 if span > 0 else 0.0
        high = tile.x_min + span * 0.6 if span > 0 else 1.0

        self._state.areas.append(
            AreaBand(x_param=self._default_x_param(), x_min=low, x_max=high)
        )
        self._state.dirty = True
        self.refresh()
        self.list.setCurrentRow(len(self._state.areas) - 1)
        self.changed.emit()

    def _describe(self, item) -> str:
        span = "{0}:  {1:g} … {2:g}".format(item.x_param or "?",
                                            item.x_min, item.x_max)
        return "{0}   ({1})".format(item.label, span) if item.label else span

    def _load_editor(self, item) -> None:
        self.visible_check.setChecked(item.visible)
        if item.x_param:
            self.x_param_combo.setCurrentText(item.x_param)
        self.min_spin.setValue(item.x_min)
        self.max_spin.setValue(item.x_max)
        self.label_edit.setText(item.label)
        self.color_swatch.set_color(item.color)
        self.alpha_spin.setValue(item.alpha)

    def _commit_label(self) -> None:
        self._label_timer.stop()
        item = self._current()
        if item is None or self._updating:
            return
        text = self.label_edit.text()
        if text == item.label:
            return          # nothing to redraw for
        item.label = text
        self._commit()

    def _commit_visible(self, checked: bool) -> None:
        item = self._current()
        if item is not None and not self._updating:
            item.visible = checked
            self._commit()

    def _commit_x_param(self, name: str) -> None:
        item = self._current()
        if item is not None and not self._updating and name:
            item.x_param = name
            self._commit()

    def _commit_span(self, _value: float) -> None:
        item = self._current()
        if item is None or self._updating:
            return
        low, high = self.min_spin.value(), self.max_spin.value()
        item.x_min, item.x_max = min(low, high), max(low, high)
        self._commit()

    def _commit_color(self, color: str) -> None:
        item = self._current()
        if item is not None:
            item.color = color
            self._commit()

    def _commit_alpha(self, value: float) -> None:
        item = self._current()
        if item is not None and not self._updating:
            item.alpha = float(value)
            self._commit()


class MarkerPanel(_AnnotationPanelBase):
    """Markers: a condition on one parameter, shown on every tile.

    On a tile plotted against the marker's own parameter it is a vertical line.
    On any other tile it becomes a dot on each curve, at the sample where the
    condition holds -- so a marker at TIME = 5 shows where five seconds falls on
    ALPHA = f(NZ).
    """

    def __init__(self, session: Session, state: AppState, parent=None):
        super().__init__(session, state, "Markers", parent)

    def _items(self):
        return self._state.markers

    def _build_editor(self, form: QFormLayout) -> None:
        self.visible_check = QCheckBox("Visible")
        self.visible_check.toggled.connect(self._commit_visible)
        form.addRow(self.visible_check)

        self.x_param_combo = QComboBox()
        self.x_param_combo.currentTextChanged.connect(self._commit_x_param)
        form.addRow("X axis", self.x_param_combo)

        self.position_spin = QDoubleSpinBox()
        self.position_spin.setRange(-1e12, 1e12)
        self.position_spin.setDecimals(6)
        self.position_spin.setKeyboardTracking(False)
        self.position_spin.valueChanged.connect(self._commit_position)
        form.addRow("At", self.position_spin)

        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("Optional label, e.g. Rotation")
        self.label_edit.editingFinished.connect(self._commit_label)
        form.addRow("Label", self.label_edit)

        self.color_swatch = ColorSwatch("#ff3333")
        self.color_swatch.color_picked.connect(self._commit_color)
        form.addRow("Colour", self.color_swatch)

        self.dot_spin = QDoubleSpinBox()
        self.dot_spin.setRange(2.0, 30.0)
        self.dot_spin.setSingleStep(1.0)
        self.dot_spin.setDecimals(1)
        self.dot_spin.setToolTip("Size of the dot drawn on tiles with a different x axis")
        self.dot_spin.setKeyboardTracking(False)
        self.dot_spin.valueChanged.connect(self._commit_dot_size)
        form.addRow("Dot size", self.dot_spin)

    def _add(self) -> None:
        tile = self._state.tile
        span = tile.x_max - tile.x_min
        position = tile.x_min + span * 0.5 if span > 0 else 0.0

        self._state.markers.append(
            Marker(x_param=self._default_x_param(), position=position)
        )
        self._state.dirty = True
        self.refresh()
        self.list.setCurrentRow(len(self._state.markers) - 1)
        self.changed.emit()

    def _describe(self, item) -> str:
        return item.describe()

    def _load_editor(self, item) -> None:
        self.visible_check.setChecked(item.visible)
        if item.x_param:
            self.x_param_combo.setCurrentText(item.x_param)
        self.position_spin.setValue(item.position)
        self.label_edit.setText(item.label)
        self.color_swatch.set_color(item.color)
        self.dot_spin.setValue(item.dot_size)

    def _commit_label(self) -> None:
        item = self._current()
        if item is not None and not self._updating:
            item.label = self.label_edit.text().strip()
            self._commit()

    def _commit_dot_size(self, value: float) -> None:
        item = self._current()
        if item is not None and not self._updating:
            item.dot_size = float(value)
            self._commit()

    def _commit_visible(self, checked: bool) -> None:
        item = self._current()
        if item is not None and not self._updating:
            item.visible = checked
            self._commit()

    def _commit_x_param(self, name: str) -> None:
        item = self._current()
        if item is not None and not self._updating and name:
            item.x_param = name
            self._commit()

    def _commit_position(self, value: float) -> None:
        item = self._current()
        if item is not None and not self._updating:
            item.position = float(value)
            self._commit()

    def _commit_color(self, color: str) -> None:
        item = self._current()
        if item is not None:
            item.color = color
            self._commit()


# ==========================================================================
# Statistics
# ==========================================================================



# ==========================================================================
# Free text
# ==========================================================================


class _DeferredTextEdit(QTextEdit):
    """A QTextEdit that also says when editing has finished.

    QLineEdit has editingFinished; QTextEdit does not, because Enter is a
    newline there. Focus leaving the box is the equivalent moment, and it is
    what lets the caption be applied when the user clicks away rather than on
    every keystroke.
    """

    editing_finished = Signal()

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.editing_finished.emit()


class TextPanel(QWidget):
    """Captions placed on the active tile.

    Deliberately per-tile rather than global, which is the opposite of markers
    and areas. Those describe a *condition* -- a time, a phase -- and belong on
    every plot where that condition can be seen. A caption describes this
    particular picture, and repeating it across a 5x5 grid because it happened
    to be global would be noise.

    Position is a fraction of the tile, not a data coordinate, so a note stays
    where it was put when the plot is panned or zoomed.
    """

    changed = Signal()

    def __init__(self, session: Session, state: AppState, parent=None):
        super().__init__(parent)
        self._session = session
        self._state = state
        self._updating = 0

        # Typing in the caption box used to commit on every keystroke, and a
        # commit repainted the plots, so writing a five-word label made the
        # tile flicker five-and-twenty times and took the caret with it. The
        # text is now committed once the typing pauses, when focus leaves the
        # box, or when the button below is pressed.
        self._text_timer = QTimer(self)
        self._text_timer.setSingleShot(True)
        self._text_timer.setInterval(400)
        self._text_timer.timeout.connect(self._commit)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.tile_label = QLabel("")
        self.tile_label.setObjectName("Hint")
        self.tile_label.setWordWrap(True)
        layout.addWidget(self.tile_label)

        add = QPushButton("Add text")
        add.clicked.connect(self._add)
        layout.addWidget(add)

        self.list = QListWidget()
        self.list.currentRowChanged.connect(lambda _r: self._refresh_editor())
        layout.addWidget(self.list, 1)

        self.editor = QGroupBox("Selected")
        form = QFormLayout(self.editor)

        self.visible_check = QCheckBox("Visible")
        self.visible_check.toggled.connect(self._commit)
        form.addRow(self.visible_check)

        self.text_edit = _DeferredTextEdit()
        self.text_edit.setPlaceholderText("Text — several lines are fine")
        self.text_edit.setFixedHeight(70)
        self.text_edit.textChanged.connect(self._text_typed)
        self.text_edit.editing_finished.connect(self._commit)
        form.addRow("Text", self.text_edit)

        update_now = QPushButton("Update text now")
        update_now.setToolTip(
            "Apply the caption immediately. It is applied on its own a moment "
            "after you stop typing, or when you click away."
        )
        update_now.clicked.connect(self._commit)
        form.addRow(update_now)

        self.x_spin = self._fraction_spin()
        self.y_spin = self._fraction_spin()
        form.addRow("X (0 = left)", self.x_spin)
        form.addRow("Y (0 = top)", self.y_spin)

        # The nine positions people actually want, so the usual case is one
        # click rather than two numbers.
        presets = QGridLayout()
        for row, (name, fy) in enumerate((("Top", 0.04), ("Middle", 0.48),
                                          ("Bottom", 0.92))):
            for col, (side, fx) in enumerate((("L", 0.03), ("C", 0.42),
                                              ("R", 0.80))):
                button = QPushButton("{0[0]}{1}".format(name, side))
                button.setFixedWidth(34)
                button.setToolTip("{0} {1}".format(
                    name, {"L": "left", "C": "centre", "R": "right"}[side]))
                button.clicked.connect(
                    lambda _c=False, a=fx, b=fy: self._set_position(a, b))
                presets.addWidget(button, row, col)
        form.addRow("Place", presets)

        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(4.0, 72.0)
        self.size_spin.setSingleStep(1.0)
        self.size_spin.setDecimals(1)
        self.size_spin.setKeyboardTracking(False)
        self.size_spin.valueChanged.connect(self._commit)
        form.addRow("Size (pt)", self.size_spin)

        self.bold_check = QCheckBox("Bold")
        self.bold_check.toggled.connect(self._commit)
        form.addRow(self.bold_check)

        self.color_swatch = ColorSwatch("#e6e9ec")
        self.color_swatch.color_picked.connect(lambda _c: self._commit())
        form.addRow("Colour", self.color_swatch)

        layout.addWidget(self.editor)

        buttons = QHBoxLayout()
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self._remove)
        copy_all = QPushButton("Copy to all tiles")
        copy_all.setToolTip("Put this tile's captions on every tile of every page")
        copy_all.clicked.connect(self._copy_to_all)
        buttons.addWidget(remove)
        buttons.addWidget(copy_all)
        layout.addLayout(buttons)

        hint = QLabel(
            "Captions belong to the active tile and are positioned as a "
            "fraction of it, so they stay put when you pan or zoom."
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

    # ------------------------------------------------------------- helpers

    @contextmanager
    def _suspended(self):
        self._updating += 1
        try:
            yield
        finally:
            self._updating -= 1

    def _fraction_spin(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.02)
        spin.setDecimals(3)
        spin.setKeyboardTracking(False)
        spin.valueChanged.connect(self._commit)
        return spin

    def _notes(self):
        return self._state.tile.texts

    def _current(self) -> Optional[TextNote]:
        row = self.list.currentRow()
        notes = self._notes()
        return notes[row] if 0 <= row < len(notes) else None

    # ------------------------------------------------------------- refresh

    def refresh(self) -> None:
        with self._suspended():
            self.tile_label.setText("Tile {0}:  {1}".format(
                self._state.active_tile + 1, tile_title(self._state.tile)))
            row = self.list.currentRow()
            self.list.clear()
            for note in self._notes():
                entry = QListWidgetItem(note.describe())
                if not note.visible:
                    entry.setForeground(Qt.GlobalColor.gray)
                self.list.addItem(entry)
            if 0 <= row < self.list.count():
                self.list.setCurrentRow(row)
            elif self.list.count():
                self.list.setCurrentRow(0)
        self._refresh_editor()

    def _refresh_editor(self) -> None:
        note = self._current()
        self.editor.setEnabled(note is not None)
        if note is None:
            return
        with self._suspended():
            self.visible_check.setChecked(note.visible)
            if self.text_edit.toPlainText() != note.text:
                self.text_edit.setPlainText(note.text)
            self.x_spin.setValue(note.x)
            self.y_spin.setValue(note.y)
            self.size_spin.setValue(note.size)
            self.bold_check.setChecked(note.bold)
            self.color_swatch.set_color(note.color)

    # ------------------------------------------------------------- actions

    def _add(self) -> None:
        note = TextNote(text="Note", color=theme_for(self._state.dark_mode).text)
        # Stagger, so a second note does not land exactly on the first.
        count = len(self._notes())
        note.y = min(0.9, 0.05 + 0.07 * count)
        self._notes().append(note)
        self._state.dirty = True
        self.refresh()
        self.list.setCurrentRow(len(self._notes()) - 1)
        self.changed.emit()

    def _remove(self) -> None:
        row = self.list.currentRow()
        notes = self._notes()
        if 0 <= row < len(notes):
            notes.pop(row)
            self._state.dirty = True
            self.refresh()
            self.changed.emit()

    def _set_position(self, x: float, y: float) -> None:
        note = self._current()
        if note is None:
            return
        note.x, note.y = x, y
        self._state.dirty = True
        self._refresh_editor()
        self.changed.emit()

    def _text_typed(self) -> None:
        """Restart the idle timer. Nothing is drawn until it fires."""
        if self._updating:
            return
        self._text_timer.start()

    def _commit(self, *_args) -> None:
        self._text_timer.stop()
        if self._updating:
            return
        note = self._current()
        if note is None:
            return
        note.visible = self.visible_check.isChecked()
        note.text = self.text_edit.toPlainText()
        note.x = self.x_spin.value()
        note.y = self.y_spin.value()
        note.size = self.size_spin.value()
        note.bold = self.bold_check.isChecked()
        note.color = self.color_swatch.color()
        self._state.dirty = True

        row = self.list.currentRow()
        if 0 <= row < self.list.count():
            entry = self.list.item(row)
            entry.setText(note.describe())
            entry.setForeground(
                Qt.GlobalColor.gray if not note.visible else Qt.GlobalColor.white
            )
        self.changed.emit()

    def _copy_to_all(self) -> None:
        notes = [TextNote.from_dict(n.to_dict()) for n in self._notes()]
        current = self._state.tile
        for page in self._state.pages:
            for tile in page.visible_tiles():
                if tile is current:
                    continue
                tile.texts = [TextNote.from_dict(n.to_dict()) for n in notes]
        self._state.dirty = True
        self.changed.emit()


# ==========================================================================
# 3D
# ==========================================================================


class Plot3DPanel(QWidget):
    """Turn the active tile into a rotatable 3D projection of three channels.

    Why a projection rather than an OpenGL scene: pyqtgraph's GL view needs
    PyOpenGL, and on a machine where the package index is blocked every extra
    wheel is a negotiation. A projection needs nothing beyond what is already
    here, and for the thing these files actually contain -- trajectories -- an
    orthographic view you can spin reads at least as well as a lit 3D scene.

    Rotation is by drag in Pan mode, or with the two dials here.
    """

    tile_changed = Signal()

    def __init__(self, session: Session, state: AppState, parent=None):
        super().__init__(parent)
        self._session = session
        self._state = state
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.enable_check = QCheckBox("Draw the active tile in 3D")
        self.enable_check.toggled.connect(self._on_enabled)
        layout.addWidget(self.enable_check)

        self.tile_label = QLabel("")
        self.tile_label.setObjectName("Hint")
        self.tile_label.setWordWrap(True)
        layout.addWidget(self.tile_label)

        axes = QGroupBox("Axes")
        form = QFormLayout(axes)
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.z_combo = QComboBox()
        for combo in (self.x_combo, self.y_combo, self.z_combo):
            combo.currentTextChanged.connect(self._on_axis_changed)
        form.addRow("X", self.x_combo)
        form.addRow("Y", self.y_combo)
        form.addRow("Z", self.z_combo)
        layout.addWidget(axes)

        view = QGroupBox("View")
        view_form = QFormLayout(view)

        self.azimuth = QSlider(Qt.Orientation.Horizontal)
        self.azimuth.setRange(0, 359)
        self.azimuth.valueChanged.connect(self._on_view_changed)
        view_form.addRow("Rotation", self.azimuth)

        self.elevation = QSlider(Qt.Orientation.Horizontal)
        self.elevation.setRange(-89, 89)
        self.elevation.valueChanged.connect(self._on_view_changed)
        view_form.addRow("Tilt", self.elevation)

        self.box_check = QCheckBox("Orientation axes (X, Y, Z)")
        self.box_check.setToolTip(
            "The fixed-size arrow triad in the corner. It shows which way the "
            "object is turned, and stays the same size however far you zoom."
        )
        self.box_check.toggled.connect(self._on_view_changed)
        view_form.addRow(self.box_check)

        self.grid_check = QCheckBox("Reference grid and tick values")
        self.grid_check.setToolTip(
            "Grid lines and tick values on the three faces of the data box "
            "that lie behind the trajectory, swapping as you turn it. Without "
            "them the projection has no depth cue and no numeric scale; with a "
            "very dense trajectory they can be turned off."
        )
        self.grid_check.toggled.connect(self._on_view_changed)
        view_form.addRow(self.grid_check)

        preset_row = QHBoxLayout()
        for label, azimuth, elevation in (
            ("Iso", 45, 25), ("Top", 0, 89), ("Front", 0, 0), ("Side", 90, 0),
        ):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _c=False, a=azimuth, e=elevation: self._set_view(a, e)
            )
            preset_row.addWidget(button)
        view_form.addRow(preset_row)

        layout.addWidget(view)

        apply_all = QPushButton("Apply these axes to every 3D tile")
        apply_all.clicked.connect(self._apply_to_all)
        layout.addWidget(apply_all)

        hint = QLabel(
            "Each axis is normalised, so three channels with different units "
            "stay readable together.\n\n"
            "In Pan mode: drag turns it, middle-drag or Shift+drag slides it, "
            "right-drag or Ctrl+drag zooms. The wheel zooms in Zoom mode. "
            "Double-click or Ctrl+0 frames everything again.\n\n"
            "The corner triad turns with the view — hover it for the parameter "
            "on each axis and its range."
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        layout.addStretch(1)

    def refresh(self) -> None:
        self._updating = True
        try:
            tile = self._state.tile
            params = self._session.parameters

            self.tile_label.setText(
                "Tile {0}:  {1}".format(self._state.active_tile + 1, tile_title(tile))
            )
            self.enable_check.setChecked(tile.plot3d)

            for combo, current in (
                (self.x_combo, tile.x_param),
                (self.y_combo, tile.y_params[0] if tile.y_params else ""),
                (self.z_combo, tile.z_param),
            ):
                combo.clear()
                combo.addItems(params)
                if current in params:
                    combo.setCurrentText(current)

            self.azimuth.setValue(int(tile.azimuth) % 360)
            self.elevation.setValue(int(max(-89, min(89, tile.elevation))))
            self.box_check.setChecked(tile.show_3d_axes)
            self.grid_check.setChecked(tile.show_3d_grid)

            for widget in (self.x_combo, self.y_combo, self.z_combo,
                           self.azimuth, self.elevation, self.box_check,
                           self.grid_check):
                widget.setEnabled(tile.plot3d)
        finally:
            self._updating = False

    # ------------------------------------------------------------- actions

    def _on_enabled(self, checked: bool) -> None:
        if self._updating:
            return
        tile = self._state.tile
        tile.plot3d = bool(checked)
        if checked:
            # Fill in sensible axes rather than showing an empty cube: without
            # this, turning 3D on appears to break the tile.
            params = self._session.parameters
            if not tile.x_param and params:
                tile.x_param = params[0]
            if not tile.y_params and len(params) > 1:
                tile.y_params = [params[1]]
            if not tile.z_param and len(params) > 2:
                tile.z_param = params[2]
        self._state.dirty = True
        self.refresh()
        self.tile_changed.emit()

    def _on_axis_changed(self, _text: str) -> None:
        if self._updating:
            return
        tile = self._state.tile
        tile.x_param = self.x_combo.currentText()
        y = self.y_combo.currentText()
        tile.y_params = [y] if y else []
        tile.z_param = self.z_combo.currentText()
        self._state.dirty = True
        self.tile_changed.emit()

    def _on_view_changed(self, *_args) -> None:
        if self._updating:
            return
        tile = self._state.tile
        tile.azimuth = float(self.azimuth.value())
        tile.elevation = float(self.elevation.value())
        tile.show_3d_axes = self.box_check.isChecked()
        tile.show_3d_grid = self.grid_check.isChecked()
        self._state.dirty = True
        self.tile_changed.emit()

    def _set_view(self, azimuth: float, elevation: float) -> None:
        tile = self._state.tile
        tile.azimuth = azimuth
        tile.elevation = elevation
        self._state.dirty = True
        self.refresh()
        self.tile_changed.emit()

    def _apply_to_all(self) -> None:
        tile = self._state.tile
        for page in self._state.pages:
            for other in page.visible_tiles():
                if other is tile or not other.plot3d:
                    continue
                other.x_param = tile.x_param
                other.y_params = list(tile.y_params)
                other.z_param = tile.z_param
                other.azimuth = tile.azimuth
                other.elevation = tile.elevation
                other.show_3d_axes = tile.show_3d_axes
                other.show_3d_grid = tile.show_3d_grid
        self._state.dirty = True
        self.tile_changed.emit()


# ==========================================================================
# Parameter aliases
# ==========================================================================


class AliasPanel(QWidget):
    """Give a channel a readable name without touching the file.

    AOE files name channels whatever the writing code called them; an alias lets
    the whole application show "Load factor" while still reading NZ. Chains are
    resolved by the core and cycles are broken, so A->B->A cannot hang anything.
    """

    aliases_changed = Signal()

    def __init__(self, session: Session, state: AppState, parent=None):
        super().__init__(parent)
        self._session = session
        self._state = state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        layout.addWidget(_title("New alias"))

        self.target_combo = QComboBox()
        self.target_combo.setEditable(False)
        layout.addWidget(QLabel("Channel"))
        layout.addWidget(self.target_combo)

        self.alias_edit = QLineEdit()
        self.alias_edit.setPlaceholderText("Show it as, e.g. Load factor")
        self.alias_edit.returnPressed.connect(self._add)
        layout.addWidget(QLabel("Display name"))
        layout.addWidget(self.alias_edit)

        add = QPushButton("Add alias")
        add.clicked.connect(self._add)
        layout.addWidget(add)

        self.message = QLabel("")
        self.message.setObjectName("Hint")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)

        layout.addWidget(_title("Defined"))
        self.list = QListWidget()
        layout.addWidget(self.list, 1)

        remove = QPushButton("Remove selected")
        remove.clicked.connect(self._remove)
        layout.addWidget(remove)

    def refresh(self) -> None:
        current = self.target_combo.currentText()
        self.target_combo.clear()
        self.target_combo.addItems(self._session.parameters)
        if current:
            self.target_combo.setCurrentText(current)

        self.list.clear()
        for name in sorted(self._session.parameters):
            resolved = self._session.resolve(name)
            if resolved != name:
                item = QListWidgetItem("{0}  →  {1}".format(name, resolved))
                item.setData(Qt.ItemDataRole.UserRole, name)
                self.list.addItem(item)

    def _add(self) -> None:
        alias = self.alias_edit.text().strip()
        target = self.target_combo.currentText().strip()
        if not alias or not target:
            self.message.setText("Pick a channel and type a display name.")
            return
        if alias == target:
            self.message.setText("The alias and the channel are the same name.")
            return
        self._session.set_alias(alias, target)
        self.alias_edit.clear()
        self.message.setText("{0} now also reachable as {1}.".format(target, alias))
        self.refresh()
        self.aliases_changed.emit()

    def _remove(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        self._session.remove_alias(item.data(Qt.ItemDataRole.UserRole))
        self.refresh()
        self.aliases_changed.emit()


# ==========================================================================
# Inspect
# ==========================================================================


class InspectPanel(QWidget):
    """Every channel of one run at one point, from a click in Inspect mode.

    Clicking a curve answers "what was everything else doing here?" -- the
    question a time history always raises and a single plot cannot answer. The
    run is named explicitly, because with a dozen overlaid runs the one you
    clicked is not obvious.

    Reading every channel means loading them, which on a large file is seconds,
    so it happens on a worker thread like every other load. The columns stay
    cached afterwards, so clicking around the same run is then instant.
    """

    load_requested = Signal(int, int, float)   # file, sim, x value

    def __init__(self, session: Session, state: AppState, parent=None):
        super().__init__(parent)
        self._session = session
        self._state = state
        self._target: Optional[tuple] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.header = QLabel(
            "Switch the mouse to Inspect, then click a curve."
        )
        self.header.setObjectName("Hint")
        self.header.setWordWrap(True)
        layout.addWidget(self.header)

        # --- slope ------------------------------------------------------------
        # A drag in Inspect mode measures a gradient. It answers the question
        # people otherwise answer by reading two points off the plot and doing
        # the arithmetic on paper: how fast is this actually changing here.
        self.slope_box = QGroupBox("Slope")
        slope_layout = QVBoxLayout(self.slope_box)
        self.slope_label = QLabel("Drag across a tile in Inspect mode.")
        self.slope_label.setWordWrap(True)
        self.slope_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        slope_layout.addWidget(self.slope_label)
        layout.addWidget(self.slope_box)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter channels")
        self.filter_edit.textChanged.connect(self._apply_filter)
        layout.addWidget(self.filter_edit)

        self.tabs = QTabWidget()
        # Deliberately not connected here. addTab() moves the current index
        # from -1 to 0, which fires currentChanged while the second page --
        # and everything refresh() reads from it -- does not exist yet. The
        # connection is made below, once both pages are built.

        point_page = QWidget()
        point_layout = QVBoxLayout(point_page)
        point_layout.setContentsMargins(0, 4, 0, 0)
        point_layout.setSpacing(4)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Channel", "Value", "Unit"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        point_layout.addWidget(self.table, 1)

        copy = QPushButton("Copy to clipboard")
        copy.clicked.connect(self._copy)
        point_layout.addWidget(copy)
        self.tabs.addTab(point_page, "At a point")

        # --- statistics, formerly its own tab ------------------------------
        # Merged in because it answers the same question as the point readout
        # -- "what are the numbers here?" -- one curve at a time rather than
        # one sample. Keeping them apart meant two tabs that were only ever
        # useful together.
        stats_page = QWidget()
        stats_layout = QVBoxLayout(stats_page)
        stats_layout.setContentsMargins(0, 4, 0, 0)
        stats_layout.setSpacing(4)

        self.stats_header = QLabel("")
        self.stats_header.setObjectName("Hint")
        self.stats_header.setWordWrap(True)
        stats_layout.addWidget(self.stats_header)

        self.whole_series = QCheckBox("Whole channel instead of the visible range")
        stats_layout.addWidget(self.whole_series)

        self.stats_table = QTableWidget(0, 6)
        self.stats_table.setHorizontalHeaderLabels(
            ["Curve", "Min", "Max", "Mean", "Std", "N"]
        )
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        stats_header_view = self.stats_table.horizontalHeader()
        stats_header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 6):
            stats_header_view.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents)
        stats_layout.addWidget(self.stats_table, 1)

        copy_stats = QPushButton("Copy table to clipboard")
        copy_stats.clicked.connect(self._copy_stats)
        stats_layout.addWidget(copy_stats)
        self.tabs.addTab(stats_page, "Statistics")

        # Everything the two handlers below read now exists. Both call back
        # into refresh paths that touch the whole page, so connecting them
        # while it was still half-built was an accident waiting to be tripped.
        self.whole_series.toggled.connect(lambda _c: self.refresh_stats())
        self.tabs.currentChanged.connect(lambda _i: self.refresh())

        layout.addWidget(self.tabs, 1)

    def request(self, file_index: int, sim_index: int, x_value: float) -> None:
        """Called when a point is clicked; the load itself is done by the window."""
        self._target = (file_index, sim_index, x_value)
        handle = self._session.file(file_index)
        name = handle.name if handle else "?"
        self.header.setText("Reading {0}...".format(name))
        self.load_requested.emit(file_index, sim_index, float(x_value))

    def show_result(self, file_index: int, sim_index: int, x_value: float,
                    x_param: str, rows) -> None:
        handle = self._session.file(file_index)
        if handle is None or sim_index >= handle.simulation_count:
            self.header.setText("That run is no longer loaded.")
            self.table.setRowCount(0)
            return

        simulation = handle.simulations[sim_index]
        self.header.setText(
            "{0}\n{1}\n{2} = {3:.6g}".format(
                handle.name, simulation.name, x_param or "x", x_value
            )
        )

        self.table.setRowCount(len(rows))
        for row, (channel, value) in enumerate(rows):
            name_item = QTableWidgetItem(channel)
            if value != value:  # NaN
                text = "-"
            else:
                text = "{0:.6g}".format(value)
            value_item = QTableWidgetItem(text)
            value_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            unit_item = QTableWidgetItem(self._session.unit(channel))
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, value_item)
            self.table.setItem(row, 2, unit_item)

        self._apply_filter(self.filter_edit.text())

    def refresh_stats(self) -> None:
        tile = self._state.tile
        rows = []

        if tile.is_empty() or not self._state.selected_sims:
            self.stats_header.setText("Nothing plotted in the active tile.")
            self.stats_table.setRowCount(0)
            return

        x_range = None
        if not self.whole_series.isChecked() and tile.x_max > tile.x_min:
            x_range = (tile.x_min, tile.x_max)

        # Whether the window was honoured is decided per curve, below: a run
        # whose x channel is shorter than its y channel cannot be windowed at
        # all, and the header must not claim an x range those figures do not
        # respect.
        windowed = 0

        for file_index, sim_index in sorted(self._state.selected_sims):
            handle = self._session.file(file_index)
            if handle is None or sim_index >= handle.simulation_count:
                continue
            x = self._session.channel(file_index, sim_index, tile.x_param)
            if x.size == 0:
                continue
            sim_id = handle.simulations[sim_index].sim_id

            for param in tile.y_params:
                y = self._session.channel(file_index, sim_index, param)
                if y.size == 0:
                    continue
                stats = Session.stats(y, x, x_range)
                if stats is None:
                    continue
                if stats.get("ranged"):
                    windowed += 1
                rows.append(
                    ("{0} · {1} · {2}".format(self._session.label(param),
                                              sim_id, handle.name), stats)
                )

        if x_range is None:
            scope = "whole channel"
        elif windowed == len(rows):
            scope = "x from {0:.6g} to {1:.6g}".format(*x_range)
        elif windowed == 0:
            scope = "whole channel (no run has an x channel to window by)"
        else:
            scope = "x from {0:.6g} to {1:.6g}; {2} of {3} over the whole channel".format(
                x_range[0], x_range[1], len(rows) - windowed, len(rows))

        self.stats_header.setText(
            "Tile {0}:  {1}\n{2}".format(
                self._state.active_tile + 1, tile_title(tile), scope)
        )

        self.stats_table.setRowCount(len(rows))
        for row, (label, stats) in enumerate(rows):
            values = [
                label,
                "{0:.6g}".format(stats["min"]),
                "{0:.6g}".format(stats["max"]),
                "{0:.6g}".format(stats["mean"]),
                "{0:.6g}".format(stats["std"]),
                str(stats["count"]),
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column > 0:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.stats_table.setItem(row, column, item)

    def _copy_stats(self) -> None:
        lines = ["\t".join(["Curve", "Min", "Max", "Mean", "Std", "N"])]
        for row in range(self.stats_table.rowCount()):
            cells = []
            for column in range(self.stats_table.columnCount()):
                item = self.stats_table.item(row, column)
                cells.append(item.text() if item else "")
            lines.append("\t".join(cells))

        QApplication.clipboard().setText("\n".join(lines))

    def refresh(self) -> None:
        """Re-read the last inspected point, if the run is still loaded.

        Every panel is expected to have this -- the window refreshes them all
        when files change -- so its absence was an AttributeError rather than
        merely a stale panel.
        """
        self.refresh_stats()
        if self._target is None:
            self.clear()
            return
        file_index, sim_index, x_value = self._target
        handle = self._session.file(file_index)
        if handle is None or sim_index >= handle.simulation_count:
            self._target = None
            self.clear()
            return
        self._apply_filter(self.filter_edit.text())

    def show_slope(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Report a gradient dragged out on the active tile."""
        tile = self._state.tile
        dx = x1 - x0
        dy = y1 - y0
        x_name = tile.x_param or "x"
        y_name = ", ".join(tile.y_params[:2]) or "y"

        if dx == 0:
            self.slope_label.setText(
                "Vertical: d{0} = {1:.6g} over no change in {2}.".format(
                    y_name, dy, x_name)
            )
            return

        slope = dy / dx
        # The reciprocal is often the number actually wanted -- seconds per
        # unit rather than units per second -- and it costs one line.
        lines = [
            "d{0} / d{1}  =  {2:.6g}".format(y_name, x_name, slope),
            "d{0} = {1:.6g}   d{2} = {3:.6g}".format(x_name, dx, y_name, dy),
            "from ({0:.6g}, {1:.6g}) to ({2:.6g}, {3:.6g})".format(x0, y0, x1, y1),
        ]
        if slope != 0:
            lines.append("inverse  =  {0:.6g}".format(1.0 / slope))
        self.slope_label.setText("\n".join(lines))

    def clear(self) -> None:
        self.table.setRowCount(0)
        self._target = None
        self.header.setText("Switch the mouse to Inspect, then click a curve.")
        self.slope_label.setText("Drag across a tile in Inspect mode.")

    def _apply_filter(self, text: str) -> None:
        pattern = text.strip().upper()
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is None:
                continue
            if not pattern:
                self.table.setRowHidden(row, False)
                continue
            name = item.text().upper()
            if "*" in pattern or "?" in pattern:
                import fnmatch
                self.table.setRowHidden(row, not fnmatch.fnmatch(name, pattern))
            else:
                self.table.setRowHidden(row, pattern not in name)

    def _copy(self) -> None:
        lines = [self.header.text().replace("\n", " | ")]
        for row in range(self.table.rowCount()):
            if self.table.isRowHidden(row):
                continue
            cells = [self.table.item(row, column) for column in range(3)]
            lines.append("\t".join(cell.text() if cell else "" for cell in cells))
        QApplication.clipboard().setText("\n".join(lines))
