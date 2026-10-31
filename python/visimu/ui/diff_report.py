# Author: Paulo R. A. Gilz
"""The comparison report: one line per run that has something to say.

Deliberately a window of its own rather than a panel in the DIFF tab. A
non-regression run over two files is read once, carefully, against the
thresholds somebody wrote down -- and then acted on, or closed. A tab would
put it behind whichever tab was open next; a window is read where it is, left
open beside the plots it generated, and closed when it has been dealt with.

Nothing here draws anything. A report with forty flagged runs would fill ten
pages of sixteen tiles nobody asked for, so plotting is one explicit action on
one line, and the window emits the run that was asked for and lets the window
behind it decide what that costs.
"""

from __future__ import annotations

import html
from typing import Dict, List, Optional, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from ..filediff import Flag, Report

# Column 0 holds the run and its flags, column 1 the per-line action.
COLUMN_RUN = 0
COLUMN_ACTION = 1

# What a group is drawn in when the user has not said. Grey rather than a
# colour: a group whose colour nobody chose should not look like a verdict.
DEFAULT_GROUP_COLOR = "#9aa3ad"


def line_html(flag_list: Sequence[Flag], colors: Dict[str, str],
              run_id: str) -> str:
    """One report line, with each group name in its own colour.

    The format is the specification's: ``RUN2: ALFA(BLACK), TETA(BLACK),
    TETA(RED)``. Everything that came out of a data file is escaped -- run
    identifiers and channel names come from whatever the solver wrote, and a
    channel called ``<rate>`` should read as a channel, not as markup.
    """
    parts: List[str] = []
    for flag in flag_list:
        for group in flag.groups:
            colour = colors.get(group) or DEFAULT_GROUP_COLOR
            parts.append(
                '{0}(<span style="color:{1}; font-weight:600">{2}</span>)'
                .format(html.escape(flag.param), html.escape(colour, True),
                        html.escape(group)))
    return '<b>{0}</b>: {1}'.format(html.escape(run_id), ", ".join(parts))


def summary_text(report: Report, total_runs: int) -> str:
    """The sentence above the list."""
    flagged = len(report.by_run())
    if not flagged:
        return ("{0} and {1} agree: no run crossed a threshold."
                .format(report.name_a or "File A", report.name_b or "File B"))
    runs = "run" if flagged == 1 else "runs"
    of_total = " of {0}".format(total_runs) if total_runs else ""
    return ("{0}{1} {2} crossed a threshold comparing {3} with {4}."
            .format(flagged, of_total, runs,
                    report.name_a or "File A", report.name_b or "File B"))


class DiffReportDialog(QDialog):
    """The report, with a Plot action per line.

    Kept non-modal on purpose: the point of plotting a line is to look at the
    plots, and a modal report would have to be closed to do that -- and then
    reopened to plot the next line.
    """

    # The run identifier whose flagged channels the user asked to see.
    plot_requested = Signal(str)

    def __init__(self, report: Report, colors: Dict[str, str],
                 total_runs: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Comparison report")
        self.setModal(False)
        self._report = report
        self._colors = dict(colors or {})
        self._rows: List[QTreeWidgetItem] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self.summary = QLabel(summary_text(report, total_runs))
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Run", ""])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionBehavior(
            QTreeWidget.SelectionBehavior.SelectRows)
        self.tree.itemDoubleClicked.connect(self._on_double_clicked)
        # The run column takes the width the window has, and the action
        # column only what its button needs. Without this the column grows to
        # fit its longest line, and a run flagged on eighteen channels has a
        # line thirty entries long -- which is a horizontal scrollbar where a
        # wrapped line belongs.
        header = self.tree.header()
        header.setSectionResizeMode(COLUMN_RUN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(COLUMN_ACTION,
                                    QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.tree, 1)

        self.warning_label = QLabel("")
        self.warning_label.setObjectName("Hint")
        self.warning_label.setWordWrap(True)
        layout.addWidget(self.warning_label)

        hint = QLabel(
            "Double-click a line, or press its Plot button, to generate "
            "DIFF pages for that run. Nothing is plotted until you ask.")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        close = QPushButton("Close")
        close.setToolTip("Close the report without plotting anything")
        close.clicked.connect(self.reject)
        buttons.addWidget(close)
        layout.addLayout(buttons)

        self.resize(560, 360)
        self.set_report(report, self._colors, total_runs)

    # ------------------------------------------------------------ contents

    def set_report(self, report: Report, colors: Optional[Dict[str, str]] = None,
                   total_runs: int = 0) -> None:
        """Show a report, replacing whatever was shown before.

        Reopening the window after a second comparison shows the second one;
        it is the same window, and two reports of the same two files side by
        side would be two answers to one question.
        """
        self._report = report
        if colors is not None:
            self._colors = dict(colors)
        self.summary.setText(summary_text(report, total_runs))

        self.tree.clear()
        self._rows = []
        for run_id, flags in report.by_run():
            item = QTreeWidgetItem(self.tree, ["", ""])
            item.setData(COLUMN_RUN, Qt.ItemDataRole.UserRole, run_id)
            # The text as well as the widget: the widget is what is read, and
            # the text is what a copy, a sort or a screen reader gets.
            item.setText(COLUMN_RUN, report.line_for(run_id))
            label = QLabel(line_html(flags, self._colors, run_id))
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setWordWrap(True)
            label.setContentsMargins(4, 2, 4, 2)
            self.tree.setItemWidget(item, COLUMN_RUN, label)

            plot = QPushButton("Plot")
            plot.setToolTip(
                "Generate DIFF pages overlaying both files for every flagged "
                "channel of this run")
            plot.clicked.connect(
                lambda _checked=False, run=run_id: self.plot_requested.emit(run))
            self.tree.setItemWidget(item, COLUMN_ACTION, plot)
            self._rows.append(item)

        # The rows carry wrapped labels, so their height is whatever the
        # label needs rather than one line each.
        self.tree.resizeColumnToContents(COLUMN_ACTION)
        notes = list(report.warnings)
        if notes:
            self.warning_label.setText(
                "Not everything could be compared:\n- " + "\n- ".join(notes))
        else:
            self.warning_label.setText("")
        self.warning_label.setVisible(bool(notes))

    def runs(self) -> List[str]:
        """The run identifiers listed, in order. For tests, and for callers
        that want to know whether there is anything to plot."""
        return [item.data(COLUMN_RUN, Qt.ItemDataRole.UserRole)
                for item in self._rows]

    # -------------------------------------------------------------- events

    def _on_double_clicked(self, item, _column=0) -> None:
        run_id = item.data(COLUMN_RUN, Qt.ItemDataRole.UserRole) \
            if item is not None else None
        if run_id:
            self.plot_requested.emit(run_id)
