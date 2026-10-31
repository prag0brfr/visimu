# Author: Paulo R. A. Gilz
"""The shortcut and panel reference, on demand.

One window, opened from the toolbar, answering the two questions the manual
answers at length: which key does what, and what is each tab for. Deliberately
short -- it is a reminder, not the guide, and a reference nobody can read at a
glance is one nobody reads at all.

The tab table is *generated* from sidebar.SHORTCUTS rather than written out
again here. A letter changed in one place and not the other would be worse
than no reference: a reminder that lies is read as the truth, and the one
thing this window exists to be trusted about is which key to press.
"""

from __future__ import annotations

from typing import Dict

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QTextBrowser,
    QVBoxLayout,
)

from .sidebar import SHORTCUTS

# One line per panel, in the order the rail builds them. What the tab is for,
# not how to use it; the guide has the how.
PANEL_NOTES: Dict[str, str] = {
    "DATA": "Add and remove result files, and tick which runs are plotted. "
            "Filter the run list by name, with wildcards.",
    "PARAMS": "Choose the tile's X axis, and move channels between "
              "<i>Available</i> and <i>Plotted</i>. Each list has its own "
              "filter.",
    "COMPUTE": "Define channels the file does not contain, from an "
               "expression over the ones it does. Also how a channel gets a "
               "readable name. <i>Help</i> there has the full syntax.",
    "DIFF": "Draw the difference between one channel's values in two runs, "
            "which may come from different files.",
    "ISO": "Filter runs by a value they hold constant &mdash; Mach, mass, "
           "altitude. Several filters combine with AND.",
    "STYLE": "Colour, dash, marker and width, keyed by parameter, file, "
             "simulation or ISO value. Legend contents and position.",
    "SCALE": "Fix the displayed range of either axis, for this tile or for "
             "every tile sharing a channel.",
    "SYNC": "Align the runs of a tile on an event &mdash; a crossing, a "
            "slope, a maximum &mdash; rather than on the clock.",
    "3D": "Turn the tile into a rotatable projection of three channels.",
    "INSPECT": "Every channel of one run at one instant, and statistics of "
               "the visible window.",
    "AREA": "Shaded X bands, for marking phases of a manoeuvre.",
    "MARKERS": "An instant marked on every tile: a line where it is the X "
               "axis, a dot on each curve where it is not.",
    "TEXT": "Captions on the active tile, dragged into place.",
}


def _tab_rows() -> str:
    """The tab table, from the one place the letters are defined."""
    rows = []
    for label, letter in SHORTCUTS.items():
        rows.append(
            "<tr><td><b>{0}</b></td><td><b>{1}</b></td><td>{2}</td></tr>"
            .format(letter, label, PANEL_NOTES.get(label, "")))
    return "".join(rows)


def help_html() -> str:
    """The whole reference. A function, because the tab table is generated."""
    return """
<h3>Keys and panels</h3>
<p>The manual has the detail; this is the reminder.</p>

<h4>The panels</h4>
<p>Press a letter on its own to go to a tab. It never reaches a tab while
anything is being typed into &mdash; a filter, a formula, a page name, a cell.
The rail also takes the <b>wheel</b> over the tab names, the <b>arrow keys</b>
once it has the focus, and <b>Home</b> / <b>End</b>; <b>drag</b> a name to
reorder it, and the order is saved with the workspace.</p>
<table cellpadding="3">{tabs}</table>

<h4>The workspace</h4>
<table cellpadding="3">
<tr><td><b>Ctrl+N</b></td><td>New workspace &mdash; offers to save first</td></tr>
<tr><td><b>Ctrl+L</b></td><td>Load a workspace</td></tr>
<tr><td><b>Ctrl+S</b></td><td>Save; asks for a name the first time</td></tr>
<tr><td><b>Ctrl+Shift+S</b></td><td>Save as &mdash; everything after this saves
there</td></tr>
</table>

<h4>The mouse, and the view</h4>
<table cellpadding="3">
<tr><td><b>Ctrl+1</b> &hellip; <b>Ctrl+4</b></td><td>Select, Inspect, Pan,
Zoom</td></tr>
<tr><td><b>Ctrl+0</b></td><td>Fit every tile, honouring axis locks</td></tr>
<tr><td><b>Double-click</b> a plot</td><td>Fit that tile, and bring PARAMS
forward</td></tr>
<tr><td><b>Double right-click</b></td><td>Fit every tile</td></tr>
<tr><td><b>Ctrl+-</b> / <b>Ctrl+=</b></td><td>Smaller or larger
interface</td></tr>
</table>

<p>In <b>Zoom</b> mode the shape of the drag chooses the axes: an
<b>area</b> zooms the area, a <b>flat</b> drag takes X alone and leaves Y as it
was, an <b>upright</b> drag takes Y alone. Hold <b>Ctrl</b>, or drag with the
<b>right button</b>, and every other tile follows on X &mdash; except a 3D
tile, one whose X is locked, and one plotted against a different channel. The
wheel zooms. A drag of a pixel or two does nothing at all.</p>

<p>In <b>Pan</b> mode a drag moves the view. On a <b>3D</b> tile Pan is
repurposed: drag turns the object, <b>Shift</b>+drag or the middle button
slides it, <b>Ctrl</b>+drag or the right button zooms.</p>

<p>In <b>Select</b> mode nothing in the view responds to a drag, which is what
makes it the mode for dragging <b>markers, bands and captions</b> about. In
<b>Inspect</b> mode a click reads every channel of that run at that point, and
a drag measures a slope.</p>

<h4>Pages and tiles</h4>
<p>The bar above the grid: previous and next page, a name box, <b>+</b> and
<b>&minus;</b> to add and remove, <b>&#9668;</b> <b>&#9658;</b> to reorder, and
<b>Duplicate</b> for the page with every tile on it. <b>Grid</b> sets the
layout up to 5&times;5. The four arrows swap the active tile with a neighbour;
the four symbols copy, cut, paste and clear it. <b>Fit</b> re-frames every
tile, and the <b>open padlock</b> beside it drops every axis lock on every page
and then fits.</p>

<p>One tile is <b>active</b>, marked by an accent border, and every per-tile
control acts on it. Click to make it active.</p>
""".format(tabs=_tab_rows())


class HelpDialog(QDialog):
    """The reference, on demand rather than permanently on screen."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keys and panels")
        self.resize(680, 700)

        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setHtml(help_html())
        browser.setOpenExternalLinks(False)
        layout.addWidget(browser)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
