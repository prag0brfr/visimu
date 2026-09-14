# Author: Paulo R. A. Gilz
"""The presentation layer, tested without a display.

Two kinds of thing live here, and they have the same reason to exist: PySide6 is
a large binary wheel and a plot widget needs a screen, so none of this could be
tested at all until qt_stubs stood in for PySide6 and pyqtgraph.

**What a redraw is allowed to cost.** Responsiveness is not decided by how fast
a curve is drawn -- decimation and per-pen grouping settled that -- but by how
often the code decides to draw a curve that is already on screen, and how much
of the widget tree it disturbs to answer a click that changed nothing. Those are
decisions, and a decision can be checked without drawing anything.

**What the interface says.** Summaries, badges and the sizes everything is set
in: the arithmetic of "5 of 12 selected, and here are the first eight" has more
edge cases than the drawing does.

    python -m unittest tests.test_redraw -v
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import numpy as np

import qt_stubs  # noqa: F401  (installs the stub modules; must precede the ui import)
from PySide6.QtCore import Qt  # noqa: E402  (the stub installed above)

from aoeviewer import isogroups
from aoeviewer import scales
from aoeviewer import theme as theme_module
from aoeviewer import model
from aoeviewer.model import MAX_TILES, AppState, StyleRegistry
from aoeviewer.scales import resample_key
from aoeviewer.ui import main_window
from aoeviewer.ui import panels
from aoeviewer.ui import plot_grid as pg_grid
from aoeviewer.ui import plot_tile as pt


class _Theme:
    accent = "#ff8800"
    border = "#333333"
    plot_foreground = "#eeeeee"
    plot_background = "#111111"
    plot_grid = "#444444"
    text_dim = "#888888"


def make_tile(index: int = 0) -> pt.PlotTile:
    """A PlotTile with only the attributes the redraw path reads.

    Deliberately not constructed: PlotWidget.__init__ builds a scene, and none
    of what is tested here needs one.
    """
    tile = object.__new__(pt.PlotTile)
    tile._plot_item = qt_stubs.PlotItem()
    tile.style_sheets = []
    tile._theme = _Theme()
    tile._index = index
    tile._state = AppState()
    tile._curves = {}
    tile._groups = []
    tile._overlays = []
    tile._legend = None
    tile._border_active = None
    return tile


def curve(key: str, x: np.ndarray, y: np.ndarray) -> pt.CurveEntry:
    return pt.CurveEntry(x, y, key, "style::" + key, key)


def plan(name=None, color="#ff0000", entries=()) -> pt._GroupPlan:
    made = pt._GroupPlan(color, "solid", 1.0, None, 6.0, name)
    made.entries = list(entries)
    return made


class TestActiveBorder(unittest.TestCase):
    """The border repaint that clicking a tile used to cost.

    _refresh_active_border() calls set_active_border() on every tile in the
    page, on every mouse press, because a press emits `activated` whether or not
    the active tile changed. QWidget.setStyleSheet does no comparison of its
    own, so each of those repolished a plot widget and scheduled a repaint of
    its whole scene. On a 5x5 page that is twenty-five plots redrawn to change a
    border on two.
    """

    def test_repeated_calls_apply_one_sheet(self):
        tile = make_tile()
        for _ in range(25):
            tile.set_active_border(False)
        self.assertEqual(len(tile.style_sheets), 1)

    def test_a_real_change_is_applied(self):
        tile = make_tile()
        tile.set_active_border(False)
        tile.set_active_border(True)
        tile.set_active_border(False)
        self.assertEqual(len(tile.style_sheets), 3)
        self.assertIn(_Theme.border, tile.style_sheets[-1])

    def test_the_active_tile_gets_the_accent_colour(self):
        tile = make_tile()
        tile.set_active_border(True)
        self.assertIn(_Theme.accent, tile.style_sheets[-1])
        self.assertIn("2px", tile.style_sheets[-1])

    def test_a_new_theme_forces_the_sheet_back_on(self):
        # The cached state describes colours, so a theme change has to drop it
        # or the border keeps the old palette's colour for ever.
        tile = make_tile()
        tile.set_active_border(True)
        tile._border_active = None          # what apply_theme() does
        tile.set_active_border(True)
        self.assertEqual(len(tile.style_sheets), 2)


class TestGroupReuse(unittest.TestCase):
    """Rebuilds that change no curve must keep the items they have.

    A rebuild arrives for many reasons that say nothing about the data --
    restyling, toggling the grid, dragging a marker, returning to a page. Each
    one used to drop every draw item and re-decimate every curve from full
    resolution for an identical answer.
    """

    def setUp(self):
        self.tile = make_tile()
        self.plot = self.tile.getPlotItem()
        self.x = np.arange(100, dtype=np.float32)
        self.y = np.sin(self.x, dtype=np.float32)

    def _existing(self, entries, key="sampled"):
        group = pt.DrawGroup(qt_stubs.PlotDataItem(name="A"))
        group.entries = list(entries)
        group.sampled_for = key
        self.plot.items.append(group.item)
        return group

    def test_identical_membership_keeps_the_item_and_its_samples(self):
        entries = [curve("0|0|ALT", self.x, self.y)]
        old = self._existing(entries)
        item = old.item

        self.tile._adopt_groups(self.plot, [old], [plan(entries=entries)])

        self.assertIs(self.tile._groups[0].item, item)
        self.assertEqual(self.tile._groups[0].sampled_for, "sampled")
        self.assertEqual(self.plot.added, [])
        self.assertEqual(self.plot.removed, [])

    def test_a_different_buffer_is_not_reused(self):
        # A reloaded file, or a column evicted by the memory cap and read back,
        # gives the same curve a different buffer. Reusing the item there would
        # leave the old file's samples on screen.
        old = self._existing([curve("0|0|ALT", self.x, self.y)])
        fresh = [curve("0|0|ALT", self.x.copy(), self.y.copy())]

        self.tile._adopt_groups(self.plot, [old], [plan(entries=fresh)])

        self.assertIsNot(self.tile._groups[0].item, old.item)
        self.assertIsNone(self.tile._groups[0].sampled_for)
        self.assertIn(old.item, self.plot.removed)

    def test_a_changed_membership_is_not_reused(self):
        first = curve("0|0|ALT", self.x, self.y)
        second = curve("0|1|ALT", self.x, self.y)
        old = self._existing([first])

        self.tile._adopt_groups(self.plot, [old], [plan(entries=[first, second])])

        self.assertIsNone(self.tile._groups[0].sampled_for)
        self.assertIn(old.item, self.plot.removed)

    def test_curves_moving_between_groups_are_redrawn(self):
        # Recolouring one curve of a pair splits them into two groups. Neither
        # new group draws what the old one drew, so neither may keep its points.
        first = curve("0|0|ALT", self.x, self.y)
        second = curve("0|1|ALT", self.x, self.y)
        old = self._existing([first, second])

        self.tile._adopt_groups(
            self.plot, [old],
            [plan(color="#ff0000", entries=[first]),
             plan(color="#00ff00", entries=[second])],
        )

        self.assertEqual(len(self.tile._groups), 2)
        for group in self.tile._groups:
            self.assertIsNone(group.sampled_for)
        self.assertIn(old.item, self.plot.removed)

    def test_a_restyled_group_keeps_its_points(self):
        # The pen is not part of what decides reuse, and must not be: the drawn
        # points depend on the samples and the view, never on the colour. This
        # is the case that makes changing a colour cheap.
        entries = [curve("0|0|ALT", self.x, self.y)]
        old = self._existing(entries)

        self.tile._adopt_groups(
            self.plot, [old], [plan(color="#00ff00", entries=entries)])

        group = self.tile._groups[0]
        self.assertIs(group.item, old.item)
        self.assertEqual(group.sampled_for, "sampled")
        self.assertEqual(len(old.item.pen_calls), 1)
        self.assertEqual(old.item.data_calls, [])

    def test_a_reused_item_gets_its_legend_row_back(self):
        # rebuild() clears the legend, and pyqtgraph only fills it from
        # addItem() -- which an adopted item does not go through. Without the
        # explicit re-add the row silently disappears.
        entries = [curve("0|0|ALT", self.x, self.y)]
        old = self._existing(entries)
        legend = qt_stubs.LegendItem()
        self.tile._legend = legend
        self.plot.legend = legend

        self.tile._adopt_groups(
            self.plot, [old], [plan(name="ALT", entries=entries)])

        self.assertEqual([name for _item, name in legend.rows], ["ALT"])
        self.assertEqual(old.item.opts["name"], "ALT")

    def test_legend_order_follows_the_plan_order(self):
        # Adopted and freshly built groups add their rows in the same loop, so
        # a page that reuses some items and rebuilds others must still list them
        # in the order the curves were walked.
        first = curve("0|0|ALT", self.x, self.y)
        second = curve("0|1|ALT", self.x, self.y)
        old = self._existing([second])
        legend = qt_stubs.LegendItem()
        self.tile._legend = legend
        self.plot.legend = legend

        self.tile._adopt_groups(
            self.plot, [old],
            [plan(name="first", entries=[first]),
             plan(name="second", entries=[second])],
        )

        self.assertEqual([name for _item, name in legend.rows],
                         ["first", "second"])

    def test_everything_not_adopted_is_removed(self):
        keep = [curve("0|0|ALT", self.x, self.y)]
        drop = [curve("0|9|ALT", self.x, self.y)]
        kept, dropped = self._existing(keep), self._existing(drop)

        self.tile._adopt_groups(self.plot, [kept, dropped],
                                [plan(entries=keep)])

        self.assertEqual(self.plot.removed, [dropped.item])
        self.assertIn(kept.item, self.plot.items)

    def test_an_empty_group_is_never_adopted(self):
        # The "... and N more" legend spacer carries no curves. Two of them
        # would compare equal on an empty signature and could be handed to a
        # plan that draws real data.
        spacer = pt.DrawGroup(qt_stubs.PlotDataItem(name="... and 3 more"))
        spacer.sampled_for = "sampled"
        entries = [curve("0|0|ALT", self.x, self.y)]

        self.tile._adopt_groups(self.plot, [spacer], [plan(entries=entries)])

        self.assertIsNot(self.tile._groups[0].item, spacer.item)
        self.assertIn(spacer.item, self.plot.removed)

    def test_reuse_survives_a_round_trip(self):
        # Adopting twice in a row must be stable: the second pass sees the
        # entries the first one installed, not the ones it was built from.
        entries = [curve("0|0|ALT", self.x, self.y)]
        old = self._existing(entries)

        self.tile._adopt_groups(self.plot, [old], [plan(entries=entries)])
        previous = self.tile._groups
        self.tile._groups = []
        again = [curve("0|0|ALT", self.x, self.y)]
        self.tile._adopt_groups(self.plot, previous, [plan(entries=again)])

        self.assertIs(self.tile._groups[0].item, old.item)
        self.assertEqual(self.tile._groups[0].sampled_for, "sampled")


class TestReuseSignature(unittest.TestCase):
    """What counts as "the same curves"."""

    def setUp(self):
        self.x = np.arange(10, dtype=np.float32)
        self.y = np.arange(10, dtype=np.float32)

    def test_same_buffers_compare_equal(self):
        a = [curve("0|0|ALT", self.x, self.y)]
        b = [curve("0|0|ALT", self.x, self.y)]
        self.assertEqual(pt.PlotTile._reuse_signature(a),
                         pt.PlotTile._reuse_signature(b))

    def test_a_different_curve_compares_unequal(self):
        a = [curve("0|0|ALT", self.x, self.y)]
        b = [curve("0|0|MACH", self.x, self.y)]
        self.assertNotEqual(pt.PlotTile._reuse_signature(a),
                            pt.PlotTile._reuse_signature(b))

    def test_a_truncated_buffer_compares_unequal(self):
        # Same address, fewer samples: a file that shrank on disk and was
        # reloaded in place. Length is part of the signature for this reason.
        a = [curve("0|0|ALT", self.x, self.y)]
        b = [curve("0|0|ALT", self.x[:5], self.y[:5])]
        self.assertNotEqual(pt.PlotTile._reuse_signature(a),
                            pt.PlotTile._reuse_signature(b))

    def test_order_matters(self):
        first = curve("0|0|ALT", self.x, self.y)
        second = curve("0|1|ALT", self.x, self.y)
        self.assertNotEqual(pt.PlotTile._reuse_signature([first, second]),
                            pt.PlotTile._reuse_signature([second, first]))


class TestResampleKey(unittest.TestCase):
    """The key that decides whether drawn points are still the right ones.

    Too strict and every mouse move re-decimates every curve; too loose and the
    plot shows a resolution the view no longer has.
    """

    def test_the_same_view_gives_the_same_key(self):
        self.assertEqual(resample_key((0.0, 10.0), 1800, False),
                         resample_key((0.0, 10.0), 1800, False))

    def test_float_wobble_does_not_change_it(self):
        # What a view range read back from a plot actually looks like after a
        # mouse move: identical to the eye, different in the last bits.
        a = resample_key((0.0, 10.0), 1800, False)
        b = resample_key((1e-13, 10.000000000001), 1800, False)
        self.assertEqual(a, b)

    def test_zooming_about_the_origin_changes_it(self):
        # The regression this class was written for. Quantising only the
        # endpoints made these two equal -- the lower edge and the bucket width
        # shrank by the same factor -- so zooming in around zero kept whatever
        # resolution the curve already had.
        wide = resample_key((-0.05, 1.05), 1800, False)
        close = resample_key((-0.0125, 0.2625), 1800, False)
        self.assertNotEqual(wide, close)

    def test_zooming_anywhere_else_changes_it(self):
        self.assertNotEqual(resample_key((9.5, 20.5), 1800, False),
                            resample_key((11.7, 18.3), 1800, False))

    def test_panning_changes_it(self):
        self.assertNotEqual(resample_key((0.0, 10.0), 1800, False),
                            resample_key((5.0, 15.0), 1800, False))

    def test_a_sub_bucket_pan_does_not(self):
        # A tenth of a bucket cannot move a drawn point, so it must not cost a
        # re-decimation of every curve in the tile.
        bucket = 10.0 / 1800
        self.assertEqual(resample_key((0.0, 10.0), 1800, False),
                         resample_key((bucket * 0.1, 10.0 + bucket * 0.1),
                                      1800, False))

    def test_the_budget_is_part_of_it(self):
        # Resizing the tile changes how many points each curve gets, which is a
        # different answer even for an unchanged view.
        self.assertNotEqual(resample_key((0.0, 10.0), 1800, False),
                            resample_key((0.0, 10.0), 900, False))

    def test_log_mode_is_part_of_it(self):
        self.assertNotEqual(resample_key((1.0, 10.0), 1800, False),
                            resample_key((1.0, 10.0), 1800, True))

    def test_no_range_means_the_whole_series(self):
        key = resample_key(None, 1800, False)
        self.assertEqual(key, resample_key(None, 1800, False))
        self.assertNotEqual(key, resample_key((0.0, 10.0), 1800, False))

    def test_a_degenerate_range_is_treated_as_no_range(self):
        # An inverted or zero-width range means "everything" to the core, so it
        # must key the same way rather than dividing by zero.
        self.assertEqual(resample_key((5.0, 5.0), 1800, False),
                         resample_key(None, 1800, False))
        self.assertEqual(resample_key((9.0, 1.0), 1800, False),
                         resample_key(None, 1800, False))

    def test_a_non_finite_range_is_survivable(self):
        for bad in ((float("nan"), 1.0), (0.0, float("inf")),
                    (float("-inf"), float("inf"))):
            self.assertEqual(resample_key(bad, 1800, False),
                             resample_key(None, 1800, False))

    def test_significant_keeps_twelve_figures(self):
        self.assertEqual(scales.significant(1.0), 1.0)
        self.assertEqual(scales.significant(0.0), 0.0)
        self.assertEqual(scales.significant(1234.5678, 4), 1235.0)
        self.assertEqual(scales.significant(-0.00012345678, 3), -0.000123)


class _RecordingSession:
    """Counts decimations, which is the thing under test."""

    def __init__(self):
        self.calls = 0

    def decimate(self, x, y, target, x_range):
        self.calls += 1
        return x[:target], y[:target]


class TestResampleSkipping(unittest.TestCase):
    """A redraw that does not move the view must not re-decimate."""

    def setUp(self):
        self.tile = make_tile()
        self.session = _RecordingSession()
        self.tile._session = self.session
        x = np.arange(1000, dtype=np.float32)
        entry = curve("0|0|ALT", x, x)
        group = pt.DrawGroup(qt_stubs.PlotDataItem())
        group.entries = [entry]
        self.tile._groups = [group]
        self.tile._curves = {"0|0|ALT": entry}
        self.group = group

    def test_the_first_pass_decimates(self):
        self.tile._resample_visible()
        self.assertEqual(self.session.calls, 1)
        self.assertIsNotNone(self.group.sampled_for)

    def test_a_second_pass_over_the_same_view_does_not(self):
        self.tile._resample_visible()
        self.tile._resample_visible()
        self.tile._resample_visible()
        self.assertEqual(self.session.calls, 1)

    def test_force_overrides_the_skip(self):
        self.tile._resample_visible()
        self.tile._resample_visible(force=True)
        self.assertEqual(self.session.calls, 2)

    def test_moving_the_view_decimates_again(self):
        self.tile._resample_visible()
        self.tile.getPlotItem().vb._range = [[0.0, 0.25], [0.0, 1.0]]
        self.tile._resample_visible()
        self.assertEqual(self.session.calls, 2)

    def test_an_adopted_group_is_left_alone(self):
        # The whole point of carrying sampled_for across a rebuild: the group
        # comes out of _adopt_groups already answering the current question.
        self.tile._resample_visible()
        entries = self.group.entries
        self.tile._groups = []
        self.tile._adopt_groups(self.tile.getPlotItem(), [self.group],
                                [plan(color="#00ff00", entries=entries)])
        self.tile._resample_visible()
        self.assertEqual(self.session.calls, 1)


class _FakeSession:
    """Only what StylePanel._style_entries() asks of a session."""

    def __init__(self):
        self.parameters = ["TIME", "ALT", "MACH"]

    def file(self, index):
        return None

    def iso_distinct(self, name):
        return np.array([], dtype=np.float32)

    def format_value(self, value):
        return "{0:g}".format(value)


class TestStyleTable(unittest.TestCase):
    """The per-curve table, which is rebuilt on every click that changes tile.

    In simulation mode it holds one row per selected run, and each row is a
    colour button and two populated combo boxes. Recreating those on every
    refresh meant a study of a few hundred runs created and destroyed a
    four-figure number of widgets to answer a click that usually changed nothing
    in this table at all.
    """

    def setUp(self):
        self.panel = object.__new__(panels.StylePanel)
        self.panel._session = _FakeSession()
        self.panel._state = AppState()
        self.panel._updating = False
        self.panel._table_signature = None
        self.panel._row_keys = []
        self.panel.table = qt_stubs.QTableWidget(0, 4)
        self.panel.style_changed  # instantiate the stub signal
        self.entries = [("ALT", "param::ALT"), ("MACH", "param::MACH")]
        self.panel._style_entries = lambda: list(self.entries)

    def test_the_first_fill_creates_the_widgets(self):
        self.panel._rebuild_table()
        self.assertEqual(self.panel.table.rowCount(), 2)
        self.assertEqual(self.panel.table.created, 6)   # three per row

    def test_an_unchanged_refresh_touches_nothing(self):
        self.panel._rebuild_table()
        created = self.panel.table.created
        for _ in range(20):
            self.panel._rebuild_table()
        self.assertEqual(self.panel.table.created, created)

    def test_a_changed_style_reuses_the_widgets(self):
        self.panel._rebuild_table()
        created = self.panel.table.created
        swatch = self.panel.table.cellWidget(0, 1)

        self.panel._state.styles.set_color("param::ALT", "#123456")
        self.panel._rebuild_table()

        self.assertEqual(self.panel.table.created, created)
        self.assertIs(self.panel.table.cellWidget(0, 1), swatch)
        self.assertEqual(swatch.color(), "#123456")

    def test_growing_the_table_only_builds_the_new_rows(self):
        self.panel._rebuild_table()
        created = self.panel.table.created
        first_row_swatch = self.panel.table.cellWidget(0, 1)

        self.entries.append(("TIME", "param::TIME"))
        self.panel._rebuild_table()

        self.assertEqual(self.panel.table.rowCount(), 3)
        self.assertEqual(self.panel.table.created, created + 3)
        self.assertIs(self.panel.table.cellWidget(0, 1), first_row_swatch)

    def test_shrinking_the_table_drops_the_extra_rows(self):
        self.panel._rebuild_table()
        self.entries.pop()
        self.panel._rebuild_table()
        self.assertEqual(self.panel.table.rowCount(), 1)
        self.assertIsNone(self.panel.table.cellWidget(1, 1))

    def test_a_reused_row_edits_the_curve_it_now_shows(self):
        # The whole risk of reuse: a widget outlives the key it was built for,
        # so a stale binding would quietly restyle the wrong curve.
        self.panel._rebuild_table()
        swatch = self.panel.table.cellWidget(0, 1)

        self.entries[0] = ("MACH", "param::MACH")
        self.entries[1] = ("ALT", "param::ALT")
        self.panel._rebuild_table()
        self.assertIs(self.panel.table.cellWidget(0, 1), swatch)

        swatch.color_picked.emit("#abcdef")
        self.assertEqual(self.panel._state.styles.color_for("param::MACH"),
                         "#abcdef")

    def test_a_dropped_row_edits_nothing(self):
        self.panel._rebuild_table()
        swatch = self.panel.table.cellWidget(1, 1)
        before = self.panel._state.styles.color_for("param::MACH")

        self.entries.pop()
        self.panel._rebuild_table()

        # The widget is gone from the table but a test can still hold it; it
        # must not write into whatever key happens to be at that index now.
        swatch.color_picked.emit("#abcdef")
        self.assertEqual(self.panel._state.styles.color_for("param::MACH"),
                         before)

    def test_filling_a_row_does_not_emit_a_style_change(self):
        # setCurrentText fires currentTextChanged exactly as a user edit does.
        # Without the guard, refreshing the panel would restyle the curve and
        # ask for a redraw of the whole grid.
        self.panel._rebuild_table()
        self.panel._state.styles.set_dash("param::ALT", "dash")
        self.panel._rebuild_table()
        self.assertEqual(self.panel.style_changed.emitted, [])

    def test_an_empty_table_shows_the_hint(self):
        self.entries = []
        self.panel._rebuild_table()
        self.assertEqual(self.panel.table.rowCount(), 1)
        self.assertEqual(self.panel.table.item(0, 0).text(),
                         "Nothing plotted yet")

    def test_the_hint_row_is_replaced_by_real_rows(self):
        self.entries = []
        self.panel._rebuild_table()
        self.entries = [("ALT", "param::ALT")]
        self.panel._rebuild_table()
        self.assertEqual(self.panel.table.item(0, 0).text(), "ALT")
        self.assertIsNotNone(self.panel.table.cellWidget(0, 1))

    def test_rows_left_over_are_not_shown_under_the_hint(self):
        self.panel._rebuild_table()
        self.entries = []
        self.panel._rebuild_table()
        self.assertEqual(self.panel.table.rowCount(), 1)
        self.assertIsNone(self.panel.table.cellWidget(0, 1))

    def test_updates_are_re_enabled_afterwards(self):
        self.panel._rebuild_table()
        self.assertTrue(self.panel.table.updates_enabled)


class TestApplyLegendToAll(unittest.TestCase):
    """The STYLE tab's "Apply legend to all tiles" button.

    The copying itself is the model's and is tested in test_monkey; what is
    checked here is the wiring, including one thing that is easy to get wrong
    and invisible until someone reads the status bar: the redraw ends by
    rewriting the status bar with the file and run counts, so the report has to
    be emitted *after* it or it is overwritten before it can be read.
    """

    def setUp(self):
        self.panel = object.__new__(panels.StylePanel)
        self.panel._session = _FakeSession()
        self.panel._state = AppState()
        self.panel._updating = False
        self.panel._table_signature = None
        self.panel._row_keys = []
        self.panel.table = qt_stubs.QTableWidget(0, 4)
        self.order = []
        self.panel.style_changed.connect(lambda: self.order.append("redraw"))
        self.panel.legend_applied.connect(lambda n: self.order.append(n))

    def test_it_copies_the_active_tile_s_legend(self):
        self.panel._state.tile.legend_show_file = True
        self.panel._state.tile.legend_position = "bottom-right"

        self.panel._apply_legend_to_all()

        for page in self.panel._state.pages:
            for tile in page.tiles:
                self.assertTrue(tile.legend_show_file)
                self.assertEqual(tile.legend_position, "bottom-right")

    def test_it_reports_after_it_redraws(self):
        self.panel._state.tile.legend_show_file = True
        self.panel._apply_legend_to_all()
        self.assertEqual(self.order[0], "redraw")
        self.assertEqual(self.order[-1], MAX_TILES - 1)

    def test_a_no_op_click_still_reports(self):
        # Every tile already matches, so there is nothing to redraw -- but the
        # button must still say something, or it reads as broken.
        self.panel._apply_legend_to_all()
        self.assertEqual(self.order, [0])


class TestIsoSelectionSummary(unittest.TestCase):
    """The one-line answer to "what is this filter letting through".

    The ISO tab shows the values of one filter at a time, so with three active
    there was no way to see what was selected without clicking through them.
    These lines are what the overview and the row badges are built from, and
    every interesting case here is an edge: nothing selected, everything
    selected, and more values than will fit on a line.
    """

    def test_it_names_the_selected_values(self):
        self.assertEqual(
            panels.iso_selection_summary(["0.70", "0.75", "0.80"],
                                         ["0.70", "0.80"]),
            "0.70, 0.80  (2 of 3)")

    def test_all_of_them_is_said_rather_than_listed(self):
        # Forty numbers to say "all forty" is worse than saying "all forty".
        labels = ["{0}".format(i) for i in range(40)]
        self.assertEqual(panels.iso_selection_summary(labels, labels), "all 40")

    def test_none_of_them_says_how_many_there_were(self):
        self.assertEqual(
            panels.iso_selection_summary(["a", "b", "c"], []), "none of 3")

    def test_a_long_selection_is_truncated_and_counted(self):
        labels = ["v{0}".format(i) for i in range(30)]
        summary = panels.iso_selection_summary(labels, labels[:20], max_values=8)
        self.assertTrue(summary.startswith("v0, v1, v2, v3, v4, v5, v6, v7, +12 more"))
        self.assertIn("(20 of 30)", summary)

    def test_exactly_the_limit_is_not_truncated(self):
        labels = ["v{0}".format(i) for i in range(10)]
        summary = panels.iso_selection_summary(labels, labels[:8], max_values=8)
        self.assertNotIn("more", summary)
        self.assertIn("(8 of 10)", summary)

    def test_a_parameter_with_no_values_says_so(self):
        self.assertEqual(panels.iso_selection_summary([], []), "no values")

    def test_the_badge_is_short_enough_for_a_list_row(self):
        self.assertEqual(panels.iso_selection_badge(12, 3), "3/12")
        self.assertEqual(panels.iso_selection_badge(5, 5), "all 5")
        self.assertEqual(panels.iso_selection_badge(4, 0), "none of 4")
        self.assertEqual(panels.iso_selection_badge(0, 0), "")


class _IsoSession:
    """Only what the ISO panel asks of a session, and it counts the asking.

    `distinct_calls` is the subject of a test: reading a parameter's distinct
    values is a scan of every block header of every open file, and doing it per
    question turned ticking one checkbox into a call per filter.
    """

    def __init__(self, values):
        self._values = {k: np.array(v, dtype=np.float32)
                        for k, v in values.items()}
        self.parameters = sorted(self._values)
        self.distinct_calls = 0

    def iso_distinct(self, name):
        self.distinct_calls += 1
        return self._values.get(name, np.empty(0, dtype=np.float32))

    def format_value(self, value):
        return "{0:g}".format(value)

    def iso_auto_group(self, values, count):
        edges = isogroups.even_edges(list(np.asarray(values).ravel()), count)
        return [_Group(isogroups.default_name(lo, hi, self.format_value), lo, hi)
                for lo, hi in zip(edges, edges[1:])]


class _Group:
    def __init__(self, name, low, high, active=True):
        self.name, self.low, self.high, self.active = name, low, high, active


def make_iso_panel(values=None):
    """An IsoPanel with the widgets the tree code touches, and nothing else."""
    panel = object.__new__(panels.IsoPanel)
    panel._session = _IsoSession(values if values is not None else {
        "MACH": [0.7, 0.75, 0.8],
        "WEIGHT": [180.0, 200.0],
    })
    panel._state = AppState()
    panel._updating = False
    panel._expanded = set()
    panel._values_cache = {}
    panel._tree_state = None
    panel.tree = qt_stubs.QTreeWidget()
    panel.summary = qt_stubs.QLabel()
    panel.param_combo = qt_stubs.QComboBox()
    panel.apply_filter = lambda: panel.filter_changed.emit()
    panel.tree.itemChanged.connect(
        lambda item, _col=0: panel._on_item_changed(item))
    return panel


def node_named(panel, param):
    for index in range(panel.tree.topLevelItemCount()):
        node = panel.tree.topLevelItem(index)
        if node.data(0, Qt.ItemDataRole.UserRole)[1] == param:
            return node
    return None


def child_labels(node):
    return [node.child(i).text(0) for i in range(node.childCount())]


class TestIsoTree(unittest.TestCase):
    """Every parameter over its own values, all on screen at once.

    The old arrangement was a list of parameters above a list of the selected
    one's values, so with three filters active there was no way to see what was
    selected without clicking through them and remembering.
    """

    def setUp(self):
        self.panel = make_iso_panel()

    def test_each_parameter_is_a_node_over_its_values(self):
        self.panel._state.iso_state("MACH")
        self.panel._state.iso_state("WEIGHT")
        self.panel._rebuild_tree()

        self.assertEqual(self.panel.tree.topLevelItemCount(), 2)
        self.assertEqual(child_labels(node_named(self.panel, "MACH")),
                         ["0.7", "0.75", "0.8"])
        self.assertEqual(child_labels(node_named(self.panel, "WEIGHT")),
                         ["180", "200"])

    def test_a_parameter_row_carries_its_count(self):
        iso = self.panel._state.iso_state("MACH")
        iso.value_enabled[self.panel._distinct("MACH")[1]] = False
        self.panel._rebuild_tree()
        self.assertIn("2/3", node_named(self.panel, "MACH").text(0))

    def test_untouched_values_are_ticked(self):
        # Absent from value_enabled means on, which is how a freshly added
        # filter behaves. A tree that showed them unticked would say the filter
        # was hiding everything.
        self.panel._state.iso_state("MACH")
        self.panel._rebuild_tree()
        node = node_named(self.panel, "MACH")
        for index in range(node.childCount()):
            self.assertEqual(node.child(index).checkState(0),
                             Qt.CheckState.Checked)

    def test_the_parameter_checkbox_is_the_filter_switch(self):
        iso = self.panel._state.iso_state("MACH")
        iso.active = False
        self.panel._rebuild_tree()
        self.assertEqual(node_named(self.panel, "MACH").checkState(0),
                         Qt.CheckState.Unchecked)

    def test_a_row_carries_the_value_it_stands_for(self):
        # Not its label. Two decimals can render distinct values identically --
        # 0.751 and 0.752 both read "0.75" -- and matching a click back by text
        # would toggle whichever came first.
        self.panel._state.iso_state("MACH")
        self.panel._rebuild_tree()
        node = node_named(self.panel, "MACH")
        kinds = [node.child(i).data(0, Qt.ItemDataRole.UserRole)[2][0]
                 for i in range(node.childCount())]
        self.assertEqual(kinds, ["value", "value", "value"])

    def test_values_that_render_alike_are_still_told_apart(self):
        panel = make_iso_panel({"NEAR": [0.751, 0.752]})
        iso = panel._state.iso_state("NEAR")
        panel._rebuild_tree()
        node = node_named(panel, "NEAR")

        node.child(1).setCheckState(0, Qt.CheckState.Unchecked)

        values = panel._distinct("NEAR")
        self.assertFalse(iso.value_enabled[values[1]])
        self.assertTrue(iso.value_enabled.get(values[0], True))

    def test_intervals_are_ticked_by_position_not_by_name(self):
        # Two intervals may legitimately be called the same thing; the row
        # knows which one it is.
        iso = self.panel._state.iso_state("MACH")
        iso.grouped = True
        iso.groups = [("band", 0.6, 0.75, True), ("band", 0.75, 0.9, True)]
        self.panel._rebuild_tree()

        node_named(self.panel, "MACH").child(1).setCheckState(
            0, Qt.CheckState.Unchecked)

        self.assertEqual([g[3] for g in iso.groups], [True, False])

    def test_ticking_a_value_reaches_the_state(self):
        iso = self.panel._state.iso_state("MACH")
        self.panel._rebuild_tree()
        node = node_named(self.panel, "MACH")

        node.child(1).setCheckState(0, Qt.CheckState.Unchecked)

        values = self.panel._distinct("MACH")
        self.assertFalse(iso.value_enabled[values[1]])
        self.assertTrue(iso.value_enabled.get(values[0], True))

    def test_ticking_a_value_switches_its_filter_on(self):
        # Choosing which values to show is meaningless while the filter is off.
        iso = self.panel._state.iso_state("MACH")
        iso.active = False
        self.panel._rebuild_tree()
        node_named(self.panel, "MACH").child(0).setCheckState(
            0, Qt.CheckState.Unchecked)
        self.assertTrue(iso.active)

    def test_the_parameter_checkbox_toggles_the_filter(self):
        iso = self.panel._state.iso_state("MACH")
        iso.active = True
        self.panel._rebuild_tree()
        node_named(self.panel, "MACH").setCheckState(0, Qt.CheckState.Unchecked)
        self.assertFalse(iso.active)

    def test_grouped_parameters_list_their_intervals(self):
        iso = self.panel._state.iso_state("MACH")
        iso.grouped = True
        iso.groups = [("low", 0.6, 0.75, True), ("high", 0.75, 0.9, False)]
        self.panel._rebuild_tree()

        node = node_named(self.panel, "MACH")
        self.assertEqual(child_labels(node), ["low", "high"])
        self.assertIn("grouped", node.text(0))
        self.assertEqual(node.child(0).checkState(0), Qt.CheckState.Checked)
        self.assertEqual(node.child(1).checkState(0), Qt.CheckState.Unchecked)

    def test_ticking_an_interval_reaches_the_state(self):
        iso = self.panel._state.iso_state("MACH")
        iso.grouped = True
        iso.groups = [("low", 0.6, 0.75, True), ("high", 0.75, 0.9, True)]
        self.panel._rebuild_tree()

        node_named(self.panel, "MACH").child(1).setCheckState(
            0, Qt.CheckState.Unchecked)

        self.assertEqual([g[3] for g in iso.groups], [True, False])

    def test_grouping_generates_its_intervals_when_asked_for(self):
        # iso.grouped can be set with no intervals behind it -- a workspace
        # saved mid-edit -- and a grouped parameter with none matches nothing.
        iso = self.panel._state.iso_state("MACH")
        iso.grouped = True
        iso.group_count = 2
        iso.groups = []
        self.panel._rebuild_tree()
        self.assertEqual(len(iso.groups), 2)
        self.assertEqual(node_named(self.panel, "MACH").childCount(), 2)

    def test_expansion_survives_a_rebuild(self):
        # A tree that closed every node whenever a file was loaded would be
        # unusable.
        self.panel._state.iso_state("MACH")
        self.panel._expanded.add("MACH")
        self.panel._rebuild_tree()
        self.assertTrue(node_named(self.panel, "MACH").isExpanded())

    def test_a_long_parameter_is_capped_and_says_so(self):
        many = [float(i) for i in range(panels.ISO_TREE_MAX_CHILDREN + 25)]
        panel = make_iso_panel({"SWEEP": many})
        panel._state.iso_state("SWEEP")
        panel._rebuild_tree()

        node = node_named(panel, "SWEEP")
        self.assertEqual(node.childCount(), panels.ISO_TREE_MAX_CHILDREN + 1)
        self.assertIn("25 more", node.child(node.childCount() - 1).text(0))
        # The hidden ones are still filtered on: their ticks live in the state,
        # not in the tree.
        self.assertIn("still filtered on",
                      node.child(node.childCount() - 1).text(0))


class TestIsoTreeAndChangingData(unittest.TestCase):
    """Files come and go under a filter that was set up for the old ones."""

    def test_a_parameter_no_longer_in_any_file_is_marked(self):
        panel = make_iso_panel({"MACH": [0.7, 0.8]})
        panel._state.iso_state("GONE")
        panel._rebuild_tree()

        node = node_named(panel, "GONE")
        self.assertEqual(node.childCount(), 0)
        self.assertIn("not in the loaded files", node.text(0))

    def test_it_is_kept_rather_than_dropped(self):
        # Removing a filter is the user's call. A file being closed for a moment
        # must not throw away the values they ticked.
        panel = make_iso_panel({"MACH": [0.7, 0.8]})
        iso = panel._state.iso_state("GONE")
        iso.value_enabled[1.0] = False
        panel._rebuild_tree()
        self.assertIn("GONE", panel._state.iso_states)
        self.assertFalse(panel._state.iso_states["GONE"].value_enabled[1.0])

    def test_values_outside_a_grouping_are_reported(self):
        # The usual way here is loading a file whose values fall outside a
        # grouping set up for the old ones. Those runs are filtered out, and
        # without this the plot just comes back short.
        panel = make_iso_panel({"MACH": [0.7, 0.8, 2.5]})
        iso = panel._state.iso_state("MACH")
        iso.grouped = True
        iso.groups = [("low", 0.6, 0.9, True)]
        panel._rebuild_tree()
        self.assertIn("1 outside the groups", node_named(panel, "MACH").text(0))

    def test_a_grouping_that_covers_everything_is_not_reported(self):
        panel = make_iso_panel({"MACH": [0.7, 0.8]})
        iso = panel._state.iso_state("MACH")
        iso.grouped = True
        iso.groups = [("all", 0.6, 0.9, True)]
        panel._rebuild_tree()
        self.assertNotIn("outside", node_named(panel, "MACH").text(0))

    def test_refresh_forgets_the_cached_values(self):
        # The cache is only safe because refresh() is the one thing that runs
        # when the files change.
        panel = make_iso_panel({"MACH": [0.7, 0.8]})
        panel._state.iso_state("MACH")
        panel._rebuild_tree()
        before = panel._session.distinct_calls

        panel._session._values["MACH"] = np.array([0.7, 0.8, 0.9],
                                                  dtype=np.float32)
        panel.refresh()

        self.assertGreater(panel._session.distinct_calls, before)
        self.assertEqual(child_labels(node_named(panel, "MACH")),
                         ["0.7", "0.8", "0.9"])

    def test_none_turns_off_values_never_ticked(self):
        # A value nobody has touched is absent from value_enabled, so a "None"
        # that only rewrote the dictionary left the new file's values showing.
        panel = make_iso_panel({"MACH": [0.7, 0.8, 0.9]})
        panel._state.iso_state("MACH")
        panel._set_all(False, "MACH")
        enabled = panel._state.iso_states["MACH"].value_enabled
        # Keyed by the values the session actually reports -- float32 widened
        # to double, which is not the literal typed here.
        self.assertEqual([enabled[v] for v in panel._distinct("MACH")],
                         [False, False, False])

    def test_all_turns_them_back_on(self):
        panel = make_iso_panel({"MACH": [0.7, 0.8]})
        iso = panel._state.iso_state("MACH")
        iso.value_enabled[panel._distinct("MACH")[0]] = False
        panel._set_all(True, "MACH")
        self.assertTrue(all(iso.value_enabled.values()))


class TestIsoTreePerformance(unittest.TestCase):
    """Reading a parameter's distinct values is a scan of every open file."""

    def test_one_read_per_parameter_per_rebuild(self):
        panel = make_iso_panel()
        panel._state.iso_state("MACH")
        panel._state.iso_state("WEIGHT")
        panel._rebuild_tree()
        self.assertEqual(panel._session.distinct_calls, 2)

    def test_ticking_a_value_reads_nothing_new(self):
        panel = make_iso_panel()
        panel._state.iso_state("MACH")
        panel._rebuild_tree()
        before = panel._session.distinct_calls

        node_named(panel, "MACH").child(0).setCheckState(
            0, Qt.CheckState.Unchecked)

        self.assertEqual(panel._session.distinct_calls, before)

    def test_an_unchanged_refresh_does_not_rebuild(self):
        # A refresh arrives for reasons that change none of this -- a page
        # switched, the panel simply shown again -- and a rebuild throws away
        # and recreates one widget item per value.
        panel = make_iso_panel()
        panel._state.iso_state("MACH")
        panel._rebuild_tree()
        node = node_named(panel, "MACH")

        panel._rebuild_tree()

        self.assertIs(node_named(panel, "MACH"), node)

    def test_a_changed_selection_is_written_in_place(self):
        # It used to rebuild, and that is what made the tree blink every time a
        # value was toggled: ticking a box told the window the selection had
        # changed, the window refreshed the visible panel -- this one -- and
        # several hundred widget items were destroyed and remade under the
        # cursor, taking the scroll position with them. Only the ticks moved,
        # so only the ticks are written.
        panel = make_iso_panel()
        iso = panel._state.iso_state("MACH")
        panel._rebuild_tree()
        node = node_named(panel, "MACH")
        child = node.child(0)

        iso.value_enabled[panel._distinct("MACH")[0]] = False
        panel._rebuild_tree()

        self.assertIs(node_named(panel, "MACH"), node)
        self.assertIs(node_named(panel, "MACH").child(0), child)
        self.assertEqual(child.checkState(0), Qt.CheckState.Unchecked)

    def test_the_badge_follows_the_ticks_that_moved(self):
        # The heading counts what is ticked, so it is as stale as the checkbox.
        # An in-place update that left "12 of 12" over eleven ticked values
        # would be worse than the rebuild it replaced.
        panel = make_iso_panel()
        iso = panel._state.iso_state("MACH")
        panel._rebuild_tree()
        before = node_named(panel, "MACH").text(0)

        iso.value_enabled[panel._distinct("MACH")[0]] = False
        panel._rebuild_tree()

        self.assertNotEqual(node_named(panel, "MACH").text(0), before)

    def test_a_new_parameter_still_rebuilds(self):
        # The in-place path is only for ticks. A parameter arriving changes
        # which rows exist, and no amount of checkbox writing produces a row
        # that is not there.
        panel = make_iso_panel()
        panel._state.iso_state("MACH")
        panel._rebuild_tree()
        node = node_named(panel, "MACH")

        panel._state.iso_state("WEIGHT")
        panel._rebuild_tree()

        self.assertIsNot(node_named(panel, "MACH"), node)
        self.assertIsNotNone(node_named(panel, "WEIGHT"))

    def test_building_the_tree_does_not_fire_the_check_handler(self):
        # Every setCheckState during a rebuild raises itemChanged exactly as a
        # click does. Unguarded, building the tree would write every box back
        # into the state and switch on every filter it drew.
        panel = make_iso_panel()
        iso = panel._state.iso_state("MACH")
        iso.active = False
        panel._rebuild_tree()
        self.assertFalse(iso.active)


class TestIsoSelectionLabels(unittest.TestCase):
    """Reading the selection back out of the panel's state."""

    def setUp(self):
        self.panel = make_iso_panel()

    def test_untouched_values_count_as_selected(self):
        self.panel._state.iso_state("MACH")
        labels, selected = self.panel._labels_for("MACH")
        self.assertEqual(labels, ["0.7", "0.75", "0.8"])
        self.assertEqual(selected, labels)

    def test_it_follows_what_was_unticked(self):
        iso = self.panel._state.iso_state("MACH")
        iso.value_enabled[self.panel._distinct("MACH")[1]] = False
        labels, selected = self.panel._labels_for("MACH")
        self.assertEqual(selected, ["0.7", "0.8"])

    def test_grouped_mode_reports_the_intervals(self):
        # Grouped, the user ticks named intervals; the underlying numbers are
        # not what is on screen, so they are not what is reported.
        iso = self.panel._state.iso_state("MACH")
        iso.grouped = True
        iso.groups = [("low", 0.0, 0.75, True), ("high", 0.75, 1.0, False)]
        labels, selected = self.panel._labels_for("MACH")
        self.assertEqual(labels, ["low", "high"])
        self.assertEqual(selected, ["low"])

    def test_a_parameter_with_no_filter_is_empty(self):
        self.assertEqual(self.panel._labels_for("MACH"), ([], []))


class TestIsoGroupArithmetic(unittest.TestCase):
    """Editing intervals as edges, which is what makes gaps unrepresentable.

    A set of independent intervals lets two of them overlap, or a gap open
    between them that silently drops every run in it. Neither shows up until a
    plot comes back short.
    """

    def groups(self, *edges):
        return isogroups.groups_from_edges(list(edges))

    def test_edges_round_trip(self):
        groups = self.groups(0.0, 1.0, 2.0, 3.0)
        self.assertEqual(isogroups.edges_of(groups), [0.0, 1.0, 2.0, 3.0])
        self.assertTrue(isogroups.is_contiguous(groups))

    def test_moving_an_edge_moves_both_intervals_that_share_it(self):
        edges = isogroups.move_edge([0.0, 1.0, 2.0], 1, 1.5)
        groups = self.groups(*edges)
        self.assertEqual([(g[1], g[2]) for g in groups],
                         [(0.0, 1.5), (1.5, 2.0)])
        self.assertTrue(isogroups.is_contiguous(groups))

    def test_an_edge_cannot_pass_its_neighbour(self):
        # Dragging one past the next would reorder the groups under the row
        # being edited, which is never what dragging a boundary means.
        self.assertEqual(isogroups.move_edge([0.0, 1.0, 2.0], 1, 99.0),
                         [0.0, 2.0, 2.0])
        self.assertEqual(isogroups.move_edge([0.0, 1.0, 2.0], 1, -99.0),
                         [0.0, 0.0, 2.0])

    def test_the_outer_edges_are_free(self):
        # Pulling one in is how you deliberately exclude an outlier.
        self.assertEqual(isogroups.move_edge([0.0, 1.0, 2.0], 0, -50.0),
                         [-50.0, 1.0, 2.0])
        self.assertEqual(isogroups.move_edge([0.0, 1.0, 2.0], 2, 50.0),
                         [0.0, 1.0, 50.0])

    def test_a_typo_changes_nothing(self):
        for bad in ("", None, float("nan"), float("inf")):
            self.assertEqual(isogroups.move_edge([0.0, 1.0, 2.0], 1, bad),
                             [0.0, 1.0, 2.0])
        self.assertEqual(isogroups.move_edge([0.0, 1.0, 2.0], 9, 0.5),
                         [0.0, 1.0, 2.0])

    def test_the_group_count_cannot_exceed_the_values(self):
        # Past that the extra intervals are empty by construction, and an empty
        # interval is a filter row that can never match anything.
        self.assertEqual(isogroups.clamp_group_count(10, 3), 3)
        self.assertEqual(isogroups.clamp_group_count(2, 3), 2)
        self.assertEqual(isogroups.clamp_group_count(0, 3), 1)
        self.assertEqual(isogroups.clamp_group_count(5, 0), 1)

    def test_even_edges_span_the_observed_range(self):
        edges = isogroups.even_edges([1.0, 2.0, 5.0, 9.0], 4)
        self.assertEqual(edges[0], 1.0)
        self.assertEqual(edges[-1], 9.0)
        self.assertEqual(len(edges), 5)

    def test_every_value_at_one_number_gives_one_interval(self):
        # Widening it would claim a range the data does not have.
        self.assertEqual(isogroups.even_edges([3.0, 3.0, 3.0], 4), [3.0, 3.0])

    def test_counts_match_how_the_filter_reads_the_bounds(self):
        # Inclusive at both ends, first group wins a shared edge -- the same
        # rule as the core's IsoGroup.contains, so the dialog cannot promise
        # something the plot then contradicts.
        groups = self.groups(0.0, 1.0, 2.0)
        self.assertEqual(isogroups.counts_per_group(groups, [0.0, 1.0, 1.5, 2.0]),
                         [2, 2])

    def test_values_outside_are_found(self):
        groups = self.groups(0.0, 1.0)
        self.assertEqual(isogroups.values_outside(groups, [-1.0, 0.5, 7.0]),
                         [-1.0, 7.0])
        self.assertEqual(isogroups.values_outside(groups, [0.0, 1.0]), [])

    def test_non_contiguous_input_still_yields_usable_edges(self):
        # A workspace saved before edges existed, or one whose values moved
        # under it. move_edge() has to have something coherent to work from.
        ragged = [("a", 5.0, 6.0, True), ("b", 0.0, 1.0, True)]
        self.assertEqual(isogroups.edges_of(ragged), [0.0, 5.0, 6.0])

    def test_names_and_flags_are_kept_by_position(self):
        groups = isogroups.groups_from_edges(
            [0.0, 1.0, 2.0], names=["low", "high"], active=[True, False])
        self.assertEqual([g[0] for g in groups], ["low", "high"])
        self.assertEqual([g[3] for g in groups], [True, False])

    def test_default_names_follow_their_bounds(self):
        renamed = isogroups.rename_defaults(
            [("stale", 0.0, 9.0, True)], "{0:g}".format)
        self.assertEqual(renamed[0][0], "0 - 9")


class _StubDialog(panels.IsoGroupDialog):
    """The dialog without a window, so its editing rules can be exercised."""

    def __init__(self, values, groups=None):
        session = _IsoSession({"MACH": values})
        object.__setattr__(self, "_dummy", None)
        self._param = "MACH"
        self._values = [float(v) for v in values]
        self._format = session.format_value
        self._auto = lambda count: session.iso_auto_group(
            np.asarray(self._values, dtype=np.float32), count)
        self._updating = False
        self._custom_names = set()
        self.table = qt_stubs.QTableWidget(0, 4)
        self.note = qt_stubs.QLabel()
        self.count_spin = qt_stubs.QSpinBox()
        self._groups = [tuple(g) for g in (groups or [])]
        if not self._groups:
            self._groups = self._even(min(3, max(1, len(self._values))))
        self._fill_table()

    def edit(self, row, column, text):
        """Type into a cell, as the table's itemChanged would deliver it."""
        item = self.table.item(row, column)
        item.setText(text)
        self._on_item_changed(item)


class TestIsoGroupDialog(unittest.TestCase):
    """The grouping editor's rules, without a window."""

    def test_it_opens_on_an_even_split(self):
        dialog = _StubDialog([0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(len(dialog.groups()), 3)
        self.assertTrue(isogroups.is_contiguous(dialog.groups()))

    def test_it_opens_on_the_existing_grouping(self):
        # Opening the dialog on a grouping edited last week must show that
        # grouping, not replace it with a fresh even split.
        existing = [("mine", 0.0, 2.0, True), ("theirs", 2.0, 4.0, False)]
        dialog = _StubDialog([0.0, 1.0, 2.0, 3.0, 4.0], existing)
        self.assertEqual([g[0] for g in dialog.groups()], ["mine", "theirs"])
        self.assertEqual([g[3] for g in dialog.groups()], [True, False])

    def test_changing_the_count_redistributes(self):
        dialog = _StubDialog([0.0, 1.0, 2.0, 3.0, 4.0])
        dialog._on_count_changed(4)
        self.assertEqual(len(dialog.groups()), 4)
        self.assertTrue(isogroups.is_contiguous(dialog.groups()))

    def test_editing_a_boundary_moves_its_neighbour_too(self):
        dialog = _StubDialog([0.0, 1.0, 2.0, 3.0, 4.0])
        dialog._on_count_changed(2)
        dialog.edit(0, dialog.TO, "3")
        groups = dialog.groups()
        self.assertEqual((groups[0][2], groups[1][1]), (3.0, 3.0))
        self.assertTrue(isogroups.is_contiguous(groups))

    def test_editing_a_from_cell_moves_the_row_above(self):
        dialog = _StubDialog([0.0, 1.0, 2.0, 3.0, 4.0])
        dialog._on_count_changed(2)
        dialog.edit(1, dialog.FROM, "1")
        groups = dialog.groups()
        self.assertEqual((groups[0][2], groups[1][1]), (1.0, 1.0))

    def test_a_boundary_cannot_pass_its_neighbour(self):
        dialog = _StubDialog([0.0, 1.0, 2.0, 3.0, 4.0])
        dialog._on_count_changed(2)
        dialog.edit(0, dialog.TO, "999")
        self.assertTrue(isogroups.is_contiguous(dialog.groups()))
        self.assertLessEqual(dialog.groups()[0][2], 4.0)

    def test_nonsense_in_a_cell_is_ignored(self):
        dialog = _StubDialog([0.0, 1.0, 2.0, 3.0, 4.0])
        before = dialog.groups()
        dialog.edit(0, dialog.TO, "banana")
        self.assertEqual(dialog.groups(), before)

    def test_a_default_name_follows_its_bounds(self):
        dialog = _StubDialog([0.0, 1.0, 2.0, 3.0, 4.0])
        dialog._on_count_changed(2)
        dialog.edit(0, dialog.TO, "3")
        self.assertIn("3", dialog.groups()[0][0])

    def test_a_name_the_user_typed_is_kept(self):
        dialog = _StubDialog([0.0, 1.0, 2.0, 3.0, 4.0])
        dialog._on_count_changed(2)
        dialog.edit(0, dialog.NAME, "cruise")
        dialog.edit(0, dialog.TO, "3")
        self.assertEqual(dialog.groups()[0][0], "cruise")

    def test_clearing_a_name_gives_the_default_back(self):
        dialog = _StubDialog([0.0, 1.0, 2.0, 3.0, 4.0])
        dialog._on_count_changed(2)
        dialog.edit(0, dialog.NAME, "cruise")
        dialog.edit(0, dialog.NAME, "  ")
        self.assertNotEqual(dialog.groups()[0][0], "cruise")

    def test_the_count_column_says_what_each_interval_claims(self):
        dialog = _StubDialog([0.0, 1.0, 2.0, 3.0, 4.0])
        dialog._on_count_changed(2)
        counts = [int(dialog.table.item(row, dialog.COUNT).text())
                  for row in range(len(dialog.groups()))]
        self.assertEqual(sum(counts), 5)

    def test_an_excluded_outlier_is_reported(self):
        dialog = _StubDialog([0.0, 1.0, 2.0, 3.0, 100.0])
        dialog._on_count_changed(1)
        dialog.edit(0, dialog.TO, "3")
        self.assertIn("outside", dialog.note.text())

    def test_full_coverage_says_so(self):
        dialog = _StubDialog([0.0, 1.0, 2.0, 3.0, 4.0])
        dialog._on_count_changed(2)
        self.assertIn("exactly one group", dialog.note.text())

    def test_distribute_evenly_discards_manual_edits(self):
        dialog = _StubDialog([0.0, 1.0, 2.0, 3.0, 4.0])
        dialog._on_count_changed(2)
        dialog.edit(0, dialog.TO, "3.5")
        dialog.count_spin.setValue(2)
        dialog._distribute_evenly()
        self.assertEqual(dialog.groups()[0][2], 2.0)

    def test_a_single_value_still_gives_a_row_to_edit(self):
        # Otherwise there is no row, nothing to edit, and no way back.
        dialog = _StubDialog([7.0])
        self.assertEqual(len(dialog.groups()), 1)

    def test_removing_the_grouping_is_reported_to_the_caller(self):
        dialog = _StubDialog([0.0, 1.0, 2.0])
        self.assertFalse(dialog.ungrouped)
        dialog._drop_grouping()
        self.assertTrue(dialog.ungrouped)


class TestUngrouping(unittest.TestCase):
    """Going back to ticking values one by one."""

    def setUp(self):
        self.panel = make_iso_panel()
        self.iso = self.panel._state.iso_state("MACH")
        self.iso.grouped = True
        self.iso.groups = [("low", 0.6, 0.75, True), ("high", 0.75, 0.9, False)]

    def test_it_shows_the_values_again(self):
        self.panel._set_grouped("MACH", False)
        self.assertEqual(child_labels(node_named(self.panel, "MACH")),
                         ["0.7", "0.75", "0.8"])

    def test_the_intervals_are_kept(self):
        # Switching grouping back on should return to what was set up, not to a
        # fresh even split.
        self.panel._set_grouped("MACH", False)
        self.assertEqual(len(self.iso.groups), 2)

        self.panel._set_grouped("MACH", True)
        node = node_named(self.panel, "MACH")
        self.assertEqual(child_labels(node), ["low", "high"])
        self.assertEqual(node.child(1).checkState(0), Qt.CheckState.Unchecked)

    def test_ungrouped_ticks_survive_a_trip_through_grouping(self):
        self.panel._set_grouped("MACH", False)
        values = self.panel._distinct("MACH")
        self.iso.value_enabled[values[0]] = False

        self.panel._set_grouped("MACH", True)
        self.panel._set_grouped("MACH", False)

        node = node_named(self.panel, "MACH")
        self.assertEqual(node.child(0).checkState(0), Qt.CheckState.Unchecked)


class TestInspectDot(unittest.TestCase):
    """The mark showing which sample the Inspect panel is reporting.

    The panel lists forty channels of one run at one instant, and the click
    lands on whichever curve was nearest -- which is not always the one you were
    aiming at. A table of numbers cannot tell you that; a dot on the plot can.
    """

    def setUp(self):
        self.tile = make_tile()
        self.tile._inspect_dot = qt_stubs.PlotDataItem()
        self.tile._inspect_dot.setVisible(False)

    def test_it_appears_at_the_sample(self):
        self.tile._show_inspect_dot(3.0, 7.0)
        self.assertTrue(self.tile._inspect_dot.opts["visible"])
        self.assertEqual(self.tile._inspect_dot.data_calls[-1], ([3.0], [7.0]))

    def test_clearing_hides_it(self):
        self.tile._show_inspect_dot(3.0, 7.0)
        self.tile.clear_inspect_dot()
        self.assertFalse(self.tile._inspect_dot.opts["visible"])

    def test_clearing_a_tile_that_never_had_one_is_harmless(self):
        # Class-level default: the early resizeEvent pyqtgraph fires during
        # construction can reach this before __init__ has made the item.
        tile = make_tile()
        tile._inspect_dot = None
        tile.clear_inspect_dot()          # must not raise

    def test_leaving_inspect_takes_the_mark_off(self):
        # Outside Inspect the panel is not reporting anything, so a dot still
        # sitting on a sample is a claim that is no longer true.
        from aoeviewer.model import MouseMode

        self.tile._show_inspect_dot(1.0, 2.0)
        self.tile._state.mouse_mode = MouseMode.PAN
        self.tile.apply_mouse_mode()
        self.assertFalse(self.tile._inspect_dot.opts["visible"])

    def test_staying_in_inspect_leaves_it(self):
        from aoeviewer.model import MouseMode

        self.tile._state.mouse_mode = MouseMode.CROSSHAIR
        self.tile._show_inspect_dot(1.0, 2.0)
        self.tile.apply_mouse_mode()
        self.assertTrue(self.tile._inspect_dot.opts["visible"])


class TestProjectionCache(unittest.TestCase):
    """Turning a 3D cube must not rebuild it.

    rotate_3d() called rebuild() on every mouse-move of a drag: three channels
    re-read from the core per run, their bounds recomputed, the samples
    re-thinned, every draw item destroyed and recreated -- to change two
    angles. Nothing but the projection depends on the angle.
    """

    def setUp(self):
        self.tile = make_tile()
        self.tile._gizmo = qt_stubs._Any()
        self.tile._texts = None
        self.tile.tile_state.plot3d = True
        self.tile.tile_state.x_param = "TIME"
        self.tile.tile_state.y_params = ["ALT"]
        self.tile.tile_state.z_param = "MACH"

        cube = np.linspace(0.0, 1.0, 64, dtype=np.float32)
        self.group = pt.DrawGroup(qt_stubs.PlotDataItem())
        member = pt._ProjectedCurve(
            "0|0|MACH", cube, cube, cube,
            np.linspace(10.0, 20.0, 64, dtype=np.float32), "run", "sim::a")
        self.tile._projection = pt._Projection(
            ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), [(self.group, [member])])

    def test_without_a_cache_it_declines(self):
        # The caller falls back to a full rebuild, which is what builds one.
        self.tile._projection = None
        self.assertFalse(self.tile._reproject_3d())

    def test_on_a_2d_tile_it_declines(self):
        self.tile.tile_state.plot3d = False
        self.assertFalse(self.tile._reproject_3d())

    def test_it_redraws_without_replacing_the_items(self):
        plot = self.tile.getPlotItem()
        plot.added = []
        plot.removed = []

        self.assertTrue(self.tile._reproject_3d())

        self.assertGreater(len(self.group.item.data_calls), 0)
        # The curve item is written to, not rebuilt. Whatever the grid adds is
        # its own business -- the box turns with the cube.
        self.assertNotIn(self.group.item, plot.added)
        self.assertNotIn(self.group.item, plot.removed)

    def test_a_new_angle_gives_new_points(self):
        self.tile.tile_state.azimuth = 0.0
        self.tile._reproject_3d()
        first = self.group.item.data_calls[-1]

        self.tile.tile_state.azimuth = 90.0
        self.tile._reproject_3d()
        second = self.group.item.data_calls[-1]

        self.assertFalse(np.allclose(first[0], second[0]))

    def test_the_curve_entries_follow(self):
        # They hold the projected coordinates, which is what a click is
        # hit-tested against -- so a stale one would inspect the old angle.
        self.tile._projection.groups[0][1][0]
        self.tile.tile_state.azimuth = 0.0
        self.tile._reproject_3d()
        before = self.tile._curves["0|0|MACH"].x.copy()

        self.tile.tile_state.azimuth = 90.0
        self.tile._reproject_3d()

        self.assertFalse(np.allclose(before, self.tile._curves["0|0|MACH"].x))

    def test_the_entry_carries_the_x_parameter_too(self):
        self.tile._reproject_3d()
        entry = self.tile._curves["0|0|MACH"]
        self.assertIsNotNone(entry.source_x)
        self.assertAlmostEqual(float(entry.source_x[0]), 10.0)
        self.assertAlmostEqual(float(entry.source_x[-1]), 20.0)


class TestInspectOnAProjection(unittest.TestCase):
    """A click on a 3D tile has to be reported in the x parameter's units.

    `x` there holds projected screen coordinates, because that is what the
    click lands on. Reporting one of those meant the Inspect panel searched the
    TIME column for a projection coordinate.
    """

    def setUp(self):
        self.tile = make_tile()
        self.tile.tile_state.plot3d = True
        # A path that doubles back, which is the normal shape of a trajectory
        # and the thing a binary search cannot handle.
        projected_x = np.array([0.0, 1.0, 2.0, 1.0, 0.0], dtype=np.float32)
        projected_y = np.array([0.0, 1.0, 0.0, -1.0, 0.0], dtype=np.float32)
        entry = pt.CurveEntry(projected_x, projected_y, "run", "sim::a",
                              "0|0|MACH")
        entry.source_x = np.array([10.0, 11.0, 12.0, 13.0, 14.0],
                                  dtype=np.float32)
        self.tile._curves = {"0|0|MACH": entry}

        # _inspect_at maps the click through Qt's scene, which the stubs have
        # no coordinate system for. The point is handed through unchanged, so
        # the test can express the click directly in view coordinates.
        self.tile.mapToScene = lambda point: point
        self.tile.sceneBoundingRect = lambda: _AlwaysContains()
        self.tile.getPlotItem().vb.mapSceneToView = lambda point: point
        self.tile.getPlotItem().vb._range = [[-1.0, 3.0], [-2.0, 2.0]]

    def test_the_reported_value_is_in_source_units(self):
        seen = []
        self.tile.point_inspected.connect(
            lambda f, s, x: seen.append((f, s, x)))
        # Aim at the fourth point of the path: projected (1, -1), which a
        # search on x alone cannot tell from the second point at (1, 1).
        self.tile._inspect_at(_ScenePoint(1.0, -1.0))

        self.assertEqual(len(seen), 1)
        self.assertAlmostEqual(seen[0][2], 13.0, places=4)

    def test_the_dot_lands_in_drawn_coordinates(self):
        self.tile._inspect_dot = qt_stubs.PlotDataItem()
        self.tile._inspect_at(_ScenePoint(1.0, -1.0))
        x, y = self.tile._inspect_dot.data_calls[-1]
        self.assertAlmostEqual(x[0], 1.0, places=4)
        self.assertAlmostEqual(y[0], -1.0, places=4)


class _AlwaysContains:
    """A scene rect that accepts every click, for tests with no real scene."""

    def contains(self, _point):
        return True


class _ScenePoint:
    """A click position the tile can map, with the stubs standing in for Qt.

    PlotTile._inspect_at goes position -> scene -> view; with no real scene the
    stubs cannot do that, so the tile is patched in make_tile()'s spirit: the
    point reports itself as already being in view coordinates.
    """

    def __init__(self, x, y):
        self._x, self._y = float(x), float(y)

    def x(self):
        return self._x

    def y(self):
        return self._y

    def toPoint(self):
        return self


class TestLabelGround(unittest.TestCase):
    """Readouts drawn over the curves need something behind them.

    Text drawn straight onto the data was dark on dark wherever a trace ran
    under it, and the eye read the two as one object.
    """

    def test_the_ground_is_the_plot_background(self):
        # Not a colour of its own: the label has to read as a card laid on the
        # plot rather than as another colour in the data.
        brush = pt._label_brush("#0e1013")
        self.assertIsNotNone(brush)

    def test_it_is_not_fully_opaque(self):
        # A little of the trace showing through is what keeps the label part of
        # the same picture instead of a hole punched in it.
        self.assertLess(pt.LABEL_GROUND_ALPHA, 1.0)
        self.assertGreater(pt.LABEL_GROUND_ALPHA, 0.7)

    def test_there_is_padding_between_text_and_edge(self):
        # Also the slack that keeps a one-word caption easy to grab.
        self.assertGreater(pt.LABEL_PAD, 0.0)


class TestAnnotationsHaveNoCaptions(unittest.TestCase):
    """Markers and bands lost their own labels, and had to.

    pyqtgraph builds an InfLineLabel from `label` and forwards labelOpts
    straight to TextItem, which accepts neither a CSS "font-size" nor much
    else. Passing one raised on every rebuild -- and an exception inside
    _draw_overlays aborted PlotGrid.refresh() partway through the page, which
    is why a dragged band moved on the tile under the cursor and nowhere else.
    """

    def test_no_marker_carries_a_caption(self):
        from aoeviewer.model import AreaBand, Marker

        self.assertFalse(hasattr(Marker(), "label"))
        self.assertFalse(hasattr(AreaBand(), "label"))

    def test_nothing_passes_label_options_to_pyqtgraph(self):
        source = (os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "python", "aoeviewer", "ui", "plot_tile.py"))
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        # The keyword, not the word: the comment explaining why it is gone
        # naturally names it.
        self.assertNotIn("labelOpts=", text)
        self.assertNotIn("label=marker.label", text)


class TestTypeScale(unittest.TestCase):
    """The stylesheet is a format string, and a mistake in one is silent.

    An unknown placeholder raises only when a theme is applied -- during window
    construction, or the moment someone toggles light and dark -- and takes the
    window down with it. Formatting both themes here costs nothing and turns
    that into a test failure.
    """

    def test_both_themes_format(self):
        # Calling it is most of the test: a brace that should have been doubled,
        # or a placeholder naming a key the theme does not have, raises here.
        for theme in (theme_module.DARK, theme_module.LIGHT):
            css = theme_module.stylesheet(theme)
            self.assertIn(theme.window, css)
            self.assertIn(theme.accent, css)

    def test_the_scale_reaches_the_stylesheet(self):
        css = theme_module.stylesheet(theme_module.DARK)
        self.assertIn("font-size: {0}px".format(theme_module.FONT_PX), css)
        self.assertIn("font-size: {0}px".format(theme_module.FONT_SMALL_PX), css)

    def test_small_is_smaller_than_body(self):
        self.assertLess(theme_module.FONT_SMALL_PX, theme_module.FONT_PX)

    def test_the_tiles_stay_in_step_with_the_panels(self):
        # Points, not pixels, and roughly 0.75x at the 96 dpi Qt assumes. A tile
        # title larger than a panel heading is the thing this guards against:
        # the plots are the content, but their furniture is not.
        as_px = theme_module.PLOT_TITLE_PT * 4.0 / 3.0
        self.assertLessEqual(as_px, theme_module.FONT_PX)
        for smaller in (theme_module.PLOT_AXIS_PT, theme_module.PLOT_LEGEND_PT,
                        theme_module.PLOT_LABEL_PT):
            self.assertLessEqual(smaller, theme_module.PLOT_TITLE_PT)

    def test_plot_font_is_asked_for_by_size_not_read_back(self):
        # The whole reason plot_font() exists: the stylesheet sizes text in
        # pixels, so a QFont read back off a widget reports pointSizeF() == -1
        # and the drawing code was left guessing a fallback.
        font = theme_module.plot_font(theme_module.PLOT_LABEL_PT)
        self.assertGreater(font.pointSizeF(), 0)


class TestLegendGround(unittest.TestCase):
    """The legend sits on a card, like every other readout on a tile.

    It was the one thing still drawn straight onto the data: rows of text with
    curve-coloured swatches beside them, over curves of those same colours.
    Nothing said where the key stopped and the plot started.
    """

    def setUp(self):
        self.tile = make_tile()

    def _legend(self):
        plot = self.tile.getPlotItem()
        self.tile._ensure_legend(plot)
        return self.tile._legend

    def test_it_is_built_with_a_fill_and_a_border(self):
        legend = self._legend()
        self.assertIsNotNone(legend.opts["brush"])
        # Framed as well as filled: at this opacity, over an empty part of the
        # tile, a fill of the plot's own background colour has no visible edge.
        self.assertIsNotNone(legend.opts["pen"])

    def test_the_fill_is_more_opaque_than_a_caption(self):
        # A caption is a word the eye separates by its shape; the legend is a
        # block whose swatches are themselves drawn in the curve colours, so a
        # trace showing through it reads as one more entry.
        self.assertGreater(pt.LEGEND_GROUND_ALPHA, pt.LABEL_GROUND_ALPHA)
        self.assertLessEqual(pt.LEGEND_GROUND_ALPHA, 1.0)

    def test_the_text_takes_the_plot_foreground(self):
        # It is written on the plot background now, not on the data.
        legend = self._legend()
        self.assertEqual(legend.opts["labelTextColor"],
                         self.tile._theme.plot_foreground)

    def test_the_rows_are_not_flush_with_the_frame(self):
        legend = self._legend()
        self.assertEqual(legend.layout.getContentsMargins(),
                         (pt.LEGEND_PAD,) * 4)
        self.assertGreater(pt.LEGEND_PAD, 0)

    def test_it_sits_above_the_curves_it_explains(self):
        # An opaque card underneath the data would be worse than none: a trace
        # crossing it would draw a line through the swatches identifying it.
        legend = self._legend()
        self.assertGreater(legend.zValue(), pt.MARKER_Z)
        self.assertGreater(legend.zValue(), pt.AREA_Z)

    def test_a_theme_change_repaints_the_card(self):
        # _ensure_legend returns early once the legend exists, so without this
        # a dark card stayed behind the legend on a light plot.
        legend = self._legend()
        first = legend.opts["brush"]

        light = _Theme()
        light.plot_background = "#ffffff"
        light.plot_foreground = "#22262b"
        light.plot_grid = "#d5dbe1"
        self.tile._theme = light
        self.tile._ensure_legend(self.tile.getPlotItem())

        self.assertIs(legend, self.tile._legend)     # not rebuilt
        self.assertIsNot(legend.opts["brush"], first)
        self.assertEqual(legend.opts["labelTextColor"], "#22262b")

    def test_the_ground_follows_the_theme_rather_than_a_colour_of_its_own(self):
        for theme in (theme_module.DARK, theme_module.LIGHT):
            ground = pt.legend_ground(theme)
            self.assertEqual(ground["labelTextColor"], theme.plot_foreground)
            self.assertIsNotNone(ground["brush"])
            self.assertIsNotNone(ground["pen"])


class _Sim:
    def __init__(self, index, sim_id):
        self.index = index
        self.sim_id = sim_id
        self.name = sim_id
        self.key = "f0::" + sim_id


class _FileHandle:
    def __init__(self, index, name, sim_ids):
        self.index = index
        self.name = name
        self.path = "/study/" + name
        self.simulations = [_Sim(i, s) for i, s in enumerate(sim_ids)]
        self.simulation_count = len(self.simulations)


class _ListSession:
    """A session with files, runs and parameters, for the list panels."""

    def __init__(self, params=("TIME", "ALT", "MACH"), sims=("R1", "R2", "R3")):
        self.parameters = list(params)
        self.files = [_FileHandle(0, "flight_a.aoe", sims)]

    def file(self, index):
        return self.files[index] if 0 <= index < len(self.files) else None

    def label(self, name):
        return name + " [-]"

    def iso_distinct(self, name):
        return np.array([], dtype=np.float32)

    def format_value(self, value):
        return "{0:g}".format(value)


def make_data_panel(session=None):
    panel = object.__new__(panels.DataPanel)
    panel._session = session or _ListSession()
    panel._state = AppState()
    panel._updating = False
    panel._tree_state = None
    panel.tree = qt_stubs.QTreeWidget()
    panel.tree.itemChanged.connect(
        lambda item, col=0: panel._on_item_changed(item, col))
    return panel


def make_param_panel(session=None):
    panel = object.__new__(panels.ParameterPanel)
    panel._session = session or _ListSession()
    panel._state = AppState()
    panel._updating = False
    panel._list_state = None
    panel.x_combo = qt_stubs.QComboBox()
    panel.y_list = qt_stubs.QListWidget()
    panel.filter_edit = qt_stubs.QLineEdit()
    panel.y_list.itemChanged.connect(panel._on_y_changed)
    panel.x_combo.currentTextChanged.connect(panel._on_x_changed)
    return panel


def sim_rows(panel):
    rows = []
    for index in range(panel.tree.topLevelItemCount()):
        parent = panel.tree.topLevelItem(index)
        for child_index in range(parent.childCount()):
            rows.append(parent.child(child_index))
    return rows


class _DrawSession:
    """Enough of a session for PlotTile.rebuild() to run end to end.

    Everything it answers is constant except the ISO values, which is the
    point: those decide the legend, and they are deliberately given out of
    order so that "the order the runs were walked in" and "the order the groups
    belong in" are different answers.
    """

    parameters = ["TIME", "ALT", "MACH"]

    def __init__(self, iso_values=(18.0, 5.0, 12.0), runs=None):
        self._iso = list(iso_values)
        names = runs or ["R{0}".format(i) for i in range(len(self._iso))]
        self.files = [_FileHandle(0, "flight_a.aoe", names)]

    def file(self, index):
        return self.files[0] if index == 0 else None

    def label(self, name):
        return name

    def unit(self, _name):
        return ""

    def channel(self, _file, _sim, _name):
        return np.arange(10, dtype=np.float32)

    def iso_values(self, _file, _name):
        return np.array(self._iso, dtype=np.float32)

    def iso_distinct(self, _name):
        return np.array(sorted(self._iso), dtype=np.float32)

    def format_value(self, value):
        return "{0:g}".format(value)

    def decimate(self, x, y, _target, _range):
        return x, y


def make_drawable_tile(session=None):
    """A PlotTile that can actually run rebuild(), legend and all."""
    tile = make_tile()
    tile._session = session or _DrawSession()
    tile._slope_item = qt_stubs.PlotDataItem()
    tile._slope_text = qt_stubs.PlotDataItem()
    tile._crosshair_label = qt_stubs.PlotDataItem()
    tile._gizmo = qt_stubs.PlotDataItem()
    tile._texts = None
    tile._projection = None
    tile._inspect_dot = None
    tile._nav_action = None
    tile._suppress_range_event = False
    return tile


def legend_names(tile):
    legend = tile.getPlotItem().legend
    return [name for _item, name in legend.rows] if legend is not None else []


def file_node(panel, name):
    for index in range(panel.tree.topLevelItemCount()):
        node = panel.tree.topLevelItem(index)
        if node.text(0) == name:
            return node
    raise AssertionError("no file row named " + name)


def click_box(item, state):
    """Do what a click on a checkbox does: set it, and let itemChanged fire."""
    item.setCheckState(0, state)


class TestSelectingAWholeFile(unittest.TestCase):
    """One box per file, because Plot all is every file at once.

    With three studies open, "all of them" is rarely the question anyone is
    asking -- and the file row is both where you look to find out how much of a
    file is plotted and the obvious place to say "all of it". So it carries a
    real tristate box: ticked means every run, cleared means none, and partly
    ticked means the selection stops somewhere inside. The third state is the
    one that earns it; without it a half-selected file has to claim to be
    either on or off, and both are lies.
    """

    def setUp(self):
        self.session = _ListSession(sims=("R1", "R2", "R3"))
        self.session.files.append(_FileHandle(1, "flight_b.aoe", ("S1", "S2")))
        self.panel = make_data_panel(self.session)
        self.panel.refresh()

    def selected(self):
        return sorted(self.panel._state.selected_sims)

    def test_ticking_a_file_plots_all_of_its_runs(self):
        click_box(file_node(self.panel, "flight_a.aoe"), Qt.CheckState.Checked)
        self.assertEqual(self.selected(), [(0, 0), (0, 1), (0, 2)])

    def test_it_leaves_the_other_files_alone(self):
        # The whole point of the feature: Plot all already did every file.
        click_box(file_node(self.panel, "flight_b.aoe"), Qt.CheckState.Checked)
        self.assertEqual(self.selected(), [(1, 0), (1, 1)])

    def test_clearing_a_file_plots_none_of_its_runs(self):
        self.panel._set_all(True)
        click_box(file_node(self.panel, "flight_a.aoe"),
                  Qt.CheckState.Unchecked)
        self.assertEqual(self.selected(), [(1, 0), (1, 1)])

    def test_the_run_boxes_follow(self):
        click_box(file_node(self.panel, "flight_a.aoe"), Qt.CheckState.Checked)
        node = file_node(self.panel, "flight_a.aoe")
        for index in range(node.childCount()):
            self.assertEqual(node.child(index).checkState(0),
                             Qt.CheckState.Checked)

    def test_a_partly_selected_file_reads_as_partly_selected(self):
        self.panel._state.set_selected(0, 1, True)
        self.panel.refresh()
        self.assertEqual(file_node(self.panel, "flight_a.aoe").checkState(0),
                         Qt.CheckState.PartiallyChecked)
        self.assertEqual(file_node(self.panel, "flight_b.aoe").checkState(0),
                         Qt.CheckState.Unchecked)

    def test_a_fully_selected_file_reads_as_checked(self):
        self.panel._set_file(0, True)
        self.assertEqual(file_node(self.panel, "flight_a.aoe").checkState(0),
                         Qt.CheckState.Checked)

    def test_ticking_one_run_updates_the_file_row(self):
        # The row above now says something that may no longer be true, and it
        # is derived, so it has to be re-derived.
        node = file_node(self.panel, "flight_a.aoe")
        click_box(node.child(0), Qt.CheckState.Checked)
        self.assertEqual(node.checkState(0), Qt.CheckState.PartiallyChecked)

        click_box(node.child(1), Qt.CheckState.Checked)
        click_box(node.child(2), Qt.CheckState.Checked)
        self.assertEqual(node.checkState(0), Qt.CheckState.Checked)

        click_box(node.child(0), Qt.CheckState.Unchecked)
        self.assertEqual(node.checkState(0), Qt.CheckState.PartiallyChecked)

    def test_the_file_row_does_not_write_its_runs_back(self):
        # Re-deriving the parent sets a checkbox, which raises itemChanged
        # exactly as a click does. Unguarded, ticking one run would be read as
        # a click on its file row and select the other two.
        node = file_node(self.panel, "flight_a.aoe")
        click_box(node.child(0), Qt.CheckState.Checked)
        self.assertEqual(self.selected(), [(0, 0)])

    def test_selecting_a_whole_file_is_one_signal(self):
        # Not one per run: the window reloads and redraws on this, and a
        # fifty-run file would have done it fifty times.
        fired = []
        self.panel.selection_changed.connect(lambda: fired.append(1))
        self.panel._set_file(0, True)
        self.assertEqual(len(fired), 1)

    def test_an_empty_file_is_not_partly_anything(self):
        self.session.files.append(_FileHandle(2, "empty.aoe", ()))
        self.panel.refresh()
        self.assertEqual(file_node(self.panel, "empty.aoe").checkState(0),
                         Qt.CheckState.Unchecked)

    def test_a_file_that_has_gone_is_not_an_error(self):
        # The tree and the session part company for a moment when a file is
        # removed, and a derived value must not raise in that window.
        node = file_node(self.panel, "flight_b.aoe")
        self.session.files.pop()
        self.panel._sync_file_node(node)          # must not raise
        self.panel._set_file(1, True)             # must not raise

    def test_plot_all_still_means_every_file(self):
        self.panel._set_all(True)
        self.assertEqual(self.selected(),
                         [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)])
        for name in ("flight_a.aoe", "flight_b.aoe"):
            self.assertEqual(file_node(self.panel, name).checkState(0),
                             Qt.CheckState.Checked)

    def test_plot_none_still_means_every_file(self):
        self.panel._set_all(True)
        self.panel._set_all(False)
        self.assertEqual(self.selected(), [])
        for name in ("flight_a.aoe", "flight_b.aoe"):
            self.assertEqual(file_node(self.panel, name).checkState(0),
                             Qt.CheckState.Unchecked)

    def test_the_file_tick_is_not_part_of_the_tree_signature(self):
        # It is derived from the runs, so it can never be the reason the rows
        # have to be rebuilt -- and if it were, selecting a file would blink
        # the tree again.
        before = self.panel._tree_signature()
        self.panel._set_file(0, True)
        self.assertEqual(self.panel._tree_signature(), before)


class TestListsDoNotBlink(unittest.TestCase):
    """Ticking a box must not rebuild the list the box is in.

    This was a loop with a cost. A tick told the window the selection had
    changed; the window marked every panel stale and refreshed the visible one;
    the visible one is the panel that was just clicked, and its refresh cleared
    the widget and built it again. So every checkbox flickered its own list --
    and took the scroll position, the expanded state and the current item with
    it, which on a study of a few hundred runs means finding your place again
    on every single tick.

    The fix is to separate what the rows ARE from what is TICKED in them: only
    the second can change on a selection, and the second is writable in place.
    """

    def test_ticking_a_run_keeps_the_same_tree_items(self):
        panel = make_data_panel()
        panel.refresh()
        before = sim_rows(panel)

        panel._state.set_selected(0, 1, True)
        panel.refresh()

        self.assertEqual([id(row) for row in sim_rows(panel)],
                         [id(row) for row in before])

    def test_the_tick_still_arrives(self):
        # In place is not the same as not at all.
        panel = make_data_panel()
        panel.refresh()
        panel._state.set_selected(0, 2, True)
        panel.refresh()

        rows = sim_rows(panel)
        self.assertEqual(rows[2].checkState(0), Qt.CheckState.Checked)
        self.assertEqual(rows[0].checkState(0), Qt.CheckState.Unchecked)

    def test_an_unchanged_refresh_writes_no_checkbox(self):
        # Each box is written only when it is wrong, so a refresh arriving for
        # a reason that changed no selection paints nothing at all.
        panel = make_data_panel()
        panel._state.set_selected(0, 0, True)
        panel.refresh()
        rows = sim_rows(panel)
        before = [row.check_writes for row in rows]

        for _ in range(10):
            panel.refresh()

        self.assertEqual([row.check_writes for row in rows], before)

    def test_a_new_file_does_rebuild(self):
        # The in-place path is for ticks. A file arriving changes which rows
        # exist, and no amount of checkbox writing produces a row that is not
        # there.
        session = _ListSession()
        panel = make_data_panel(session)
        panel.refresh()
        before = sim_rows(panel)

        session.files.append(_FileHandle(1, "flight_b.aoe", ("R1",)))
        panel.refresh()

        self.assertEqual(len(sim_rows(panel)), 4)
        self.assertNotIn(id(sim_rows(panel)[0]), [id(row) for row in before])

    def test_a_renamed_run_does_rebuild(self):
        # The name is in the row, so it is in the signature.
        session = _ListSession()
        panel = make_data_panel(session)
        panel.refresh()
        session.files[0].simulations[0].name = "takeoff"
        panel.refresh()
        self.assertEqual(sim_rows(panel)[0].text(0), "takeoff")

    def test_syncing_ticks_does_not_fire_the_handler(self):
        # Every setCheckState raises itemChanged exactly as a click does.
        # Unguarded, syncing would write the boxes straight back into the state
        # -- harmless while they agree, and a loop the moment they do not.
        panel = make_data_panel()
        panel.refresh()
        panel._state.set_selected(0, 1, True)
        fired = []
        panel.tree.itemChanged.connect(lambda *a: fired.append(a))
        panel.refresh()
        self.assertEqual(fired, [])

    def test_ticking_a_parameter_keeps_the_same_list_items(self):
        panel = make_param_panel()
        panel.refresh()
        before = [panel.y_list.item(i) for i in range(panel.y_list.count())]

        panel._state.tile.y_params = ["ALT"]
        panel.refresh()

        self.assertEqual(
            [id(panel.y_list.item(i)) for i in range(panel.y_list.count())],
            [id(item) for item in before])

    def test_the_parameter_tick_still_arrives(self):
        panel = make_param_panel()
        panel.refresh()
        panel._state.tile.y_params = ["MACH"]
        panel.refresh()

        states = {panel.y_list.item(i).data(Qt.ItemDataRole.UserRole):
                  panel.y_list.item(i).checkState()
                  for i in range(panel.y_list.count())}
        self.assertEqual(states["MACH"], Qt.CheckState.Checked)
        self.assertEqual(states["ALT"], Qt.CheckState.Unchecked)

    def test_switching_tile_moves_the_x_combo_without_a_rebuild(self):
        # The common case: clicking another tile refreshes this panel, and the
        # parameters are the same four hundred they were a moment ago.
        panel = make_param_panel()
        panel._state.tile.x_param = "TIME"
        panel.refresh()
        before = [panel.y_list.item(i) for i in range(panel.y_list.count())]

        panel._state.tile.x_param = "MACH"
        panel.refresh()

        self.assertEqual(panel.x_combo.currentData(), "MACH")
        self.assertEqual(
            [id(panel.y_list.item(i)) for i in range(panel.y_list.count())],
            [id(item) for item in before])

    def test_a_new_parameter_does_rebuild_the_list(self):
        session = _ListSession()
        panel = make_param_panel(session)
        panel.refresh()
        session.parameters.append("NZ")
        panel.refresh()
        self.assertEqual(panel.y_list.count(), 4)

    def test_a_changed_alias_does_rebuild_the_list(self):
        # The displayed label is what the row says, so renaming a channel has
        # to reach the row even though the parameter set is unchanged.
        session = _ListSession()
        panel = make_param_panel(session)
        panel.refresh()
        session.label = lambda name: "renamed " + name
        panel.refresh()
        self.assertTrue(panel.y_list.item(0).text().startswith("renamed"))


class TestFirstSwitchToIsoColouring(unittest.TestCase):
    """Choosing ISO must settle which ISO before the table is built from it.

    ISO mode with no ISO parameter named is not a state: _style_entries() and
    the tile's own colouring both test that name, and both fall back to
    colouring by simulation when it is empty. The default WAS committed -- but
    by apply_parameter_defaults(), which runs inside refresh_plots(), i.e.
    after style_changed has already reached the window. One step too late for
    the table: the plots came back coloured by ISO while CURVES still listed
    runs, under a dropdown already showing the parameter it had picked.
    """

    def setUp(self):
        self.panel = object.__new__(panels.StylePanel)
        self.panel._session = _ListSession()
        self.panel._state = AppState()
        self.panel._updating = False
        self.panel._table_signature = None
        self.panel._row_keys = []
        self.panel.table = qt_stubs.QTableWidget(0, 4)
        self.panel.refresh = lambda: None      # only the commit is under test
        self.panel.style_changed

    def test_choosing_iso_names_the_parameter(self):
        from aoeviewer.model import ColorMode

        self.panel._on_mode_changed(int(ColorMode.ISO))

        tile = self.panel._state.tile
        self.assertEqual(tile.color_mode, ColorMode.ISO)
        self.assertTrue(tile.color_iso_param)

    def test_it_prefers_a_parameter_already_set_up_as_an_iso(self):
        from aoeviewer.model import ColorMode

        self.panel._state.iso_state("MACH")
        self.panel._on_mode_changed(int(ColorMode.ISO))

        self.assertEqual(self.panel._state.tile.color_iso_param, "MACH")

    def test_the_entries_are_iso_entries_not_run_entries(self):
        # The symptom exactly: with the name still empty, _style_entries() fell
        # through to the simulation branch and CURVES listed runs.
        from aoeviewer.model import ColorMode

        self.panel._on_mode_changed(int(ColorMode.ISO))
        for _label, key in self.panel._style_entries():
            self.assertTrue(key.startswith("iso::"), key)

    def test_a_choice_already_made_is_left_alone(self):
        from aoeviewer.model import ColorMode

        self.panel._state.tile.color_iso_param = "ALT"
        self.panel._on_mode_changed(int(ColorMode.ISO))
        self.assertEqual(self.panel._state.tile.color_iso_param, "ALT")

    def test_the_dropdown_and_the_commit_agree(self):
        # Two readers of one rule. A panel that displayed one parameter while
        # the tile adopted another is the confusion this prevents.
        state = self.panel._state
        state.iso_state("MACH")
        params = self.panel._session.parameters
        self.assertEqual(state.default_iso_param(params), "MACH")

        state.apply_parameter_defaults(params)
        from aoeviewer.model import ColorMode
        state.tile.color_mode = ColorMode.ISO
        state.tile.color_iso_param = ""
        state.apply_parameter_defaults(params)
        self.assertEqual(state.tile.color_iso_param,
                         state.default_iso_param(params))

    def test_no_parameters_means_no_choice_to_make(self):
        from aoeviewer.model import ColorMode

        self.panel._session.parameters = []
        self.panel._on_mode_changed(int(ColorMode.ISO))
        self.assertEqual(self.panel._state.tile.color_iso_param, "")


class TestAnnotationMoveReachesEveryTile(unittest.TestCase):
    """A band is global, so a drag that reaches one tile is a wrong picture.

    _apply_annotation_move() refreshed the two annotation panels and then the
    grid. When AreaPanel.refresh() raised -- on the caption field removed from
    the model -- the grid refresh never ran: the band moved on the tile under
    the cursor, pyqtgraph having already dragged that item, and stayed put on
    every other tile. One exception, two bugs, and only one of them looked like
    an exception.
    """

    def test_the_plots_are_refreshed_before_the_lists(self):
        source = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "python", "aoeviewer", "ui", "main_window.py")
        with open(source, encoding="utf-8") as handle:
            body = handle.read()
        start = body.index("def _apply_annotation_move")
        block = body[start:body.index("\n    def ", start + 10)]
        self.assertLess(block.index("self.grid.refresh()"),
                        block.index("self.marker_panel"))

    def test_a_failing_panel_does_not_stop_the_others(self):
        class Boom:
            def refresh(self):
                raise RuntimeError("no")

        class Fine:
            def __init__(self):
                self.refreshed = 0

            def refresh(self):
                self.refreshed += 1

        fine = Fine()
        # The failure is logged, which is the point of it -- quieten the log
        # for the length of the test so the suite's output stays readable.
        import logging
        main_window._log.setLevel(logging.CRITICAL + 1)
        try:
            self.assertFalse(main_window.MainWindow._refresh_now(Boom()))
        finally:
            main_window._log.setLevel(logging.NOTSET)
        self.assertTrue(main_window.MainWindow._refresh_now(fine))
        self.assertEqual(fine.refreshed, 1)

    def test_a_panel_without_a_refresh_is_not_an_error(self):
        self.assertFalse(main_window.MainWindow._refresh_now(None))
        self.assertFalse(main_window.MainWindow._refresh_now(object()))


class _AnnotationSession:
    parameters = ["TIME", "ALPHA"]

    def label(self, name):
        return name


class TestAddingAnnotations(unittest.TestCase):
    """Adding a marker or a band must not raise.

    It did, and the failure was as bad as it gets: an unhandled exception in a
    Qt slot takes the process down, so the button closed the application. The
    cause was a leftover from removing the caption field -- AreaPanel._describe
    still read item.label to build the row text, and refresh() calls it for
    every band in the list, so the first Add killed it.
    """

    def setUp(self):
        self.state = AppState()
        self.session = _AnnotationSession()

    def test_adding_a_band_lists_it(self):
        panel = panels.AreaPanel(self.session, self.state)
        panel._add()                      # must not raise
        self.assertEqual(len(self.state.areas), 1)
        self.assertEqual(panel.list.count(), 1)

    def test_adding_a_marker_lists_it(self):
        panel = panels.MarkerPanel(self.session, self.state)
        panel._add()                      # must not raise
        self.assertEqual(len(self.state.markers), 1)
        self.assertEqual(panel.list.count(), 1)

    def test_a_band_is_described_by_its_span(self):
        # The span is what identifies a band now that there is no caption, so
        # the row has to carry the parameter and both ends of it.
        from aoeviewer.model import AreaBand

        panel = panels.AreaPanel(self.session, self.state)
        text = panel._describe(AreaBand(x_param="TIME", x_min=2.0, x_max=5.0))
        self.assertIn("TIME", text)
        self.assertIn("2", text)
        self.assertIn("5", text)

    def test_neither_panel_reads_a_caption_off_the_model(self):
        # The field is gone from both dataclasses; anything still reaching for
        # it raises AttributeError at exactly the wrong moment.
        from aoeviewer.model import AreaBand, Marker

        area_panel = panels.AreaPanel(self.session, self.state)
        marker_panel = panels.MarkerPanel(self.session, self.state)
        area_panel._describe(AreaBand())
        marker_panel._describe(Marker())

    def test_refreshing_a_full_list_is_safe(self):
        # refresh() calls _describe once per item, which is how one stale
        # attribute reached every band at once.
        panel = panels.AreaPanel(self.session, self.state)
        for _ in range(5):
            panel._add()
        panel.refresh()
        self.assertEqual(panel.list.count(), 5)


class TestLegendOrderIsIntrinsic(unittest.TestCase):
    """Legend rows must not reshuffle when a run is added or removed.

    Rows are emitted in the order the draw groups are built -- pyqtgraph fills
    the legend from addItem(), and an adopted item has its row put back in the
    same loop -- and that order used to be the order the groups were first
    *seen* while walking the selected runs. So which ISO group came first
    depended on which runs happened to be selected: tick one off and
    0-10, 10-15, 15-20 became 15-20, 0-10, 10-15, with the right colours on
    the right curves and the legend in an order nobody chose.

    The ISO values here are deliberately out of order -- run 0 carries 18, the
    top group -- so "walk order" and "belongs-here order" are different
    answers and the test can tell them apart.
    """

    GROUPS = [("group 0 - 10", 0.0, 10.0, True),
              ("group 10 - 15", 10.0, 15.0, True),
              ("group 15 - 20", 15.0, 20.0, True)]

    def setUp(self):
        from aoeviewer.model import ColorMode

        self.session = _DrawSession(iso_values=(18.0, 5.0, 12.0))
        self.tile = make_drawable_tile(self.session)
        state = self.tile._state
        state.selected_sims.update({(0, 0), (0, 1), (0, 2)})

        tile = state.tile
        tile.x_param, tile.y_params = "TIME", ["ALT"]
        tile.color_mode = ColorMode.ISO
        tile.color_iso_param = "MACH"
        tile.show_legend = True
        tile.legend_show_iso = True

    def group_iso(self):
        iso = self.tile._state.iso_state("MACH")
        iso.grouped = True
        iso.groups = list(self.GROUPS)

    def test_grouped_rows_follow_the_grouping_the_user_set_up(self):
        self.group_iso()
        self.tile.rebuild()
        self.assertEqual(
            legend_names(self.tile),
            ["ALT | MACH = group 0 - 10",
             "ALT | MACH = group 10 - 15",
             "ALT | MACH = group 15 - 20"])

    def test_removing_a_run_does_not_reshuffle_the_rest(self):
        # The reported symptom, exactly.
        self.group_iso()
        self.tile.rebuild()
        before = legend_names(self.tile)

        self.tile._state.selected_sims.discard((0, 1))   # the 0-10 run
        self.tile.rebuild()

        self.assertEqual(legend_names(self.tile),
                         [name for name in before if "0 - 10" not in name])

    def test_adding_a_run_back_restores_its_place(self):
        self.group_iso()
        self.tile._state.selected_sims.discard((0, 1))
        self.tile.rebuild()
        self.tile._state.selected_sims.add((0, 1))
        self.tile.rebuild()
        self.assertEqual(
            legend_names(self.tile),
            ["ALT | MACH = group 0 - 10",
             "ALT | MACH = group 10 - 15",
             "ALT | MACH = group 15 - 20"])

    def test_the_order_does_not_depend_on_which_run_is_walked_first(self):
        # Same three groups, the runs carrying them permuted. The legend must
        # come out identical, which is the whole claim.
        self.group_iso()
        self.tile.rebuild()
        expected = legend_names(self.tile)

        for values in ((5.0, 12.0, 18.0), (12.0, 18.0, 5.0),
                       (18.0, 12.0, 5.0)):
            self.session._iso = list(values)
            self.tile.rebuild()
            self.assertEqual(legend_names(self.tile), expected, values)

    def test_ungrouped_values_are_ordered_by_value(self):
        # No grouping: the rows are the values themselves, and ascending is the
        # only order anyone expects of a list of numbers.
        self.tile.rebuild()
        self.assertEqual(
            legend_names(self.tile),
            ["ALT | MACH = 5", "ALT | MACH = 12", "ALT | MACH = 18"])

    def test_a_run_in_no_group_goes_last_rather_than_anywhere(self):
        # It is still plotted -- the grouping decides colour, not whether a run
        # is drawn -- but it has no ISO label, so its row cannot be placed
        # among the groups. Last is the one position that says so, and it is
        # where an unrankable row belongs however the runs are walked.
        self.group_iso()
        self.session._iso = [18.0, 5.0, 99.0]     # 99 is in no group
        self.tile.rebuild()
        self.assertEqual(legend_names(self.tile),
                         ["ALT | MACH = group 0 - 10",
                          "ALT | MACH = group 15 - 20",
                          "ALT"])

    def test_parameter_mode_follows_the_tile_s_own_parameter_order(self):
        from aoeviewer.model import ColorMode

        tile = self.tile._state.tile
        tile.color_mode = ColorMode.PARAMETER
        tile.legend_show_iso = False
        tile.y_params = ["MACH", "ALT"]
        self.tile.rebuild()
        self.assertEqual(legend_names(self.tile), ["MACH", "ALT"])

        tile.y_params = ["ALT", "MACH"]
        self.tile.rebuild()
        self.assertEqual(legend_names(self.tile), ["ALT", "MACH"])

    def test_simulation_mode_follows_file_then_run(self):
        from aoeviewer.model import ColorMode

        tile = self.tile._state.tile
        tile.color_mode = ColorMode.SIMULATION
        tile.legend_show_iso = False
        tile.legend_show_sim = True
        self.tile.rebuild()
        self.assertEqual(legend_names(self.tile),
                         ["ALT | R0", "ALT | R1", "ALT | R2"])


class TestTheTwoConstraintsAreIndependent(unittest.TestCase):
    """The DATA tab and the ISO tab narrow the same picture from two sides.

    They used to share one piece of state. Applying an ISO filter walked every
    run in every file calling set_selected(), so switching a filter on rewrote
    the DATA tab: ask for a mass between 50000 and 100000 and every run in that
    band joined the plot whether or not you had chosen it, and every run
    outside it was unticked whether or not you had. Two controls, one variable,
    each overwriting the other's answer.

    Now each records its own answer and a run is drawn only if both admit it.
    """

    MASS = [40000.0, 75000.0, 120000.0]

    def setUp(self):
        self.state = AppState()
        self.files = [(0, 3)]

    def values_for(self, _file_index, _param):
        return np.array(self.MASS, dtype=np.float32)

    def band(self, low_on=False, mid_on=True, high_on=False):
        """A MASS grouping with only the chosen bands active."""
        iso = self.state.iso_state("MASS")
        iso.active = True
        iso.grouped = True
        iso.groups = [("low", 0.0, 50000.0, low_on),
                      ("mid", 50000.0, 100000.0, mid_on),
                      ("high", 100000.0, 200000.0, high_on)]
        self.state.resolve_iso_filters(self.files, self.values_for)
        return iso

    def test_a_filter_narrows_the_data_choice_rather_than_replacing_it(self):
        self.state.set_selected(0, 0, True)      # 40000, outside the band
        self.state.set_selected(0, 1, True)      # 75000, inside it
        self.band()
        self.assertEqual(self.state.plotted_sims(), ((0, 1),))

    def test_a_filter_does_not_touch_the_data_ticks(self):
        self.state.set_selected(0, 0, True)
        self.state.set_selected(0, 1, True)
        self.band()
        self.assertEqual(sorted(self.state.selected_sims), [(0, 0), (0, 1)])

    def test_a_filter_never_adds_a_run_the_user_did_not_choose(self):
        # Run 1 is in the band and untouched in DATA. The band is a filter, not
        # a selection: it can only take runs away.
        self.band()
        self.assertEqual(self.state.plotted_sims(), ())
        self.assertEqual(self.state.selected_sims, set())

    def test_widening_a_filter_still_adds_nothing(self):
        self.band(low_on=True, high_on=True)
        self.assertEqual(self.state.plotted_sims(), ())

    def test_ticking_a_filtered_out_run_puts_it_on_no_tile(self):
        # Stated in the report: a run whose ISO value is not selected must not
        # appear, however firmly it is ticked.
        self.band()
        self.state.set_selected(0, 2, True)      # 120000, above the band
        self.assertFalse(self.state.is_plotted(0, 2))
        self.assertEqual(self.state.plotted_sims(), ())

    def test_ticking_a_run_does_not_touch_the_filters(self):
        iso = self.band()
        before = (iso.active, iso.grouped, list(iso.groups),
                  dict(iso.value_enabled))
        self.state.set_selected(0, 2, True)
        self.assertEqual((iso.active, iso.grouped, list(iso.groups),
                          dict(iso.value_enabled)), before)

    def test_both_halves_are_required(self):
        self.band()
        for pair, expected in (((0, 0), False),    # ticked later, outside
                               ((0, 1), False),    # inside, not ticked
                               ((0, 2), False)):
            self.assertEqual(self.state.is_plotted(*pair), expected, pair)

        self.state.set_selected(0, 1, True)
        self.assertTrue(self.state.is_plotted(0, 1))

    def test_it_says_how_many_runs_are_held_back(self):
        # The number that explains a plot shorter than the DATA tab suggests.
        for sim in range(3):
            self.state.set_selected(0, sim, True)
        self.band()
        self.assertEqual(self.state.excluded_by_iso(), 2)

    def test_nothing_is_held_back_when_no_filter_is_active(self):
        self.state.set_selected(0, 0, True)
        self.state.resolve_iso_filters(self.files, self.values_for)
        self.assertEqual(self.state.excluded_by_iso(), 0)
        self.assertFalse(self.state.iso_filters_resolved())

    def test_no_active_filter_costs_nothing_to_represent(self):
        # None rather than a set of every run: "no filter" is then free to
        # represent and free to keep up to date when a file arrives.
        self.state.resolve_iso_filters(self.files, self.values_for)
        self.assertIsNone(self.state._iso_admitted)
        self.assertTrue(self.state.iso_admitted(0, 99))

    def test_an_ungrouped_filter_selects_by_value(self):
        iso = self.state.iso_state("MASS")
        iso.active = True
        iso.value_enabled = {40000.0: False, 75000.0: True, 120000.0: False}
        for sim in range(3):
            self.state.set_selected(0, sim, True)
        self.state.resolve_iso_filters(self.files, self.values_for)
        self.assertEqual(self.state.plotted_sims(), ((0, 1),))

    def test_a_value_never_ticked_is_shown(self):
        # Absent means on: a value the user has not touched is not a value they
        # excluded.
        iso = self.state.iso_state("MASS")
        iso.active = True
        for sim in range(3):
            self.state.set_selected(0, sim, True)
        self.state.resolve_iso_filters(self.files, self.values_for)
        self.assertEqual(len(self.state.plotted_sims()), 3)

    def test_filters_combine_with_and(self):
        # Adding a second filter must narrow, not widen.
        mass = self.state.iso_state("MASS")
        mass.active = True
        mass.value_enabled = {40000.0: True, 75000.0: True, 120000.0: False}
        alt = self.state.iso_state("ALT")
        alt.active = True
        alt.value_enabled = {0.0: False, 1.0: True, 2.0: True}

        def values_for(_file_index, param):
            if param == "MASS":
                return np.array(self.MASS, dtype=np.float32)
            return np.array([0.0, 1.0, 2.0], dtype=np.float32)

        for sim in range(3):
            self.state.set_selected(0, sim, True)
        self.state.resolve_iso_filters(self.files, values_for)
        # MASS admits 0 and 1; ALT admits 1 and 2; both admit only 1.
        self.assertEqual(self.state.plotted_sims(), ((0, 1),))

    def test_an_inactive_filter_does_not_narrow(self):
        iso = self.state.iso_state("MASS")
        iso.active = False
        iso.value_enabled = {40000.0: False, 75000.0: False, 120000.0: False}
        self.state.set_selected(0, 0, True)
        self.state.resolve_iso_filters(self.files, self.values_for)
        self.assertEqual(self.state.plotted_sims(), ((0, 0),))

    def test_a_run_with_no_value_for_the_parameter_is_excluded(self):
        self.MASS = [40000.0, float("nan"), 120000.0]
        for sim in range(3):
            self.state.set_selected(0, sim, True)
        iso = self.state.iso_state("MASS")
        iso.active = True
        self.state.resolve_iso_filters(self.files, self.values_for)
        self.assertFalse(self.state.is_plotted(0, 1))

    def test_a_short_iso_column_does_not_exclude_the_runs_it_does_not_cover(self):
        """A filter says nothing about a run it has no value for.

        This asserted the opposite until ISO filters became part of a saved
        workspace. Excluding the uncovered runs is defensible read on its own
        and wrong in every case it actually arises: a filter on a parameter only
        some files carry hid every run of the others, and a workspace whose ISO
        parameter had since disappeared opened to an empty plot with nothing to
        explain it. A NaN *in* the column is still an exclusion -- the column is
        there and the run has no value -- which the test above pins down.
        """
        self.MASS = [75000.0]                   # one value, three runs
        for sim in range(3):
            self.state.set_selected(0, sim, True)
        iso = self.state.iso_state("MASS")
        iso.active = True
        self.state.resolve_iso_filters(self.files, self.values_for)
        self.assertEqual(self.state.plotted_sims(), ((0, 0), (0, 1), (0, 2)))

    def test_a_filter_on_an_absent_parameter_filters_nothing(self):
        """The workspace-restore case: the parameter is gone entirely."""
        def values_for(_file_index, param):
            if param == "MASS":
                return np.array(self.MASS, dtype=np.float32)
            return np.array([], dtype=np.float32)

        for sim in range(3):
            self.state.set_selected(0, sim, True)
        iso = self.state.iso_state("NOT_IN_ANY_FILE")
        iso.active = True
        iso.value_enabled = {1.0: False}        # would exclude, if it applied
        self.state.resolve_iso_filters(self.files, values_for)
        self.assertEqual(self.state.plotted_sims(), ((0, 0), (0, 1), (0, 2)))
        self.assertEqual(self.state.iso_inapplicable_params(),
                         {"NOT_IN_ANY_FILE"})
        self.assertEqual(self.state.excluded_by_iso(), 0)

    def test_a_filter_no_run_satisfies_is_reported_as_empty(self):
        """Applicable, and admits nothing. Legitimate, but worth saying."""
        for sim in range(3):
            self.state.set_selected(0, sim, True)
        iso = self.state.iso_state("MASS")
        iso.active = True
        iso.grouped = True
        iso.groups = [("900000 - 1000000", 900000.0, 1000000.0, True)]
        self.state.resolve_iso_filters(self.files, self.values_for)
        self.assertEqual(self.state.plotted_sims(), ())
        self.assertEqual(self.state.iso_empty_params(), {"MASS"})
        self.assertFalse(self.state.iso_inapplicable_params())


class TestPlottedSimsCache(unittest.TestCase):
    """Every tile asks this on every rebuild; it must not recompute per tile."""

    def setUp(self):
        self.state = AppState()
        for sim in range(50):
            self.state.set_selected(0, sim, True)

    def test_repeated_asks_are_the_same_object(self):
        first = self.state.plotted_sims()
        for _ in range(25):
            self.assertIs(self.state.plotted_sims(), first)

    def test_ticking_a_run_invalidates_it(self):
        before = self.state.plotted_sims()
        self.state.set_selected(0, 99, True)
        self.assertIsNot(self.state.plotted_sims(), before)
        self.assertIn((0, 99), self.state.plotted_sims())

    def test_clearing_the_selection_invalidates_it(self):
        self.state.plotted_sims()
        self.state.clear_selection()
        self.assertEqual(self.state.plotted_sims(), ())

    def test_resolving_the_filters_invalidates_it(self):
        self.state.plotted_sims()
        iso = self.state.iso_state("MASS")
        iso.active = True
        iso.value_enabled = {float(v): v < 10 for v in range(50)}
        self.state.resolve_iso_filters(
            [(0, 50)], lambda _f, _p: np.arange(50, dtype=np.float32))
        self.assertEqual(len(self.state.plotted_sims()), 10)

    def test_writing_the_set_directly_is_still_noticed(self):
        # Tests and older code reach for selected_sims itself. Going through
        # set_selected() is what invalidates the cache properly; this is the
        # cheap backstop that catches the rest.
        self.state.plotted_sims()
        self.state.selected_sims.add((1, 0))
        self.assertIn((1, 0), self.state.plotted_sims())

    def test_dropping_a_file_moves_the_iso_answer_with_it(self):
        # The admitted set is indexed by file too. Left unshifted it would
        # start admitting the wrong runs of the wrong files.
        state = AppState()
        state.set_selected(0, 0, True)
        state.set_selected(1, 0, True)
        state.set_selected(1, 1, True)
        iso = state.iso_state("MASS")
        iso.active = True
        iso.value_enabled = {0.0: True, 1.0: False}
        state.resolve_iso_filters(
            [(0, 2), (1, 2)], lambda _f, _p: np.array([0.0, 1.0], "f4"))
        self.assertEqual(state.plotted_sims(), ((0, 0), (1, 0)))

        state.drop_file(0)
        self.assertEqual(state.plotted_sims(), ((0, 0),))
        self.assertTrue(state.iso_admitted(0, 0))
        self.assertFalse(state.iso_admitted(0, 1))


class TestIsoAdmits(unittest.TestCase):
    """One filter's verdict on one value, which is all the rest is built on."""

    def setUp(self):
        self.state = AppState()
        self.iso = self.state.iso_state("MASS")

    def test_a_grouped_value_needs_its_group_active(self):
        self.iso.grouped = True
        self.iso.groups = [("low", 0.0, 10.0, True), ("high", 10.0, 20.0, False)]
        self.assertTrue(self.state.iso_admits(self.iso, 5.0))
        self.assertFalse(self.state.iso_admits(self.iso, 15.0))

    def test_a_shared_edge_belongs_to_the_earlier_group(self):
        # Inclusive at both ends, first claim wins -- the rule isogroups owns
        # and the core applies, so the plot and the group counts agree.
        self.iso.grouped = True
        self.iso.groups = [("low", 0.0, 10.0, True), ("high", 10.0, 20.0, False)]
        self.assertTrue(self.state.iso_admits(self.iso, 10.0))

    def test_a_value_in_no_group_is_excluded(self):
        self.iso.grouped = True
        self.iso.groups = [("low", 0.0, 10.0, True)]
        self.assertFalse(self.state.iso_admits(self.iso, 99.0))

    def test_nan_is_excluded(self):
        self.assertFalse(self.state.iso_admits(self.iso, float("nan")))

    def test_grouping_switched_on_with_no_intervals_admits_by_value(self):
        # A grouped parameter with no groups matches nothing, so the value
        # branch is the safe reading of it rather than excluding everything.
        self.iso.grouped = True
        self.iso.groups = []
        self.assertTrue(self.state.iso_admits(self.iso, 5.0))


class TestFilteredRunsAreNotDrawn(unittest.TestCase):
    """The end of the chain: a held-back run puts no curve on a tile."""

    def setUp(self):
        from aoeviewer.model import ColorMode

        self.session = _DrawSession(iso_values=(40000.0, 75000.0, 120000.0))
        self.tile = make_drawable_tile(self.session)
        state = self.tile._state
        for sim in range(3):
            state.set_selected(0, sim, True)

        tile = state.tile
        tile.x_param, tile.y_params = "TIME", ["ALT"]
        tile.color_mode = ColorMode.SIMULATION
        tile.show_legend = True
        tile.legend_show_sim = True

    def band(self):
        state = self.tile._state
        iso = state.iso_state("MASS")
        iso.active = True
        iso.grouped = True
        iso.groups = [("low", 0.0, 50000.0, False),
                      ("mid", 50000.0, 100000.0, True),
                      ("high", 100000.0, 200000.0, False)]
        state.resolve_iso_filters([(0, 3)], self.session.iso_values)

    def test_all_three_runs_draw_without_a_filter(self):
        self.tile.rebuild()
        self.assertEqual(legend_names(self.tile),
                         ["ALT | R0", "ALT | R1", "ALT | R2"])

    def test_only_the_admitted_run_draws_with_one(self):
        self.band()
        self.tile.rebuild()
        self.assertEqual(legend_names(self.tile), ["ALT | R1"])

    def test_the_tile_reads_as_empty_when_the_filter_excludes_everything(self):
        state = self.tile._state
        iso = state.iso_state("MASS")
        iso.active = True
        iso.grouped = True
        iso.groups = [("none", 0.0, 1.0, True)]
        state.resolve_iso_filters([(0, 3)], self.session.iso_values)

        self.tile.rebuild()
        self.assertEqual(legend_names(self.tile), [])
        self.assertEqual(self.tile._curves, {})

    def test_the_preload_asks_only_for_runs_that_will_be_drawn(self):
        # Reading columns for runs nothing is going to draw is the cost of
        # keying the preload on the wrong one of the two constraints.
        self.band()
        grid = object.__new__(pg_grid.PlotGrid)
        grid._state = self.tile._state
        grid._tiles = [self.tile]
        self.assertEqual(grid.plotted_sims_by_file(), {0: [1]})


class TestTheIsoPanelDoesNotSelectRuns(unittest.TestCase):
    """apply_filter() declares a filter. It does not choose runs."""

    def test_it_leaves_the_data_selection_alone(self):
        panel = make_iso_panel()
        panel.apply_filter = panels.IsoPanel.apply_filter.__get__(panel)
        panel._state.set_selected(0, 0, True)
        iso = panel._state.iso_state("MACH")
        iso.active = True

        panel.apply_filter()

        self.assertEqual(sorted(panel._state.selected_sims), [(0, 0)])

    def test_it_says_that_the_filters_changed(self):
        panel = make_iso_panel()
        panel.apply_filter = panels.IsoPanel.apply_filter.__get__(panel)
        fired = []
        panel.filter_changed.connect(lambda: fired.append(1))
        panel.apply_filter()
        self.assertEqual(len(fired), 1)


class TestHeldBackRunsAreMarked(unittest.TestCase):
    """A ticked run that is not on screen has to say why.

    Two independent constraints is the right model and it has one cost: a run
    can be firmly ticked in DATA and still be on no tile. Unmarked, that is
    indistinguishable from a bug -- the checkbox claims the run is plotted and
    the plot disagrees.
    """

    def setUp(self):
        self.session = _ListSession(sims=("R0", "R1", "R2"))
        self.panel = make_data_panel(self.session)
        state = self.panel._state
        for sim in range(3):
            state.set_selected(0, sim, True)
        self.panel.refresh()

    def held(self):
        return [bool(row.data(0, panels._HELD_BACK_ROLE))
                for row in sim_rows(self.panel)]

    def filter_to(self, admitted):
        state = self.panel._state
        iso = state.iso_state("MASS")
        iso.active = True
        iso.value_enabled = {float(v): v in admitted for v in range(3)}
        state.resolve_iso_filters(
            [(0, 3)], lambda _f, _p: np.arange(3, dtype=np.float32))

    def test_nothing_is_marked_without_a_filter(self):
        self.assertEqual(self.held(), [False, False, False])

    def test_the_excluded_rows_are_marked(self):
        self.filter_to({1})
        self.panel.refresh()
        self.assertEqual(self.held(), [True, False, True])

    def test_the_mark_comes_off_when_the_filter_widens(self):
        self.filter_to({1})
        self.panel.refresh()
        self.filter_to({0, 1, 2})
        self.panel.refresh()
        self.assertEqual(self.held(), [False, False, False])

    def test_an_unticked_row_is_not_marked(self):
        # It is not held back, it was simply not chosen. Marking it would say
        # the filter is responsible for something the user did.
        self.filter_to({1})
        self.panel._state.set_selected(0, 0, False)
        self.panel.refresh()
        self.assertEqual(self.held(), [False, False, True])

    def test_the_reason_is_on_hover(self):
        self.filter_to({1})
        self.panel.refresh()
        rows = sim_rows(self.panel)
        self.assertIn("held back by an ISO filter", rows[0].toolTip(0))
        self.assertNotIn("held back", rows[1].toolTip(0))

    def test_the_original_tooltip_comes_back(self):
        rows = sim_rows(self.panel)
        before = rows[0].toolTip(0)
        self.filter_to({1})
        self.panel.refresh()
        self.filter_to({0, 1, 2})
        self.panel.refresh()
        self.assertEqual(sim_rows(self.panel)[0].toolTip(0), before)

    def test_the_ticks_themselves_are_untouched(self):
        # The filter dims rows. It does not untick them -- that is the DATA
        # tab's state and the ISO tab does not own it.
        self.filter_to({1})
        self.panel.refresh()
        for row in sim_rows(self.panel):
            self.assertEqual(row.checkState(0), Qt.CheckState.Checked)

    def test_marking_writes_nothing_when_nothing_changed(self):
        self.filter_to({1})
        self.panel.refresh()
        rows = sim_rows(self.panel)
        before = [row.data(0, panels._HELD_BACK_ROLE) for row in rows]
        tips = [row.toolTip(0) for row in rows]

        for _ in range(5):
            self.panel.refresh()

        self.assertEqual([row.data(0, panels._HELD_BACK_ROLE) for row in rows],
                         before)
        self.assertEqual([row.toolTip(0) for row in rows], tips)


class TestIsoLabelRanks(unittest.TestCase):
    """The ranks that order the legend, read straight off the lookup.

    Computed in the same pass as the labels because both come out of the same
    per-run ISO read, and that read is the expensive part.
    """

    def setUp(self):
        self.tile = make_drawable_tile()
        self.tile._state.selected_sims.update({(0, 0), (0, 1), (0, 2)})

    def test_grouped_labels_rank_by_their_place_in_the_grouping(self):
        iso = self.tile._state.iso_state("MACH")
        iso.grouped = True
        iso.groups = [("low", 0.0, 10.0, True),
                      ("mid", 10.0, 15.0, True),
                      ("high", 15.0, 20.0, True)]

        _labels, ranks = self.tile._iso_labels("MACH")
        self.assertEqual(ranks, {"low": 0.0, "mid": 1.0, "high": 2.0})

    def test_ungrouped_labels_rank_by_value(self):
        _labels, ranks = self.tile._iso_labels("MACH")
        self.assertEqual(ranks, {"18": 18.0, "5": 5.0, "12": 12.0})

    def test_two_runs_sharing_a_label_rank_by_the_lower_one(self):
        # A row's place must not depend on which of its runs was visited
        # first, and ungrouped, two distinct values can format to one label.
        self.tile._session = _DrawSession(iso_values=(0.752, 0.751))
        self.tile._session.format_value = lambda value: "{0:.2f}".format(value)
        self.tile._state.selected_sims.clear()
        self.tile._state.selected_sims.update({(0, 0), (0, 1)})

        _labels, ranks = self.tile._iso_labels("MACH")
        self.assertEqual(list(ranks), ["0.75"])
        self.assertAlmostEqual(ranks["0.75"], 0.751, places=6)

    def test_no_iso_parameter_is_two_empty_maps(self):
        labels, ranks = self.tile._iso_labels("")
        self.assertEqual((labels, ranks), ({}, {}))


if __name__ == "__main__":
    unittest.main()


class TestWorkspaceKeepsTheDefinitions(unittest.TestCase):
    """Aliases, formulas and ISO filters survive a save and load.

    All three were absent from the workspace file. Formulas were written but
    aliases were passed as an empty mapping by the writer, and ISO filters were
    not written at all -- so a study whose whole point was three mass bands and
    a derived channel came back as a bare layout. These tests pin down the file
    format, and the next class pins down what happens when the data on the far
    side of the load no longer matches.
    """

    def _state_with_filters(self):
        state = AppState()
        state.set_selected(0, 0, True)
        state.set_selected(0, 2, True)

        ungrouped = state.iso_state("MACH")
        ungrouped.active = True
        ungrouped.value_enabled = {0.75: True, 0.8: False, 0.85: True}

        grouped = state.iso_state("MASS")
        grouped.active = True
        grouped.grouped = True
        grouped.group_count = 3
        grouped.groups = [
            ("light", 40000.0, 60000.0, True),
            ("medium", 60000.0, 80000.0, False),
            ("heavy", 80000.0, 100000.0, True),
        ]
        return state

    def test_iso_filters_round_trip(self):
        saved = self._state_with_filters().to_dict()
        loaded = AppState()
        loaded.load_dict(saved)

        self.assertEqual(sorted(loaded.iso_states), ["MACH", "MASS"])

        mach = loaded.iso_states["MACH"]
        self.assertTrue(mach.active)
        self.assertFalse(mach.grouped)
        # Float keys, not the strings a JSON object would have handed back.
        self.assertEqual(mach.value_enabled,
                         {0.75: True, 0.8: False, 0.85: True})
        self.assertTrue(all(isinstance(v, float) for v in mach.value_enabled))

        mass = loaded.iso_states["MASS"]
        self.assertTrue(mass.grouped)
        self.assertEqual(mass.group_count, 3)
        self.assertEqual(mass.groups, [
            ("light", 40000.0, 60000.0, True),
            ("medium", 60000.0, 80000.0, False),
            ("heavy", 80000.0, 100000.0, True),
        ])

    def test_it_survives_a_trip_through_real_json(self):
        """The keys must be portable, not just Python-round-trippable."""
        text = json.dumps(self._state_with_filters().to_dict())
        loaded = AppState()
        loaded.load_dict(json.loads(text))
        self.assertEqual(loaded.iso_states["MACH"].value_enabled,
                         {0.75: True, 0.8: False, 0.85: True})
        self.assertEqual(len(loaded.iso_states["MASS"].groups), 3)

    def test_the_derived_answer_is_not_saved(self):
        """What the filters *admit* describes the old files, so it is dropped."""
        state = self._state_with_filters()
        state.resolve_iso_filters(
            [(0, 3)], lambda _f, _p: np.array([50000.0, 70000.0, 90000.0],
                                              dtype=np.float32))
        self.assertTrue(state.iso_filters_resolved())
        loaded = AppState()
        loaded.load_dict(state.to_dict())
        self.assertFalse(loaded.iso_filters_resolved())

    def test_a_workspace_without_iso_states_still_loads(self):
        """Files written before filters were saved must open unchanged."""
        old = self._state_with_filters().to_dict()
        del old["iso_states"]
        loaded = AppState()
        loaded.load_dict(old)
        self.assertEqual(loaded.iso_states, {})
        self.assertEqual(sorted(loaded.selected_sims), [(0, 0), (0, 2)])

    def test_save_workspace_writes_formulas_and_aliases(self):
        import tempfile
        state = self._state_with_filters()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "w.aoews")
            model.save_workspace(
                path, state, ["/data/a.aoe"],
                {"CLIMB": "deriv(ALTITUDE, TIME)"},
                {"Angle of attack": "ALPHA"},
            )
            data = model.load_workspace(path)
        self.assertEqual(data["computed"], {"CLIMB": "deriv(ALTITUDE, TIME)"})
        self.assertEqual(data["aliases"], {"Angle of attack": "ALPHA"})
        self.assertEqual(data["files"], ["/data/a.aoe"])
        self.assertEqual(len(data["iso_states"]), 2)


class TestWorkspaceLoadsAgainstDifferentData(unittest.TestCase):
    """The load must survive data that no longer matches the workspace.

    A workspace outlives the run it was made from: the solver is re-run with
    fewer channels, a file is replaced, a parameter is renamed. None of that may
    raise, and none of it may silently empty the plot.
    """

    def test_a_malformed_iso_entry_is_skipped_not_fatal(self):
        loaded = AppState()
        loaded.load_dict({"iso_states": [
            {"param": "GOOD", "active": True},
            {"param": ""},                       # no name
            {"active": True},                    # no name at all
            "not a dict",
            None,
            {"param": "ALSO_GOOD", "group_count": "not a number"},
        ]})
        self.assertEqual(sorted(loaded.iso_states), ["ALSO_GOOD", "GOOD"])
        self.assertEqual(loaded.iso_states["ALSO_GOOD"].group_count, 3)

    def test_junk_values_and_groups_are_dropped_individually(self):
        loaded = AppState()
        loaded.load_dict({"iso_states": [{
            "param": "MASS",
            "value_enabled": [[1.0, True], ["oops", True], [2.0, False], "x"],
            "groups": [["ok", 10.0, 20.0, True],
                       ["bad", "low", 20.0, True],
                       ["short", 1.0],
                       ["reversed", 90.0, 80.0, True]],
        }]})
        iso = loaded.iso_states["MASS"]
        self.assertEqual(iso.value_enabled, {1.0: True, 2.0: False})
        self.assertEqual([g[0] for g in iso.groups], ["ok", "reversed"])
        # A reversed interval is normalised rather than refused.
        self.assertEqual(iso.groups[1][1:3], (80.0, 90.0))

    def test_non_finite_values_do_not_reach_the_file(self):
        state = AppState()
        iso = state.iso_state("MASS")
        iso.value_enabled = {1.0: True, float("nan"): True, float("inf"): False}
        written = state.to_dict()["iso_states"][0]["value_enabled"]
        self.assertEqual(written, [[1.0, True]])
        # And the file is real JSON, which NaN would not be.
        self.assertNotIn("NaN", json.dumps(state.to_dict()))

    def test_a_grouped_filter_that_lost_its_intervals_is_demoted(self):
        loaded = AppState()
        loaded.load_dict({"iso_states": [
            {"param": "MASS", "grouped": True, "groups": []},
        ]})
        iso = loaded.iso_states["MASS"]
        self.assertFalse(iso.grouped)

    def test_a_restored_filter_on_a_vanished_parameter_plots_everything(self):
        """The headline case. Nothing raises, and nothing disappears."""
        loaded = AppState()
        loaded.load_dict({
            "selected_sims": [[0, 0], [0, 1], [0, 2]],
            "iso_states": [{"param": "GONE", "active": True,
                            "grouped": True,
                            "groups": [["band", 1.0, 2.0, True]]}],
        })
        loaded.resolve_iso_filters(
            [(0, 3)], lambda _f, _p: np.array([], dtype=np.float32))
        self.assertEqual(loaded.plotted_sims(), ((0, 0), (0, 1), (0, 2)))
        self.assertEqual(loaded.iso_inapplicable_params(), {"GONE"})

    def test_a_restored_grouping_no_value_satisfies_says_so(self):
        loaded = AppState()
        loaded.load_dict({
            "selected_sims": [[0, 0], [0, 1]],
            "iso_states": [{"param": "MASS", "active": True, "grouped": True,
                            "groups": [["band", 1.0, 2.0, True]]}],
        })
        loaded.resolve_iso_filters(
            [(0, 2)], lambda _f, _p: np.array([50000.0, 70000.0],
                                              dtype=np.float32))
        self.assertEqual(loaded.plotted_sims(), ())
        self.assertEqual(loaded.iso_empty_params(), {"MASS"})
        self.assertEqual(loaded.excluded_by_iso(), 2)

    def test_a_mixed_study_keeps_the_files_that_lack_the_parameter(self):
        """File 0 carries MASS, file 1 does not. File 1 is not punished."""
        loaded = AppState()
        for sim in range(2):
            loaded.set_selected(0, sim, True)
            loaded.set_selected(1, sim, True)
        iso = loaded.iso_state("MASS")
        iso.active = True
        iso.grouped = True
        iso.groups = [("heavy", 60000.0, 100000.0, True)]

        def values_for(file_index, _param):
            if file_index == 0:
                return np.array([50000.0, 70000.0], dtype=np.float32)
            return np.array([], dtype=np.float32)

        loaded.resolve_iso_filters([(0, 2), (1, 2)], values_for)
        # File 0: only the 70000 run passes. File 1: both, unfiltered.
        self.assertEqual(loaded.plotted_sims(), ((0, 1), (1, 0), (1, 1)))


class TestPaletteOrder(unittest.TestCase):
    """The first three colours are fixed, in every colouring mode."""

    def test_the_first_three_are_red_blue_dark_green(self):
        self.assertEqual(model.PALETTE[:3], ("#e8232a", "#1f6fe0", "#1a7a3c"))

    def test_no_cyan_among_the_leading_colours(self):
        """Cyan and the lighter green are the same colour at one pixel wide."""
        for colour in model.PALETTE[:6]:
            red = int(colour[1:3], 16)
            green = int(colour[3:5], 16)
            blue = int(colour[5:7], 16)
            cyanish = green > 150 and blue > 150 and red < 120
            self.assertFalse(cyanish, "{0} reads as cyan".format(colour))

    def test_every_mode_allocates_the_same_first_three(self):
        """Allocation is by order of first use, so the keys must not matter."""
        expected = list(model.PALETTE[:3])
        for keys in (
            ["sim::a", "sim::b", "sim::c"],
            ["param::ALPHA", "param::MACH", "param::NZ"],
            ["iso::MASS::low", "iso::MASS::mid", "iso::MASS::high"],
        ):
            registry = StyleRegistry()
            self.assertEqual([registry.color_for(k) for k in keys], expected)

    def test_the_palette_has_no_duplicates(self):
        self.assertEqual(len(set(model.PALETTE)), len(model.PALETTE))

    def test_the_inspect_dot_is_not_a_palette_colour(self):
        """It marks a sample; it must not read as one more curve."""
        from aoeviewer.ui import plot_tile
        self.assertNotIn(plot_tile.INSPECT_DOT_COLOUR.lower(),
                         [c.lower() for c in model.PALETTE])


class _FakeAction:
    def __init__(self):
        self.text = ""
        self.tip = ""

    def setText(self, text):
        self.text = text

    def setToolTip(self, tip):
        self.tip = tip


class _FakeSession:
    """Just enough of a session to answer what the definitions are."""

    def __init__(self, computed=None, aliases=None, plain=()):
        self._computed = dict(computed or {})
        self._aliases = dict(aliases or {})
        self._plain = list(plain)

    @property
    def parameters(self):
        return sorted(set(self._plain) | set(self._computed) | set(self._aliases))

    def is_computed(self, name):
        return name in self._computed

    def computed_source(self, name):
        return self._computed[name]

    def resolve(self, name):
        return self._aliases.get(name, name)

    @property
    def files(self):
        return []


class _WindowStub:
    """A stand-in for MainWindow carrying only what these methods touch.

    The Qt stubs do not run a real main window, and building one would test the
    widget toolkit rather than the behaviour. The methods under test are bound
    manually instead, which is the same trick the redraw-guard test above uses.
    """

    def __init__(self, session=None, path=None, dirty=False):
        self.state = AppState()
        self.state.workspace_path = path
        self.state.dirty = dirty
        self.session = session or _FakeSession()
        self.save_action = _FakeAction()
        self.save_as_action = _FakeAction()
        self.title = ""
        self.status = ""
        self.asked = []
        self.written = []

    # -- the bits of MainWindow the methods call ------------------------
    def setWindowTitle(self, text):
        self.title = text

    @property
    def status_label(self):
        holder = self

        class _Label:
            def setText(self, text):
                holder.status = text
        return _Label()

    # -- the methods under test, bound to this stub ---------------------
    def call(self, name, *args):
        return getattr(self, name)(*args)


# Borrow the real implementations. Copying them onto the stub rather than
# calling them unbound means one method may call another, which several of them
# do -- _update_window_title() asks _workspace_label(), and both Save paths go
# through _write_workspace().
for _name in ("_workspace_label", "_update_window_title",
              "_update_workspace_actions", "_ask_workspace_path",
              "_collect_definitions", "_write_workspace",
              "save_workspace", "save_workspace_as"):
    setattr(_WindowStub, _name, getattr(main_window.MainWindow, _name))


class TestSaveAsAndTheCurrentFile(unittest.TestCase):
    """Save, Save As, and making it obvious which file is being written."""

    def test_the_title_names_the_workspace(self):
        window = _WindowStub(path=os.path.join("study", "sweep.aoews"))
        window.call("_update_window_title")
        self.assertIn("sweep.aoews", window.title)
        self.assertIn("aoeViewer", window.title)
        self.assertFalse(window.title.startswith("*"))

    def test_an_unsaved_session_says_untitled(self):
        window = _WindowStub(path=None)
        window.call("_update_window_title")
        self.assertIn("Untitled workspace", window.title)

    def test_unsaved_changes_are_marked_in_the_title(self):
        window = _WindowStub(path="a.aoews", dirty=True)
        window.call("_update_window_title")
        self.assertTrue(window.title.startswith("*"))

    def test_the_save_tooltip_names_the_file_it_would_overwrite(self):
        path = os.path.join("study", "sweep.aoews")
        window = _WindowStub(path=path)
        window.call("_update_workspace_actions")
        self.assertIn(path, window.save_action.tip)
        self.assertIn("Overwrite", window.save_action.tip)

    def test_the_save_tooltip_says_when_there_is_no_file_yet(self):
        window = _WindowStub(path=None)
        window.call("_update_workspace_actions")
        self.assertIn("ask where to save", window.save_action.tip)

    def test_the_save_as_tooltip_says_it_continues_in_the_new_file(self):
        window = _WindowStub(path="old.aoews")
        window.call("_update_workspace_actions")
        self.assertIn("old.aoews", window.save_as_action.tip)
        self.assertIn("new workspace file", window.save_as_action.tip)

    def test_updating_the_actions_before_they_exist_is_harmless(self):
        """_build_toolbar calls it while the bar is half-built."""
        window = _WindowStub()
        del window.save_action
        window.call("_update_workspace_actions")   # must not raise

    def test_collect_definitions_separates_formulas_from_aliases(self):
        session = _FakeSession(
            computed={"CLIMB": "deriv(ALTITUDE, TIME)"},
            aliases={"Angle of attack": "ALPHA"},
            plain=["ALPHA", "ALTITUDE", "TIME"],
        )
        window = _WindowStub(session=session)
        computed, aliases = window.call("_collect_definitions")
        self.assertEqual(computed, {"CLIMB": "deriv(ALTITUDE, TIME)"})
        self.assertEqual(aliases, {"Angle of attack": "ALPHA"})

    def test_a_computed_channel_is_never_reported_as_an_alias(self):
        """resolve() on a computed name must not put it in both maps."""
        session = _FakeSession(computed={"CLIMB": "deriv(ALTITUDE, TIME)"},
                               aliases={"CLIMB": "SOMETHING_ELSE"},
                               plain=["ALTITUDE", "TIME"])
        window = _WindowStub(session=session)
        computed, aliases = window.call("_collect_definitions")
        self.assertIn("CLIMB", computed)
        self.assertNotIn("CLIMB", aliases)

    def test_save_uses_the_current_path_without_asking(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "w.aoews")
            window = _WindowStub(path=path)
            original = main_window.QFileDialog
            main_window.QFileDialog = _RefusingDialog()
            try:
                window.call("save_workspace")
            finally:
                main_window.QFileDialog = original
            self.assertTrue(os.path.exists(path))
            self.assertIn(path, window.status)
            self.assertFalse(window.state.dirty)

    def test_save_as_always_asks_and_adopts_the_new_path(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            first = os.path.join(folder, "first.aoews")
            second = os.path.join(folder, "second.aoews")
            window = _WindowStub(path=first)
            original = main_window.QFileDialog
            main_window.QFileDialog = _AnsweringDialog(second)
            try:
                window.call("save_workspace_as")
            finally:
                main_window.QFileDialog = original
            self.assertTrue(os.path.exists(second))
            self.assertFalse(os.path.exists(first))   # left alone
            self.assertEqual(window.state.workspace_path, second)
            self.assertIn("second.aoews", window.title)

    def test_save_as_appends_the_extension(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            window = _WindowStub()
            original = main_window.QFileDialog
            main_window.QFileDialog = _AnsweringDialog(
                os.path.join(folder, "no-suffix"))
            try:
                window.call("save_workspace_as")
            finally:
                main_window.QFileDialog = original
            self.assertTrue(window.state.workspace_path.endswith(".aoews"))

    def test_cancelling_save_as_changes_nothing(self):
        window = _WindowStub(path="kept.aoews")
        original = main_window.QFileDialog
        main_window.QFileDialog = _AnsweringDialog("")
        try:
            window.call("save_workspace_as")
        finally:
            main_window.QFileDialog = original
        self.assertEqual(window.state.workspace_path, "kept.aoews")

    def test_the_saved_file_carries_the_definitions_and_the_filters(self):
        import tempfile
        session = _FakeSession(
            computed={"CLIMB": "deriv(ALTITUDE, TIME)"},
            aliases={"Angle of attack": "ALPHA"},
            plain=["ALPHA", "ALTITUDE", "TIME"],
        )
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "w.aoews")
            window = _WindowStub(session=session, path=path)
            iso = window.state.iso_state("MASS")
            iso.active = True
            iso.value_enabled = {50000.0: True}
            window.call("save_workspace")
            data = model.load_workspace(path)
        self.assertEqual(data["computed"], {"CLIMB": "deriv(ALTITUDE, TIME)"})
        self.assertEqual(data["aliases"], {"Angle of attack": "ALPHA"})
        self.assertEqual(len(data["iso_states"]), 1)
        # The status line tells the user what went with it.
        self.assertIn("1 formula(s)", window.status)
        self.assertIn("1 alias(es)", window.status)
        self.assertIn("1 ISO filter(s)", window.status)


class _AnsweringDialog:
    """A QFileDialog stand-in that returns a fixed path."""

    def __init__(self, path):
        self._path = path

    def getSaveFileName(self, *_args, **_kwargs):
        return (self._path, "")


class _RefusingDialog:
    """A QFileDialog stand-in that fails the test if it is consulted."""

    def getSaveFileName(self, *_args, **_kwargs):
        raise AssertionError("Save must not ask when a path is already known")


class TestFormulaHelpLogarithms(unittest.TestCase):
    """log is natural, ln is the same function, log10 is base ten.

    Which logarithm a name gives is the easiest thing in the language to assume
    wrongly -- a pocket calculator's log is base ten and this one is not -- and
    the dialog listed neither ln nor the distinction.
    """

    def setUp(self):
        self.help = panels.FORMULA_HELP

    def test_all_three_names_are_listed(self):
        for name in ("log(x)", "ln(x)", "log10(x)"):
            self.assertIn(name, self.help, name)

    def test_ln_is_offered_in_the_function_list(self):
        self.assertIn("exp log ln log10", self.help)

    def test_it_says_log_is_natural_and_not_base_ten(self):
        self.assertIn("Natural</b> logarithm", self.help)
        self.assertIn("not</i> base 10", self.help)

    def test_it_says_ln_and_log_are_the_same_function(self):
        self.assertIn("identical results", self.help)

    def test_it_says_log10_is_base_ten(self):
        self.assertIn("Base-10 logarithm", self.help)

    def test_it_states_the_domain_and_the_nan_rule(self):
        self.assertIn("zero or negative", self.help)
        self.assertIn("infinity would destroy", self.help)

    def test_it_names_the_inverses(self):
        self.assertIn("exp(x)", self.help)
        self.assertIn("pow(10, x)", self.help)

    def test_it_distinguishes_the_log_y_checkbox(self):
        """Taking a logarithm is not the same as drawing on a log axis."""
        self.assertIn("Log Y", self.help)
        self.assertIn("leaves the data alone", self.help)

    def test_it_gives_worked_examples(self):
        self.assertIn("20 * log10(", self.help)


class TestSaveAsIcon(unittest.TestCase):
    def test_the_painter_is_registered(self):
        from aoeviewer.ui import icons
        self.assertIn("save-as", icons._PAINTERS)

    def test_it_is_distinct_from_save(self):
        from aoeviewer.ui import icons
        self.assertIsNot(icons._PAINTERS["save-as"], icons._PAINTERS["save"])


class TestConflictingVariableLimits(unittest.TestCase):
    """Two constrained y parameters on one y axis.

    Three tiles: the first draws P1, the second draws P1 and P2, the third draws
    P2. Constrain P1 from the first and P2 from the third, and the second tile
    is asked for two different windows on one axis.

    This used to be resolved by taking the union, which honours neither and,
    when the magnitudes differ, destroys the smaller channel: ALTITUDE at
    [0, 13000] unioned with MACH at [0, 1] leaves Mach on 0.008% of the axis --
    a flat line on the bottom edge, with nothing on screen to say why.
    """

    def setUp(self):
        self.state = AppState()
        page = self.state.page
        page.rows, page.cols = 1, 3
        self.t1, self.t2, self.t3 = page.tiles[0], page.tiles[1], page.tiles[2]
        for tile in (self.t1, self.t2, self.t3):
            tile.x_param = "TIME"
        self.t1.y_params = ["ALTITUDE"]
        self.t2.y_params = ["ALTITUDE", "MACH"]
        self.t3.y_params = ["MACH"]

    def _constrain_both(self):
        self.state.set_global_y_limit("ALTITUDE", 0.0, 13000.0)
        self.state.set_global_y_limit("MACH", 0.0, 1.0)
        self.state.apply_global_limits_everywhere()

    def test_the_unambiguous_tiles_are_constrained(self):
        self._constrain_both()
        self.assertTrue(self.t1.lock_y)
        self.assertEqual((self.t1.y_min, self.t1.y_max), (0.0, 13000.0))
        self.assertTrue(self.t3.lock_y)
        self.assertEqual((self.t3.y_min, self.t3.y_max), (0.0, 1.0))

    def test_the_conflicted_tile_is_left_free(self):
        self._constrain_both()
        self.assertFalse(self.t2.lock_y)

    def test_the_conflict_is_reported_with_both_ranges(self):
        self._constrain_both()
        self.assertEqual(self.state.y_limit_conflict(self.t2),
                         [("ALTITUDE", 0.0, 13000.0), ("MACH", 0.0, 1.0)])

    def test_the_unambiguous_tiles_report_no_conflict(self):
        self._constrain_both()
        self.assertEqual(self.state.y_limit_conflict(self.t1), [])
        self.assertEqual(self.state.y_limit_conflict(self.t3), [])

    def test_the_union_is_no_longer_imposed(self):
        """The specific regression: Mach must not be squashed to nothing."""
        self._constrain_both()
        span = self.t2.y_max - self.t2.y_min
        self.assertNotEqual((self.t2.y_min, self.t2.y_max), (0.0, 13000.0))
        self.assertLess(span, 13000.0)

    def test_agreeing_parameters_are_still_applied(self):
        """Only disagreement is a conflict; two equal windows are one window."""
        self.state.set_global_y_limit("ALTITUDE", -5.0, 5.0)
        self.state.set_global_y_limit("MACH", -5.0, 5.0)
        self.state.apply_global_limits_everywhere()
        self.assertEqual(self.state.y_limit_conflict(self.t2), [])
        self.assertTrue(self.t2.lock_y)
        self.assertEqual((self.t2.y_min, self.t2.y_max), (-5.0, 5.0))

    def test_one_constrained_parameter_is_never_a_conflict(self):
        self.state.set_global_y_limit("ALTITUDE", 0.0, 13000.0)
        self.state.apply_global_limits_everywhere()
        self.assertEqual(self.state.y_limit_conflict(self.t2), [])
        self.assertTrue(self.t2.lock_y)
        self.assertEqual((self.t2.y_min, self.t2.y_max), (0.0, 13000.0))

    def test_removing_one_constraint_resolves_the_conflict(self):
        self._constrain_both()
        self.state.clear_global_limit("MACH")
        self.state.apply_global_limits_everywhere()
        self.assertEqual(self.state.y_limit_conflict(self.t2), [])
        self.assertTrue(self.t2.lock_y)
        self.assertEqual((self.t2.y_min, self.t2.y_max), (0.0, 13000.0))

    def test_a_hand_locked_tile_keeps_its_own_range(self):
        """A conflict must not silently unlock what the user locked."""
        self.t2.lock_y = True
        self.t2.y_min, self.t2.y_max = -1.0, 2.0
        self._constrain_both()
        self.assertTrue(self.t2.lock_y)
        self.assertEqual((self.t2.y_min, self.t2.y_max), (-1.0, 2.0))

    def test_three_way_disagreement_reports_all_three(self):
        self.t2.y_params = ["ALTITUDE", "MACH", "NZ"]
        self.state.set_global_y_limit("ALTITUDE", 0.0, 13000.0)
        self.state.set_global_y_limit("MACH", 0.0, 1.0)
        self.state.set_global_y_limit("NZ", -1.0, 4.0)
        self.state.apply_global_limits_everywhere()
        self.assertEqual([p for p, _l, _h in self.state.y_limit_conflict(self.t2)],
                         ["ALTITUDE", "MACH", "NZ"])

    def test_x_cannot_conflict_because_a_tile_has_one_x(self):
        """The same failure is structurally impossible on the x axis."""
        self.state.set_global_x_limit("TIME", 0.0, 60.0)
        self.state.set_global_x_limit("ALTITUDE", 0.0, 13000.0)
        self.state.apply_global_limits_everywhere()
        for tile in (self.t1, self.t2, self.t3):
            self.assertTrue(tile.lock_x)
            self.assertEqual((tile.x_min, tile.x_max), (0.0, 60.0))

    def test_the_same_channel_carries_one_window_per_axis(self):
        """ALTITUDE as an x axis and as a y axis are separate constraints."""
        self.t2.x_param = "ALTITUDE"
        self.t2.y_params = ["MACH"]
        self.state.set_global_x_limit("ALTITUDE", 0.0, 13000.0)
        self.state.set_global_y_limit("ALTITUDE", 0.0, 9000.0)
        self.state.set_global_y_limit("MACH", 0.0, 1.0)
        self.state.apply_global_limits_everywhere()
        self.assertEqual((self.t2.x_min, self.t2.x_max), (0.0, 13000.0))
        self.assertEqual((self.t1.y_min, self.t1.y_max), (0.0, 9000.0))

    def test_a_3d_tile_is_left_alone(self):
        """Its frame is the 3D view; the data ranges mean nothing to it."""
        self.t2.plot3d = True
        self.t2.y_params = ["ALTITUDE"]
        self.t2.z_param = "MACH"
        self.state.set_global_y_limit("ALTITUDE", 0.0, 13000.0)
        self.state.set_global_x_limit("TIME", 0.0, 60.0)
        self.assertFalse(self.state.apply_global_limits(self.t2))
        self.assertFalse(self.t2.lock_y)
        self.assertFalse(self.t2.lock_x)

    def test_the_union_is_available_as_a_deliberate_choice(self):
        """The SCALE tab offers it; taking it locks the tile, not the variables."""
        self._constrain_both()
        panel = panels.ScalePanel(_AnnotationSession(), self.state)
        self.state.active_tile = 1
        panel._resolve_conflict_with_union()
        self.assertTrue(self.t2.lock_y)
        self.assertEqual((self.t2.y_min, self.t2.y_max), (0.0, 13000.0))
        # The variables keep the windows the other tiles were given.
        self.assertEqual(self.state.global_y_limits["ALTITUDE"], (0.0, 13000.0))
        self.assertEqual(self.state.global_y_limits["MACH"], (0.0, 1.0))
        self.assertEqual((self.t3.y_min, self.t3.y_max), (0.0, 1.0))

    def test_taking_the_union_does_not_survive_as_a_constraint(self):
        """It is the tile's own lock, so the conflict is still reported."""
        self._constrain_both()
        panel = panels.ScalePanel(_AnnotationSession(), self.state)
        self.state.active_tile = 1
        panel._resolve_conflict_with_union()
        self.assertTrue(self.state.y_limit_conflict(self.t2))
        # And a later redraw must not wipe the lock the user just chose.
        self.state.apply_global_limits_everywhere()
        self.assertTrue(self.t2.lock_y)
        self.assertEqual((self.t2.y_min, self.t2.y_max), (0.0, 13000.0))
