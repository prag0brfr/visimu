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

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import numpy as np

import qt_stubs  # noqa: F401  (installs the stub modules; must precede the ui import)
from PySide6.QtCore import Qt  # noqa: E402  (the stub installed above)

from aoeviewer import scales
from aoeviewer import theme as theme_module
from aoeviewer.model import MAX_TILES, AppState
from aoeviewer.scales import resample_key
from aoeviewer.ui import panels
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

    def test_names_from_a_file_cannot_break_the_rich_text(self):
        # Parameter names come out of the user's files, so one containing "<"
        # would otherwise swallow the rest of the line as an unknown tag.
        self.assertEqual(panels._escape("A<B & C>D"), "A&lt;B &amp; C&gt;D")


class _IsoSession:
    """Just the two calls the ISO overview makes of a session."""

    def __init__(self, values):
        self._values = {k: np.array(v, dtype=np.float32)
                        for k, v in values.items()}
        self.parameters = sorted(self._values)

    def iso_distinct(self, name):
        return self._values.get(name, np.empty(0, dtype=np.float32))

    def format_value(self, value):
        return "{0:g}".format(value)


class TestIsoOverview(unittest.TestCase):
    """Reading the selection back out of the panel's state."""

    def setUp(self):
        self.panel = object.__new__(panels.IsoPanel)
        self.panel._session = _IsoSession({
            "MACH": [0.7, 0.75, 0.8],
            "WEIGHT": [180.0, 200.0],
        })
        self.panel._state = AppState()

    def test_untouched_values_count_as_selected(self):
        # Absent from value_enabled means on, which is how a freshly added
        # filter behaves; a summary that called them off would contradict the
        # list beside it.
        self.panel._state.iso_state("MACH")
        labels, selected = self.panel._selection_for("MACH")
        self.assertEqual(labels, ["0.7", "0.75", "0.8"])
        self.assertEqual(selected, labels)

    def test_it_follows_what_was_unticked(self):
        iso = self.panel._state.iso_state("MACH")
        iso.value_enabled[0.75] = False
        labels, selected = self.panel._selection_for("MACH")
        self.assertEqual(labels, ["0.7", "0.75", "0.8"])
        self.assertEqual(selected, ["0.7", "0.8"])

    def test_grouped_mode_reports_the_intervals(self):
        # Grouped, the user ticks named intervals; the underlying numbers are
        # not what is on screen, so they are not what is reported.
        iso = self.panel._state.iso_state("MACH")
        iso.grouped = True
        iso.groups = [("low", 0.0, 0.75, True), ("high", 0.75, 1.0, False)]
        labels, selected = self.panel._selection_for("MACH")
        self.assertEqual(labels, ["low", "high"])
        self.assertEqual(selected, ["low"])

    def test_a_parameter_with_no_filter_is_empty(self):
        self.assertEqual(self.panel._selection_for("MACH"), ([], []))

    def _wire_lists(self):
        """The two list widgets and the box, as __init__ would leave them."""
        self.panel._updating = False
        self.panel.filter_list = qt_stubs.QListWidget()
        self.panel.overview = qt_stubs.QTextBrowser()
        self.panel.filter_list.itemChanged.connect(self.panel._on_filter_toggled)
        for name in sorted(self.panel._state.iso_states):
            item = qt_stubs.QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.panel.filter_list.addItem(item)

    def test_the_badge_reaches_the_row(self):
        iso = self.panel._state.iso_state("MACH")
        iso.value_enabled[0.75] = False
        self._wire_lists()

        self.panel._refresh_selection_views()

        row = self.panel.filter_list.item(0)
        self.assertIn("MACH", row.text())
        self.assertIn("2/3", row.text())
        self.assertIn("0.7, 0.8", row.toolTip())

    def test_filling_a_badge_does_not_toggle_the_filter(self):
        # setText raises itemChanged exactly as ticking the checkbox does. The
        # handler reads the check state and writes it to iso.active, so without
        # the guard, restating a row would switch its filter off.
        iso = self.panel._state.iso_state("MACH")
        iso.active = True
        self._wire_lists()
        # No check state was ever set on these stub rows, so an unguarded
        # handler would read None and turn the filter off.
        self.panel._refresh_selection_views()
        self.assertTrue(iso.active)

    def test_every_filter_reaches_the_overview(self):
        self.panel._state.iso_state("MACH")
        weight = self.panel._state.iso_state("WEIGHT")
        weight.value_enabled[180.0] = False
        self._wire_lists()

        self.panel._refresh_selection_views()

        html = self.panel.overview.html
        self.assertIn("MACH", html)
        self.assertIn("WEIGHT", html)
        self.assertIn("all 3", html)      # MACH, untouched
        self.assertIn("200", html)        # WEIGHT, the one still ticked

    def test_an_inactive_filter_is_marked_but_still_listed(self):
        # It is kept precisely so it can be switched back on, so it still shows
        # its values -- but it must not read as if it were narrowing anything.
        iso = self.panel._state.iso_state("MACH")
        iso.active = False
        self._wire_lists()

        self.panel._refresh_selection_views()

        self.assertIn("off", self.panel.overview.html)
        self.assertIn("MACH", self.panel.overview.html)

    def test_no_filters_leaves_the_box_empty(self):
        self._wire_lists()
        self.panel._refresh_selection_views()
        self.assertEqual(self.panel.overview.html, "")

    def test_grouping_with_no_groups_yet_falls_back_to_values(self):
        # iso.grouped is set before the intervals are generated; reporting an
        # empty list there would say "no values" about a full parameter.
        iso = self.panel._state.iso_state("MACH")
        iso.grouped = True
        iso.groups = []
        labels, _selected = self.panel._selection_for("MACH")
        self.assertEqual(labels, ["0.7", "0.75", "0.8"])


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


if __name__ == "__main__":
    unittest.main()
