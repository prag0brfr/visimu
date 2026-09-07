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

from aoeviewer import isogroups
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

    def test_a_changed_selection_does_rebuild(self):
        panel = make_iso_panel()
        iso = panel._state.iso_state("MACH")
        panel._rebuild_tree()
        node = node_named(panel, "MACH")

        iso.value_enabled[panel._distinct("MACH")[0]] = False
        panel._rebuild_tree()

        self.assertIsNot(node_named(panel, "MACH"), node)

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
