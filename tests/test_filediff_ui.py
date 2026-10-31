# Author: Paulo R. A. Gilz
"""The file-to-file comparison, from the panel to the pages.

Three things are checked here, and they are the three the arithmetic cannot
check for itself: that the DIFF tab offers the comparison without sending the
user to another tab, that the report says what the specification says it says
and plots nothing until asked, and that a generated page survives the
workspace it is saved into.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import qt_stubs  # noqa: F401  (installs the stub modules; must precede the ui import)
from PySide6.QtCore import Qt  # noqa: E402

from visimu.diffconfig import Rule                     # noqa: E402
from visimu.filediff import Flag, Report, Violation    # noqa: E402
from visimu.model import AppState, PageState, TileState  # noqa: E402
from visimu.ui import diff_report, panels              # noqa: E402
from visimu.ui import plot_tile as pt                  # noqa: E402
from test_redraw import (                              # noqa: E402
    fit_file_diff_widgets,
    make_drawable_tile,
    make_tile,
)


def a_flag(run="RUN2", param="BETA"):
    """The worked example's flag: BLACK on [3, 5], RED on [2, 3)."""
    return Flag(run, param, "TIME",
                [Violation("BLACK", 6.0, [(3.0, 5.0)]),
                 Violation("RED", 4.0, [(2.0, 3.0)])])


class _Sim:
    def __init__(self, index, sim_id):
        self.index = index
        self.sim_id = sim_id


class _File:
    def __init__(self, index, name, runs, headers):
        self.index = index
        self.name = name
        self.simulations = [_Sim(at, run) for at, run in enumerate(runs)]
        self.headers = list(headers)

    @property
    def simulation_count(self):
        return len(self.simulations)


class _Session:
    """Two files of two runs each, with one engineered violation."""

    def __init__(self):
        self.files_list = [
            _File(0, "File1", ["RUN1", "RUN2"], ["TIME", "ALFA", "BETA"]),
            _File(1, "File2", ["RUN1", "RUN2"], ["TIME", "ALFA", "BETA"]),
        ]

    @property
    def file_count(self):
        return len(self.files_list)

    @property
    def files(self):
        return list(self.files_list)

    def file(self, index):
        return self.files_list[index] if 0 <= index < 2 else None

    @property
    def parameters(self):
        return ["TIME", "ALFA", "BETA"]

    def label(self, name):
        return name

    def decimate(self, x, y, _target, _range):
        # Untouched: what the decimation does to a curve is tested where the
        # decimation is, and a fixture that thinned the arrays here would make
        # every sample count in this file a count of something else.
        return x, y

    def channel(self, file_index, sim_index, name):
        # Empty for a channel the file does not carry, which is what the real
        # session does: a reader that invents zeros for an unknown name makes
        # "the channel is not there" untestable.
        if name not in ("TIME", "ALFA", "BETA"):
            return np.array([])
        time = np.arange(0.0, 7.0)
        if name == "TIME":
            return time
        out = np.zeros_like(time)
        if name == "BETA" and file_index == 1 and sim_index == 1:
            out[2] = 5.0
            out[3:6] = 7.0
        return out


def a_panel(state=None):
    panel = object.__new__(panels.DiffPanel)
    panel._session = _Session()
    panel._state = state or AppState()
    panel._updating = False
    panel.x_combo = qt_stubs.QComboBox()
    panel.param_combo = qt_stubs.QComboBox()
    panel.a_combo = qt_stubs.QComboBox()
    panel.b_combo = qt_stubs.QComboBox()
    panel.tile_label = qt_stubs.QLabel()
    panel.state_label = qt_stubs.QLabel()
    panel.tile_changed = qt_stubs.Signal().__get__(panel)
    fit_file_diff_widgets(panel)
    panel.refresh()
    return panel


class TestThePanelOffersTheComparison(unittest.TestCase):
    """Everything the comparison needs is chosen here and nowhere else."""

    def test_both_files_are_listed(self):
        panel = a_panel()
        self.assertEqual(
            [panel.file_a_combo.itemText(i)
             for i in range(panel.file_a_combo.count())],
            ["--", "File1", "File2"])

    def test_choosing_the_files_reaches_the_state(self):
        panel = a_panel()
        panel.file_a_combo.setCurrentIndex(1)
        panel.file_b_combo.setCurrentIndex(2)
        panel._on_file_changed()
        self.assertEqual((panel._state.diff_file_a, panel._state.diff_file_b),
                         (0, 1))

    def test_the_choice_survives_a_refresh(self):
        state = AppState()
        state.diff_file_a, state.diff_file_b = 1, 0
        panel = a_panel(state)
        self.assertEqual(panel.file_a_combo.currentIndex(), 2)
        self.assertEqual(panel.file_b_combo.currentIndex(), 1)

    def test_no_configuration_says_so(self):
        self.assertIn("No thresholds", a_panel().config_label.text())

    def test_a_configuration_is_counted_by_rule_and_by_group(self):
        state = AppState()
        state.diff_config_path = os.path.join("some", "where", "limits.yaml")
        state.diff_rules = [Rule("BLACK", "ALFA", "TIME", 25.0).to_dict(),
                            Rule("BLACK", "BETA", "TIME", 10.0).to_dict(),
                            Rule("AMBER", "ALFA", "TIME", 10.0).to_dict()]
        panel = a_panel(state)
        text = panel.config_label.text()
        self.assertIn("limits.yaml", text)
        self.assertIn("3", text)
        self.assertIn("2", text)

    def test_every_group_gets_a_swatch(self):
        state = AppState()
        state.diff_rules = [Rule("BLACK", "A", "TIME", 9.0).to_dict(),
                            Rule("AMBER", "A", "TIME", 4.0).to_dict()]
        panel = a_panel(state)
        self.assertEqual(sorted(panel._group_swatches), ["AMBER", "BLACK"])
        self.assertTrue(panel.group_box.isVisible())

    def test_the_default_colours_are_the_specification_s(self):
        """Grey, red, orange, green -- defaults, not meanings: a group called
        BLACK is an identifier, and nothing here reads its name."""
        state = AppState()
        state.diff_rules = [
            Rule(name, "A", "TIME", float(10 - at)).to_dict()
            for at, name in enumerate(["ONE", "TWO", "THREE", "FOUR"])]
        a_panel(state)
        self.assertEqual(
            [state.diff_group_colors[name]
             for name in ("ONE", "TWO", "THREE", "FOUR")],
            list(panels.DiffPanel.DEFAULT_GROUP_COLORS))

    def test_a_colour_the_user_chose_is_kept(self):
        state = AppState()
        state.diff_rules = [Rule("BLACK", "A", "TIME", 9.0).to_dict()]
        state.diff_group_colors["BLACK"] = "#123456"
        a_panel(state)
        self.assertEqual(state.diff_group_colors["BLACK"], "#123456")

    def test_the_rows_of_an_unloaded_configuration_go(self):
        state = AppState()
        state.diff_rules = [Rule("BLACK", "A", "TIME", 9.0).to_dict(),
                            Rule("AMBER", "A", "TIME", 4.0).to_dict()]
        panel = a_panel(state)
        self.assertEqual(panel.group_layout.count(), 2)
        state.diff_rules = [Rule("ONLY", "A", "TIME", 1.0).to_dict()]
        panel.refresh_files()
        self.assertEqual(panel.group_layout.count(), 1)
        self.assertEqual(list(panel._group_swatches), ["ONLY"])

    def test_the_report_button_is_dead_until_there_is_one(self):
        panel = a_panel()
        self.assertFalse(panel.report_button.isEnabled())

    def test_a_report_is_summarised(self):
        state = AppState()
        state.diff_report = Report(0, 1, "File1", "File2",
                                   [a_flag()]).to_dict()
        panel = a_panel(state)
        self.assertIn("1 run", panel.file_state_label.text())
        self.assertTrue(panel.report_button.isEnabled())

    def test_the_generated_pages_are_offered(self):
        state = AppState()
        state.build_diff_pages("RUN2", [a_flag()], 0, 1)
        panel = a_panel(state)
        self.assertEqual(
            [panel.pages_combo.itemText(i)
             for i in range(panel.pages_combo.count())],
            ["DIFF (1)"])
        self.assertTrue(panel.pages_row.isVisible())

    def test_choosing_a_page_goes_to_it(self):
        state = AppState()
        state.build_diff_pages("RUN2", [a_flag()], 0, 1)
        state.active_page = 0
        panel = a_panel(state)
        panel._on_page_chosen(0)
        self.assertEqual(state.active_page, 1)

    def test_the_run_to_run_half_is_untouched(self):
        """The existing comparison stays exactly where it was."""
        panel = a_panel()
        self.assertTrue(hasattr(panel, "param_combo"))
        self.assertTrue(hasattr(panel, "a_combo"))
        self.assertEqual(panel.a_combo.count(), 5)   # "--" and four runs


class TestThePages(unittest.TestCase):
    """What plotting a report line puts on screen."""

    def flags(self, count):
        return [a_flag(param="P{0}".format(at)) for at in range(count)]

    def test_one_tile_per_flagged_channel(self):
        state = AppState()
        pages = state.build_diff_pages("RUN2", self.flags(3), 0, 1)
        self.assertEqual(len(pages), 1)
        filled = [tile for tile in pages[0].tiles if tile.is_file_diff()]
        self.assertEqual([tile.fdiff_param for tile in filled],
                         ["P0", "P1", "P2"])

    def test_the_grid_is_four_by_four(self):
        state = AppState()
        page = state.build_diff_pages("RUN2", self.flags(2), 0, 1)[0]
        self.assertEqual((page.rows, page.cols), (4, 4))

    def test_a_seventeenth_channel_starts_a_second_page(self):
        state = AppState()
        pages = state.build_diff_pages("RUN2", self.flags(17), 0, 1)
        self.assertEqual([page.name for page in pages],
                         ["DIFF (1)", "DIFF (2)"])
        self.assertEqual(
            len([t for t in pages[0].tiles if t.is_file_diff()]), 16)
        self.assertEqual(
            len([t for t in pages[1].tiles if t.is_file_diff()]), 1)

    def test_a_second_run_continues_the_numbering(self):
        """Not "DIFF (1) (2)": the pages of a comparison read as one
        sequence however many runs were plotted."""
        state = AppState()
        state.build_diff_pages("RUN2", self.flags(1), 0, 1)
        second = state.build_diff_pages("RUN3", self.flags(1), 0, 1)
        self.assertEqual([page.name for page in second], ["DIFF (2)"])

    def test_the_tile_carries_the_bands_rather_than_the_rules(self):
        state = AppState()
        page = state.build_diff_pages("RUN2", [a_flag()], 0, 1)[0]
        self.assertEqual(page.tiles[0].fdiff_bands,
                         [("BLACK", 3.0, 5.0), ("RED", 2.0, 3.0)])

    def test_the_tile_names_the_run_rather_than_its_index(self):
        """Run 3 of one build is run 7 of the next, and the files were
        matched on the name."""
        state = AppState()
        page = state.build_diff_pages("RUN2", [a_flag()], 0, 1)[0]
        self.assertEqual(page.tiles[0].fdiff_run, "RUN2")

    def test_the_legend_names_the_file(self):
        state = AppState()
        page = state.build_diff_pages("RUN2", [a_flag()], 0, 1)[0]
        self.assertTrue(page.tiles[0].show_legend)
        self.assertTrue(page.tiles[0].legend_show_file)
        self.assertTrue(page.tiles[0].legend_show_param)

    def test_a_generated_tile_is_fitted_and_not_empty(self):
        state = AppState()
        page = state.build_diff_pages("RUN2", [a_flag()], 0, 1)[0]
        self.assertFalse(page.tiles[0].view_initialised)
        self.assertFalse(page.tiles[0].is_empty())

    def test_the_x_axis_is_the_rule_s(self):
        state = AppState()
        flag = Flag("RUN2", "BETA", "DISTANCE",
                    [Violation("BLACK", 6.0, [(1.0, 2.0)])])
        page = state.build_diff_pages("RUN2", [flag], 0, 1)[0]
        self.assertEqual(page.tiles[0].x_param, "DISTANCE")

    def test_nothing_flagged_adds_no_page(self):
        state = AppState()
        before = len(state.pages)
        self.assertEqual(state.build_diff_pages("RUN2", [], 0, 1), [])
        self.assertEqual(len(state.pages), before)

    def test_the_generated_pages_are_found_again(self):
        state = AppState()
        state.build_diff_pages("RUN2", self.flags(1), 0, 1)
        self.assertEqual(state.diff_pages(), [1])

    def test_a_renamed_page_is_still_one_of_them(self):
        state = AppState()
        state.build_diff_pages("RUN2", self.flags(1), 0, 1)
        state.pages[1].name = "the regression"
        self.assertEqual(state.diff_pages(), [1])

    def test_a_page_called_diff_by_hand_is_not_one_of_them(self):
        state = AppState()
        state.pages[0].name = "DIFF (1)"
        self.assertEqual(state.diff_pages(), [])

    def test_a_second_comparison_does_not_leave_the_first_s_pages(self):
        state = AppState()
        state.build_diff_pages("RUN2", self.flags(2), 0, 1)
        self.assertEqual(state.clear_diff_pages(), 1)
        self.assertEqual(state.diff_pages(), [])
        self.assertEqual(len(state.pages), 1)

    def test_clearing_keeps_a_page_to_look_at(self):
        state = AppState()
        state.pages = []
        state.build_diff_pages("RUN2", self.flags(1), 0, 1)
        state.clear_diff_pages()
        self.assertEqual(len(state.pages), 1)
        self.assertFalse(state.pages[0].tiles[0].is_file_diff())


class TestTheBandsDoNotOutliveTheirTile(unittest.TestCase):
    """The bands belong to the tile state and are drawn on the tile widget.

    The widgets are reused from page to page -- that is what makes changing
    page quick -- so a widget that drew an overlay and is now drawing
    something else arrives at the rebuild still holding the bands it drew
    last. That is what was happening: page one's shading showed through on
    page two, and on pages that had nothing to do with the comparison.
    """

    def tile(self):
        # Drawable, because these drive the whole rebuild rather than the one
        # method: what is being checked is what a rebuild does to bands it did
        # not draw.
        tile = make_drawable_tile(_Session())
        tile._sync_report = (0, 0)
        shape = tile._state.tile
        shape.x_param = "TIME"
        shape.fdiff_run = "RUN2"
        shape.fdiff_param = "BETA"
        shape.fdiff_file_a, shape.fdiff_file_b = 0, 1
        shape.fdiff_bands = [("BLACK", 3.0, 5.0)]
        tile._rebuild_file_diff(tile._plot_item, shape, [])
        self.assertEqual(len(tile._fdiff_items), 1, "nothing was drawn")
        return tile, shape

    def test_a_tile_that_stops_being_an_overlay_drops_them(self):
        tile, shape = self.tile()
        # What the second page of a comparison does to a widget that drew the
        # first: the state under it is replaced, and the widget redraws.
        shape.fdiff_param = ""
        shape.fdiff_file_a = shape.fdiff_file_b = -1
        shape.fdiff_bands = []
        shape.y_params = ["ALFA"]
        tile.rebuild()
        self.assertEqual(tile._fdiff_items, [])

    def test_an_ordinary_page_drops_them(self):
        """Loading a page that has nothing to do with the comparison."""
        tile, shape = self.tile()
        tile._state.tile.__init__()
        tile.rebuild()
        self.assertEqual(tile._fdiff_items, [])

    def test_an_empty_tile_drops_them(self):
        tile, shape = self.tile()
        shape.fdiff_param = ""
        shape.fdiff_file_a = shape.fdiff_file_b = -1
        shape.x_param = ""
        shape.y_params = []
        tile.rebuild()
        self.assertEqual(tile._fdiff_items, [])

    def test_a_tile_that_is_still_an_overlay_keeps_them(self):
        """Dropping them on every rebuild would make them flicker away on a
        restyle, a grid toggle or a marker drag."""
        tile, shape = self.tile()
        tile.rebuild()
        self.assertEqual(len(tile._fdiff_items), 1)

    def test_the_second_page_s_bands_are_its_own(self):
        tile, shape = self.tile()
        shape.fdiff_param = "ALFA"
        shape.fdiff_bands = [("RED", 1.0, 2.0)]
        tile.rebuild()
        self.assertEqual([item.getRegion() for item in tile._fdiff_items],
                         [(1.0, 2.0)])


class TestThePagesSurviveTheWorkspace(unittest.TestCase):
    """FR-6: reopening restores the comparison without recomputing it."""

    def saved(self):
        state = AppState()
        state.diff_file_a, state.diff_file_b = 0, 1
        state.diff_config_path = "/limits.yaml"
        state.diff_rules = [Rule("BLACK", "BETA", "TIME", 6.0).to_dict()]
        state.diff_group_colors = {"BLACK": "#101010", "RED": "#ff0000"}
        state.diff_report = Report(0, 1, "File1", "File2",
                                   [a_flag()]).to_dict()
        state.build_diff_pages("RUN2", [a_flag()], 0, 1)
        copy = AppState()
        copy.load_dict(state.to_dict())
        return copy

    def test_the_files_come_back(self):
        state = self.saved()
        self.assertEqual((state.diff_file_a, state.diff_file_b), (0, 1))

    def test_the_configuration_comes_back_parsed(self):
        state = self.saved()
        self.assertEqual(state.diff_config_path, "/limits.yaml")
        self.assertEqual(Rule.from_dict(state.diff_rules[0]).threshold, 6.0)

    def test_the_colours_come_back(self):
        self.assertEqual(self.saved().diff_group_colors["BLACK"], "#101010")

    def test_the_report_comes_back(self):
        report = Report.from_dict(self.saved().diff_report)
        self.assertEqual(report.line_for("RUN2"),
                         "RUN2: BETA(BLACK), BETA(RED)")

    def test_the_pages_come_back_with_their_bands(self):
        state = self.saved()
        self.assertEqual(state.diff_pages(), [1])
        tile = state.pages[1].tiles[0]
        self.assertEqual(tile.fdiff_run, "RUN2")
        self.assertEqual(tile.fdiff_param, "BETA")
        self.assertEqual((tile.fdiff_file_a, tile.fdiff_file_b), (0, 1))
        self.assertEqual(tile.fdiff_bands,
                         [("BLACK", 3.0, 5.0), ("RED", 2.0, 3.0)])

    def test_a_workspace_from_before_the_comparison_opens(self):
        state = AppState()
        state.load_dict({"pages": [PageState().to_dict()]})
        self.assertEqual(state.diff_rules, [])
        self.assertIsNone(state.diff_report)
        self.assertEqual(state.diff_pages(), [])

    def test_a_mangled_report_costs_only_the_report(self):
        state = AppState()
        state.load_dict({"pages": [PageState().to_dict()],
                         "diff_report": "not a report",
                         "diff_rules": "not a list",
                         "diff_group_colors": 7})
        self.assertIsNone(state.diff_report)
        self.assertEqual(state.diff_rules, [])
        self.assertEqual(state.diff_group_colors, {})
        self.assertEqual(len(state.pages), 1)

    def test_a_band_that_is_not_one_is_dropped(self):
        tile = TileState.from_dict(
            {"fdiff_param": "BETA", "fdiff_file_a": 0, "fdiff_file_b": 1,
             "fdiff_bands": [["BLACK", 1.0, 2.0], ["RED"], None, 5]})
        self.assertEqual(tile.fdiff_bands, [("BLACK", 1.0, 2.0)])


class TestTheReportWindow(unittest.TestCase):
    """FR-4: a window of its own, one line per flagged run, nothing plotted
    until the user says so."""

    # A sentinel rather than None: {} is a colour map, and a test about a
    # group with no colour has to be able to pass one.
    NO_COLORS = object()

    def window(self, report=None, colors=NO_COLORS, total=2):
        report = report or Report(0, 1, "File1", "File2", [a_flag()])
        if colors is self.NO_COLORS:
            colors = {"BLACK": "#101010", "RED": "#dd2222"}
        return diff_report.DiffReportDialog(report, colors, total)

    def test_it_is_not_modal(self):
        """The point of plotting a line is to look at the plots."""
        self.assertFalse(self.window().isModal())

    def test_one_line_per_flagged_run(self):
        report = Report(0, 1, "File1", "File2",
                        [a_flag("RUN2"), a_flag("RUN2", "ALFA"),
                         a_flag("RUN4")])
        self.assertEqual(self.window(report).runs(), ["RUN2", "RUN4"])

    def test_a_run_with_nothing_is_not_a_line(self):
        self.assertEqual(self.window().runs(), ["RUN2"])

    def test_the_line_is_the_specification_s_format(self):
        window = self.window()
        item = window.tree.topLevelItem(0)
        self.assertEqual(item.text(diff_report.COLUMN_RUN),
                         "RUN2: BETA(BLACK), BETA(RED)")

    def test_each_group_is_in_its_own_colour(self):
        window = self.window()
        label = window.tree.itemWidget(window.tree.topLevelItem(0),
                                       diff_report.COLUMN_RUN)
        self.assertIn("#101010", label.text())
        self.assertIn("#dd2222", label.text())

    def test_a_group_with_no_colour_is_still_readable(self):
        window = self.window(colors={})
        label = window.tree.itemWidget(window.tree.topLevelItem(0),
                                       diff_report.COLUMN_RUN)
        self.assertIn(diff_report.DEFAULT_GROUP_COLOR, label.text())

    def test_a_channel_named_like_markup_reads_as_a_channel(self):
        """Channel names come from whatever the solver wrote."""
        report = Report(0, 1, "a", "b", [a_flag(param="<rate>")])
        label = diff_report.line_html(report.flags, {}, "RUN2")
        self.assertIn("&lt;rate&gt;", label)

    def test_nothing_is_plotted_by_opening_it(self):
        asked = []
        window = self.window()
        window.plot_requested.connect(asked.append)
        self.assertEqual(asked, [])

    def test_the_per_line_button_asks_for_that_run(self):
        asked = []
        window = self.window()
        window.plot_requested.connect(asked.append)
        button = window.tree.itemWidget(window.tree.topLevelItem(0),
                                        diff_report.COLUMN_ACTION)
        button.clicked.emit()
        self.assertEqual(asked, ["RUN2"])

    def test_a_double_click_asks_for_that_run(self):
        asked = []
        window = self.window()
        window.plot_requested.connect(asked.append)
        window.tree.double_click(window.tree.topLevelItem(0))
        self.assertEqual(asked, ["RUN2"])

    def test_the_right_run_is_asked_for_from_the_right_line(self):
        """The lambda per row has to close over its own run."""
        report = Report(0, 1, "a", "b", [a_flag("RUN2"), a_flag("RUN7")])
        asked = []
        window = self.window(report)
        window.plot_requested.connect(asked.append)
        for row in range(2):
            window.tree.itemWidget(window.tree.topLevelItem(row),
                                   diff_report.COLUMN_ACTION).clicked.emit()
        self.assertEqual(asked, ["RUN2", "RUN7"])

    def test_closing_it_plots_nothing(self):
        asked = []
        window = self.window()
        window.plot_requested.connect(asked.append)
        window.reject()
        self.assertEqual(asked, [])

    def test_a_clean_comparison_says_so(self):
        window = self.window(Report(0, 1, "File1", "File2", []))
        self.assertEqual(window.runs(), [])
        self.assertIn("agree", window.summary.text())

    def test_the_count_is_of_the_runs_in_the_file(self):
        self.assertIn("of 2", self.window(total=2).summary.text())

    def test_what_could_not_be_compared_is_said(self):
        report = Report(0, 1, "a", "b", [a_flag()],
                        ["RUN3/BETA: the two runs share no TIME range"])
        window = self.window(report)
        self.assertIn("share no", window.warning_label.text())
        self.assertTrue(window.warning_label.isVisible())

    def test_a_clean_run_has_no_warning_line(self):
        self.assertFalse(self.window().warning_label.isVisible())

    def test_a_long_line_wraps_rather_than_scrolling(self):
        """A run flagged on eighteen channels has a line thirty entries long,
        and the samples contain exactly that run."""
        window = self.window()
        label = window.tree.itemWidget(window.tree.topLevelItem(0),
                                       diff_report.COLUMN_RUN)
        self.assertTrue(label.word_wrap)

    def test_a_second_comparison_replaces_the_first(self):
        window = self.window()
        window.set_report(Report(0, 1, "File1", "File2", [a_flag("RUN9")]),
                          {}, 2)
        self.assertEqual(window.runs(), ["RUN9"])


class TestTheOverlayIsDrawn(unittest.TestCase):
    """FR-5: two series, named by file, with the bands behind them."""

    def tile(self, bands=None, run="RUN2", param="BETA",
             files=(0, 1), x="TIME"):
        tile = make_tile()
        tile._session = _Session()
        tile._fdiff_items = []
        tile._sync_report = (0, 0)
        state = tile._state
        state.diff_group_colors = {"BLACK": "#101010", "RED": "#dd2222"}
        shape = state.tile
        shape.x_param = x
        shape.fdiff_run = run
        shape.fdiff_param = param
        shape.fdiff_file_a, shape.fdiff_file_b = files
        shape.fdiff_bands = bands if bands is not None else [
            ("BLACK", 3.0, 5.0), ("RED", 2.0, 3.0)]
        tile._rebuild_file_diff(tile._plot_item, shape, [])
        return tile, shape

    def test_both_files_are_drawn(self):
        tile, _shape = self.tile()
        self.assertEqual(len(tile._curves), 2)

    def test_each_file_has_its_own_colour(self):
        tile, _shape = self.tile()
        colours = {tile._state.styles.color_for(entry.key)
                   for entry in tile._curves.values()}
        self.assertEqual(colours, {pt.FDIFF_COLOR_A, pt.FDIFF_COLOR_B})

    def test_the_legend_names_the_file_the_run_and_the_channel(self):
        """"File1 - RUN2 - BETA", as the specification writes it."""
        tile, _shape = self.tile()
        self.assertEqual(
            sorted(entry.label for entry in tile._curves.values()),
            ["File1 - RUN2 - BETA", "File2 - RUN2 - BETA"])

    def test_the_run_is_found_by_name_in_each_file(self):
        """RUN2 is run 1 of both of these; the point is that it is looked up
        rather than indexed, so a file holding its runs in another order
        still draws the right one."""
        tile, _shape = self.tile()
        self.assertEqual(tile._fdiff_run(0, "RUN1"), (0, 0))
        self.assertEqual(tile._fdiff_run(0, "RUN2"), (0, 1))
        self.assertIsNone(tile._fdiff_run(0, "RUN9"))

    def test_one_band_per_region(self):
        tile, _shape = self.tile()
        self.assertEqual(len(tile._fdiff_items), 2)

    def test_a_band_is_in_its_group_s_colour(self):
        tile, _shape = self.tile()
        values = [item.getRegion() for item in tile._fdiff_items]
        self.assertIn((3.0, 5.0), values)
        self.assertIn((2.0, 3.0), values)

    def test_a_band_cannot_be_dragged(self):
        """Unlike an AREA band this is a measurement, not an annotation."""
        tile, _shape = self.tile()
        for item in tile._fdiff_items:
            self.assertFalse(item.movable)

    def test_the_bands_sit_behind_the_curves(self):
        tile, _shape = self.tile()
        self.assertLess(pt.VIOLATION_Z, 0)

    def test_a_rebuild_does_not_stack_the_bands(self):
        tile, shape = self.tile()
        tile._rebuild_file_diff(tile._plot_item, shape, [])
        self.assertEqual(len(tile._fdiff_items), 2)

    def test_a_group_with_no_colour_is_still_drawn(self):
        tile, _shape = self.tile(bands=[("UNKNOWN", 1.0, 2.0)])
        self.assertEqual(len(tile._fdiff_items), 1)

    def test_a_single_sample_band_is_still_drawn(self):
        tile, _shape = self.tile(bands=[("BLACK", 4.0, 4.0)])
        self.assertEqual(len(tile._fdiff_items), 1)

    def test_a_channel_neither_file_has_draws_nothing(self):
        tile, _shape = self.tile(param="NOT_THERE")
        self.assertEqual(tile._curves, {})

    def test_a_verdict_is_not_left_over_an_empty_tile(self):
        """The files were closed, or reloaded from a build that renamed
        things: saved bands with nothing to judge are not drawn."""
        tile, _shape = self.tile(run="RUN_NOT_THERE")
        self.assertEqual(tile._curves, {})
        self.assertEqual(tile._fdiff_items, [])

    def test_a_file_that_is_gone_does_not_stop_the_other(self):
        tile, _shape = self.tile(files=(0, 7))
        self.assertEqual(len(tile._curves), 1)

    def test_it_draws_with_nothing_ticked_in_data(self):
        """The overlay names its run and its two files, and was generated to
        answer one question about them. Unticking a run in DATA afterwards --
        or opening the workspace on a machine where the ticks were saved
        narrower -- must not blank the page that answered it."""
        tile = make_drawable_tile(_Session())
        tile._sync_report = (0, 0)
        shape = tile._state.tile
        shape.x_param = "TIME"
        shape.fdiff_run = "RUN2"
        shape.fdiff_param = "BETA"
        shape.fdiff_file_a, shape.fdiff_file_b = 0, 1
        shape.fdiff_bands = [("BLACK", 3.0, 5.0)]
        self.assertEqual(tile._state.plotted_sims(), ())
        tile.rebuild()
        self.assertEqual(len(tile._curves), 2)
        self.assertEqual(len(tile._fdiff_items), 1)


if __name__ == "__main__":      # pragma: no cover
    unittest.main()
