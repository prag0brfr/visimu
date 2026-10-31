# Author: Paulo R. A. Gilz
"""The sample pair the comparison is tried on, and the window that runs it.

The samples are a feature, not a fixture: FR-7 asks for two files with the
same runs and at least one engineered violation to be there after every build,
and "the comparison has nothing to open" is exactly the state that makes the
feature untestable by hand. So what is engineered into them is asserted here,
in the places it was engineered.
"""

from __future__ import annotations

import io
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np

from visimu.diffconfig import group_names, parse_rules, ties   # noqa: E402
from visimu.model import AppState                              # noqa: E402
from visimu.filediff import (                                   # noqa: E402
    check_compatible,
    compare_files,
)

import gen_diff_samples                                        # noqa: E402


class _Reader:
    """Reads the generated files without the core: the samples are text, and
    this suite runs where the shared library may not."""

    def __init__(self, folder):
        self.files = [self._read(os.path.join(folder, name))
                      for name in ("regression_a.aoe", "regression_b.aoe")]

    @staticmethod
    def _read(path):
        names, runs, current = [], [], None
        with io.open(path, encoding="ascii") as handle:
            for line in handle:
                if line.startswith("\t") and "'" in line:
                    names.append(line.split("'")[1])
                elif line.startswith("*TABLEAU"):
                    current = (line.split(":")[1].strip(), [])
                    runs.append(current)
                elif line.strip() and current is not None \
                        and not line.startswith("*"):
                    current[1].append([float(v) for v in line.split()])
        return names, [(run_id, np.array(rows)) for run_id, rows in runs]

    def names(self, index):
        return self.files[index][0]

    def runs(self, index):
        return [(at, run_id)
                for at, (run_id, _rows) in enumerate(self.files[index][1])]

    def channel(self, file_index, run_index, name):
        names, runs = self.files[file_index]
        if name not in names:
            return np.array([])
        return runs[run_index][1][:, names.index(name)]


class TestTheSamplesAreGenerated(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="visimu-diff-samples-")
        gen_diff_samples.generate(self.folder, quiet=True)

    def config(self):
        with io.open(os.path.join(self.folder, "regression_thresholds.yaml"),
                     encoding="utf-8") as handle:
            return handle.read()

    def test_all_three_files_are_written(self):
        self.assertEqual(
            sorted(name for name in os.listdir(self.folder)),
            ["regression_a.aoe", "regression_b.aoe",
             "regression_thresholds.yaml"])

    def test_nothing_half_written_is_left_behind(self):
        self.assertEqual(
            [name for name in os.listdir(self.folder)
             if name.endswith(".partial")], [])

    def test_a_file_already_there_is_left_alone(self):
        """Someone who trimmed one down to try something should not find it
        replaced by the next build."""
        path = os.path.join(self.folder, "regression_a.aoe")
        with io.open(path, "w", encoding="ascii") as handle:
            handle.write("mine\n")
        self.assertEqual(gen_diff_samples.generate(self.folder, quiet=True), [])
        with io.open(path, encoding="ascii") as handle:
            self.assertEqual(handle.read(), "mine\n")

    def test_generating_twice_writes_nothing_the_second_time(self):
        self.assertEqual(gen_diff_samples.generate(self.folder, quiet=True), [])

    def test_both_files_hold_the_same_runs(self):
        reader = _Reader(self.folder)
        self.assertEqual(sorted(name for _at, name in reader.runs(0)),
                         sorted(name for _at, name in reader.runs(1)))

    def test_the_runs_are_in_a_different_order_in_the_two_files(self):
        """The files are matched by identifier, and a pair that happened to
        agree positionally would never show it."""
        reader = _Reader(self.folder)
        self.assertNotEqual([name for _at, name in reader.runs(0)],
                            [name for _at, name in reader.runs(1)])

    def test_both_files_hold_the_same_channels(self):
        reader = _Reader(self.folder)
        self.assertEqual(reader.names(0), reader.names(1))

    def test_the_two_files_are_sampled_differently(self):
        """So the comparison has to interpolate rather than subtract."""
        reader = _Reader(self.folder)
        a = reader.channel(0, 0, "TIME")
        b = reader.channel(1, 0, "TIME")
        self.assertNotEqual(a.size, b.size)

    def test_the_configuration_parses(self):
        rules = parse_rules(self.config())
        self.assertEqual(group_names(rules), ["BLACK", "AMBER"])
        self.assertEqual(ties(rules), [])

    def test_every_channel_has_a_threshold_in_both_groups(self):
        """A channel nobody wrote a threshold for is not compared, so a
        sample pair with channels missing from the configuration would be
        quietly comparing less than it looks like it is."""
        rules = parse_rules(self.config())
        for group in ("BLACK", "AMBER"):
            covered = {rule.param for rule in rules if rule.group == group}
            self.assertEqual(covered, set(gen_diff_samples.SIGNALS),
                             "{0} does not cover every channel".format(group))

    def test_alfa_keeps_the_specification_s_own_thresholds(self):
        rules = {(rule.group, rule.param): rule.threshold
                 for rule in parse_rules(self.config())}
        self.assertEqual(rules[("BLACK", "ALFA")], 25.0)
        self.assertEqual(rules[("AMBER", "ALFA")], 10.0)


class TestTheSamplesSayWhatTheyWereBuiltToSay(unittest.TestCase):
    """The engineered violations, read back through the real comparison.

    Against the plan in the generator rather than against numbers copied out
    of it: the plan is what somebody edits when they want the samples to say
    something else, and a test that agrees with it by coincidence is a test
    that will not notice.
    """

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.mkdtemp(prefix="visimu-diff-engine-")
        gen_diff_samples.generate(cls.folder, quiet=True)
        reader = _Reader(cls.folder)
        with io.open(os.path.join(cls.folder, "regression_thresholds.yaml"),
                     encoding="utf-8") as handle:
            rules = parse_rules(handle.read())
        check_compatible([n for _a, n in reader.runs(0)],
                         [n for _a, n in reader.runs(1)],
                         reader.names(0), reader.names(1))
        cls.report = compare_files(reader.channel, 0, 1, reader.runs(0),
                                   reader.runs(1), rules,
                                   "regression_a", "regression_b")

    def flags_of(self, run_id):
        return [flag for flag in self.report.flags if flag.run_id == run_id]

    def test_every_run_that_was_made_to_differ_is_flagged(self):
        expected = [run for run in gen_diff_samples.RUNS
                    if gen_diff_samples.flagged(run)]
        self.assertEqual([run for run, _flags in self.report.by_run()],
                         expected)

    def test_the_run_that_agrees_is_not_flagged(self):
        """A comparison that flags everything looks like one that works."""
        self.assertEqual(self.flags_of("RUN_002"), [])

    def test_each_run_is_flagged_on_exactly_its_own_channels(self):
        for run_id in gen_diff_samples.RUNS:
            self.assertEqual([flag.param for flag in self.flags_of(run_id)],
                             gen_diff_samples.flagged(run_id), run_id)

    def test_one_run_is_flagged_on_more_channels_than_a_page_holds(self):
        """The case the paging exists for, and it is in the samples so that
        it can be tried by hand rather than only asserted."""
        self.assertGreater(len(self.flags_of("RUN_001")),
                           AppState.DIFF_PAGE_ROWS * AppState.DIFF_PAGE_COLS)

    def test_the_runs_are_of_three_different_sizes(self):
        sizes = {len(self.flags_of(run)) for run in gen_diff_samples.RUNS
                 if gen_diff_samples.flagged(run)}
        self.assertEqual(len(sizes), 3, sizes)

    def test_a_window_with_one_group_reports_one(self):
        flag = self.by_shape("amber")
        self.assertEqual(flag.groups, ["AMBER"])

    def test_a_window_with_a_group_inside_it_reports_both(self):
        flag = self.by_shape("both")
        self.assertEqual(flag.groups, ["BLACK", "AMBER"])

    def test_the_higher_group_owns_the_middle_of_that_window(self):
        """The priority rule, in a file: AMBER either side, BLACK inside."""
        flag = self.by_shape("both")
        bands = {v.group: v.regions for v in flag.violations}
        self.assertEqual(len(bands["BLACK"]), 1)
        self.assertEqual(len(bands["AMBER"]), 2)
        inner_low, inner_high = bands["BLACK"][0]
        before, after = bands["AMBER"]
        self.assertLessEqual(before[1], inner_low)
        self.assertGreaterEqual(after[0], inner_high)

    def test_that_window_is_where_it_was_engineered(self):
        """Within one sample of the coarser file's step, which is as close as
        an interpolated crossing can be."""
        param, (_shape, start, end) = self.plan("both")
        flag = self.by_shape("both")
        bands = {v.group: v.regions for v in flag.violations}
        slack = 2.0 * gen_diff_samples.STEP_B
        self.assertAlmostEqual(bands["AMBER"][0][0], start, delta=slack)
        self.assertAlmostEqual(bands["AMBER"][1][1], end, delta=slack)
        self.assertAlmostEqual(bands["BLACK"][0][0],
                               start + gen_diff_samples.INNER_IN,
                               delta=slack)
        self.assertAlmostEqual(bands["BLACK"][0][1],
                               start + gen_diff_samples.INNER_OUT,
                               delta=slack)

    def test_a_deviation_above_both_leaves_the_window_to_the_higher(self):
        flag = self.by_shape("buried")
        bands = {v.group: v.regions for v in flag.violations}
        self.assertEqual(flag.groups, ["BLACK", "AMBER"])
        self.assertEqual(len(bands["BLACK"]), 1)
        covered = bands["BLACK"][0][1] - bands["BLACK"][0][0]
        lower = sum(high - low for low, high in bands["AMBER"])
        self.assertGreater(covered, 10.0 * max(lower, 0.01),
                           "the higher group should own nearly all of it")

    def test_nothing_could_not_be_compared(self):
        self.assertEqual(self.report.warnings, [])

    # -- finding the channel each shape was written on

    def plan(self, shape, run_id="RUN_001"):
        for (run, param), entry in sorted(gen_diff_samples.DEVIATIONS.items()):
            if run == run_id and entry[0] == shape:
                return param, entry
        raise AssertionError("no {0} window in {1}".format(shape, run_id))

    def by_shape(self, shape, run_id="RUN_001"):
        param, _entry = self.plan(shape, run_id)
        for flag in self.flags_of(run_id):
            if flag.param == param:
                return flag
        raise AssertionError("{0} was not flagged".format(param))


class TestPlottingARunThatFillsMoreThanOnePage(unittest.TestCase):
    """What the samples are for: plotting a run with eighteen flags.

    Driven by the real report over the real files, because the thing being
    checked is that every flagged channel gets a tile -- and "every" is a
    claim about the report, not about a list written here.
    """

    @classmethod
    def setUpClass(cls):
        cls.report = TestTheSamplesSayWhatTheyWereBuiltToSay.report \
            if hasattr(TestTheSamplesSayWhatTheyWereBuiltToSay, "report") \
            else None
        if cls.report is None:
            folder = tempfile.mkdtemp(prefix="visimu-diff-pages-")
            gen_diff_samples.generate(folder, quiet=True)
            reader = _Reader(folder)
            with io.open(os.path.join(folder,
                                      "regression_thresholds.yaml"),
                         encoding="utf-8") as handle:
                rules = parse_rules(handle.read())
            cls.report = compare_files(reader.channel, 0, 1, reader.runs(0),
                                       reader.runs(1), rules, "a", "b")

    def plotted(self, run_id="RUN_001", state=None):
        state = state or AppState()
        flags = [f for f in self.report.flags if f.run_id == run_id]
        return state, flags, state.build_diff_pages(run_id, flags, 0, 1)

    def test_every_flagged_channel_gets_a_tile(self):
        _state, flags, pages = self.plotted()
        tiles = [tile for page in pages for tile in page.tiles
                 if tile.is_file_diff()]
        self.assertEqual([tile.fdiff_param for tile in tiles],
                         [flag.param for flag in flags])

    def test_it_fills_a_page_before_starting_another(self):
        _state, _flags, pages = self.plotted()
        per_page = AppState.DIFF_PAGE_ROWS * AppState.DIFF_PAGE_COLS
        counts = [len([t for t in page.tiles if t.is_file_diff()])
                  for page in pages]
        self.assertEqual(counts[:-1], [per_page] * (len(pages) - 1))
        self.assertGreater(counts[-1], 0)

    def test_eighteen_flags_make_two_pages(self):
        _state, flags, pages = self.plotted()
        self.assertEqual(len(flags), 18)
        self.assertEqual([page.name for page in pages],
                         ["DIFF (1)", "DIFF (2)"])

    def test_the_tiles_are_in_the_order_the_report_lists_them(self):
        """Worst group first, so the first tile of the first page is the
        worst news."""
        _state, flags, pages = self.plotted()
        first = pages[0].tiles[0]
        self.assertEqual(first.fdiff_param, flags[0].param)

    def test_each_tile_carries_its_own_bands(self):
        _state, flags, pages = self.plotted()
        by_param = {flag.param: flag for flag in flags}
        for page in pages:
            for tile in page.tiles:
                if not tile.is_file_diff():
                    continue
                expected = [(v.group, low, high)
                            for v in by_param[tile.fdiff_param].violations
                            for low, high in v.regions]
                self.assertEqual(tile.fdiff_bands, expected,
                                 tile.fdiff_param)

    def test_no_tile_is_left_half_filled(self):
        _state, _flags, pages = self.plotted()
        for page in pages:
            for tile in page.tiles:
                if tile.is_file_diff():
                    self.assertTrue(tile.fdiff_run)
                    self.assertTrue(tile.x_param)
                    self.assertNotEqual(tile.fdiff_file_a, tile.fdiff_file_b)

    def test_plotting_a_second_run_keeps_the_first_run_s_pages(self):
        state, _flags, first = self.plotted()
        _state, _f, second = self.plotted("RUN_003", state)
        self.assertEqual([page.name for page in first + second],
                         ["DIFF (1)", "DIFF (2)", "DIFF (3)"])
        self.assertEqual(len(state.diff_pages()), 3)

    def test_the_first_generated_page_is_the_one_shown(self):
        state, _flags, pages = self.plotted()
        self.assertEqual(state.pages[state.active_page].name, pages[0].name)


class TestTheBuildGeneratesThem(unittest.TestCase):
    """FR-7.1 says "on every compile/build", so the hook is asserted rather
    than the behaviour: running a build in a test is not a test."""

    def body(self):
        path = os.path.join(ROOT, "packaging", "build_exe.py")
        with io.open(path, encoding="utf-8") as handle:
            return handle.read()

    def test_the_build_calls_it(self):
        self.assertIn("ensure_diff_samples()", self.body())

    def test_it_runs_before_the_big_samples(self):
        """Half a gigabyte of flight data takes minutes; a build interrupted
        during it should still have left the comparison its pair."""
        body = self.body()
        order = body.index("if not args.no_samples:")
        tail = body[order:order + 200]
        self.assertLess(tail.index("ensure_diff_samples()"),
                        tail.index("ensure_samples()"))

    def test_a_failure_does_not_fail_the_build(self):
        body = self.body()
        start = body.index("def ensure_diff_samples")
        end = body.index("\ndef ", start + 10)
        self.assertIn("except OSError", body[start:end])
        self.assertIn("except ImportError", body[start:end])


class TestTheWindowRunsIt(unittest.TestCase):
    """The order the window does things in, read off the source.

    MainWindow needs a running Qt application, so what is checked here is the
    sequence -- and the sequence is the requirement: validate, admit
    everything, compare, report. A comparison that ran before the filters were
    opened would generate pages that came out blank, which is the bug this
    guards.
    """

    def body(self):
        path = os.path.join(ROOT, "python", "visimu", "ui", "main_window.py")
        with io.open(path, encoding="utf-8") as handle:
            return handle.read()

    def method(self, name):
        body = self.body()
        start = body.index("    def {0}(self".format(name))
        end = body.index("\n    def ", start + 10)
        return body[start:end]

    def test_the_panel_is_wired_to_it(self):
        body = self.body()
        self.assertIn("run_file_diff.connect(self.run_file_diff)", body)
        self.assertIn("show_report.connect(self.show_diff_report)", body)

    def test_compatibility_is_checked_before_anything_is_read(self):
        source = self.method("run_file_diff")
        self.assertLess(source.index("check_compatible"),
                        source.index("compare_files"))

    def test_the_filters_are_opened_before_the_comparison(self):
        source = self.method("run_file_diff")
        self.assertLess(source.index("_admit_everything"),
                        source.index("compare_files"))

    def test_every_run_is_ticked_and_every_iso_value_admitted(self):
        source = self.method("_admit_everything")
        self.assertIn("set_selected", source)
        self.assertIn("value_enabled", source)
        self.assertIn("iso.groups", source)
        self.assertIn("_resolve_iso_filters", source)

    def test_a_refusal_stops_the_comparison(self):
        """Each refusal warns and returns: a non-regression check that
        half-ran is worse than one that did not run."""
        source = self.method("run_file_diff")
        warnings = len(re.findall(r"QMessageBox\.warning", source))
        self.assertGreaterEqual(warnings, 5)
        self.assertEqual(warnings, len(re.findall(r"\n            return",
                                                  source)))

    def test_the_report_opens_itself_after_a_clean_run(self):
        source = self.method("run_file_diff")
        self.assertIn("self.show_diff_report()", source)

    def test_the_previous_comparison_s_pages_do_not_survive(self):
        self.assertIn("clear_diff_pages", self.method("run_file_diff"))

    def test_the_report_window_is_reused_rather_than_stacked(self):
        source = self.method("show_diff_report")
        self.assertIn("_diff_report_window", source)
        self.assertIn("set_report", source)

    def test_the_grid_is_laid_out_again_after_plotting(self):
        """refresh_all() does not touch the grid. The generated pages are 4x4
        and whatever was on screen probably was not, so without this the
        sixteen tiles are drawn into the old shape and all but the first few
        are never seen."""
        source = self.method("plot_diff_run")
        self.assertIn("_on_pages_changed()", source)
        self.assertLess(source.index("build_diff_pages"),
                        source.index("_on_pages_changed()"))

    def test_the_grid_is_laid_out_again_after_a_comparison(self):
        """Clearing the previous comparison's pages can move the active one
        to a page of another shape."""
        self.assertIn("_on_pages_changed()", self.method("run_file_diff"))

    def test_plotting_happens_only_for_a_run_the_report_flagged(self):
        source = self.method("plot_diff_run")
        self.assertIn("build_diff_pages", source)
        self.assertIn("if not flags:", source)

    def test_nothing_else_calls_the_page_builder(self):
        """FR-4.9: no plots are generated automatically."""
        body = self.body()
        self.assertEqual(len(re.findall(r"build_diff_pages", body)), 1)


if __name__ == "__main__":      # pragma: no cover
    unittest.main()
