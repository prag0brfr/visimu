# Author: Paulo R. A. Gilz
"""The file-to-file comparison: the configuration, and the arithmetic.

Both are pure logic over arrays and strings, and both decide what a
non-regression report says -- so they are tested on their own, away from any
widget. The worked example from the specification is in here verbatim: it is
the one case whose answer is given rather than derived, which makes it the
case worth failing on.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from visimu.diffconfig import (          # noqa: E402
    ConfigError,
    Rule,
    group_names,
    parse_rules,
    ties,
)
from visimu.filediff import (            # noqa: E402
    DiffError,
    Report,
    aligned,
    check_compatible,
    compare_files,
    compare_run,
)

SPEC_EXAMPLE = '''
...
 - group_name: "BLACK"
 ...
      algoconfs:
      ...
        - variables: "ALFA"
        ...
          threshold: 25
          ...
        - variables: "Q1;TETA;BETA"
        ...
          x: "TIME"
          ...
          threshold: 10
          ...
 - group_name: "AMBER"
 ...
      algoconfs:
      ...
        - variables: "ALFA"
          threshold: 10
        - variables: "Q1;TETA;BETA"
          threshold: 5
...
'''


class TestTheConfiguration(unittest.TestCase):
    """The specification's own file, elisions and all."""

    def rules(self, text=SPEC_EXAMPLE):
        return parse_rules(text)

    def test_the_example_parses(self):
        rules = self.rules()
        self.assertEqual(
            [(r.group, r.param, r.threshold) for r in rules],
            [("BLACK", "ALFA", 25.0), ("BLACK", "Q1", 10.0),
             ("BLACK", "TETA", 10.0), ("BLACK", "BETA", 10.0),
             ("AMBER", "ALFA", 10.0), ("AMBER", "Q1", 5.0),
             ("AMBER", "TETA", 5.0), ("AMBER", "BETA", 5.0)])

    def test_a_semicolon_list_becomes_one_rule_each(self):
        rules = [r for r in self.rules() if r.group == "BLACK"]
        self.assertEqual([r.param for r in rules],
                         ["ALFA", "Q1", "TETA", "BETA"])
        self.assertEqual({r.threshold for r in rules[1:]}, {10.0})

    def test_x_defaults_to_time(self):
        for rule in self.rules():
            self.assertEqual(rule.x_param, "TIME")

    def test_an_explicit_x_is_kept(self):
        rules = parse_rules(
            '- group_name: "G"\n'
            '  algoconfs:\n'
            '    - variables: "A"\n'
            '      x: "DISTANCE"\n'
            '      threshold: 1\n')
        self.assertEqual(rules[0].x_param, "DISTANCE")

    def test_unknown_keys_are_ignored_at_any_level(self):
        rules = parse_rules(
            '- group_name: "G"\n'
            '  comment: "whatever this is"\n'
            '  owner: "someone"\n'
            '  algoconfs:\n'
            '    - variables: "A"\n'
            '      units: "deg"\n'
            '      threshold: 2\n'
            '      note: "ignored"\n')
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].threshold, 2.0)

    def test_comments_and_blank_lines_are_ignored(self):
        rules = parse_rules(
            '# the thresholds\n'
            '\n'
            '- group_name: "G"   # the group\n'
            '  algoconfs:\n'
            '    - variables: "A"\n'
            '      threshold: 3  # generous\n')
        self.assertEqual(rules[0].threshold, 3.0)

    def test_a_hash_inside_a_name_is_not_a_comment(self):
        rules = parse_rules(
            '- group_name: "G"\n'
            '  algoconfs:\n'
            '    - variables: "A#1"\n'
            '      threshold: 1\n')
        self.assertEqual(rules[0].param, "A#1")

    def test_the_last_definition_of_a_pair_wins(self):
        rules = parse_rules(
            '- group_name: "G"\n'
            '  algoconfs:\n'
            '    - variables: "A"\n'
            '      threshold: 1\n'
            '    - variables: "A"\n'
            '      threshold: 9\n')
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].threshold, 9.0)

    def test_a_float_threshold_is_a_float(self):
        rules = parse_rules(
            '- group_name: "G"\n'
            '  algoconfs:\n'
            '    - variables: "A"\n'
            '      threshold: 0.25\n')
        self.assertEqual(rules[0].threshold, 0.25)

    def test_the_groups_keep_the_order_they_appear_in(self):
        self.assertEqual(group_names(self.rules()), ["BLACK", "AMBER"])

    # ----------------------------------------------------------- refusals

    def test_an_empty_file_is_refused(self):
        with self.assertRaises(ConfigError):
            parse_rules("   \n\n# nothing\n")

    def test_a_file_with_no_groups_is_refused(self):
        with self.assertRaises(ConfigError):
            parse_rules("hello: world\n")

    def test_a_threshold_that_is_not_a_number_is_refused_with_its_line(self):
        with self.assertRaises(ConfigError) as caught:
            parse_rules(
                '- group_name: "G"\n'
                '  algoconfs:\n'
                '    - variables: "A"\n'
                '      threshold: "quite a lot"\n')
        self.assertIn("line 4", str(caught.exception))

    def test_an_entry_with_no_threshold_is_refused(self):
        with self.assertRaises(ConfigError):
            parse_rules(
                '- group_name: "G"\n'
                '  algoconfs:\n'
                '    - variables: "A"\n')

    def test_an_entry_with_no_variables_is_refused(self):
        with self.assertRaises(ConfigError):
            parse_rules(
                '- group_name: "G"\n'
                '  algoconfs:\n'
                '    - threshold: 1\n')

    def test_a_group_with_no_name_is_refused(self):
        with self.assertRaises(ConfigError):
            parse_rules('- group_name: ""\n  algoconfs:\n'
                        '    - variables: "A"\n      threshold: 1\n')

    # ------------------------------------------------------------ ties

    def test_two_groups_at_the_same_threshold_are_reported(self):
        """Priority is the number and nothing else, so equal numbers have no
        order between them. Reported rather than resolved: picking one
        silently would put a colour on the plot the file does not justify."""
        rules = parse_rules(
            '- group_name: "ONE"\n'
            '  algoconfs:\n'
            '    - variables: "A"\n'
            '      threshold: 5\n'
            '- group_name: "TWO"\n'
            '  algoconfs:\n'
            '    - variables: "A"\n'
            '      threshold: 5\n')
        self.assertEqual(ties(rules), [("A", ["ONE", "TWO"])])

    def test_different_thresholds_are_not_a_tie(self):
        self.assertEqual(ties(self.rules()), [])


class TestCompatibility(unittest.TestCase):
    """What must hold before anything is compared at all."""

    def test_matching_files_pass(self):
        check_compatible(["R1", "R2"], ["R2", "R1"], ["A"], ["A"])

    def test_a_different_run_count_is_refused(self):
        with self.assertRaises(DiffError) as caught:
            check_compatible(["R1"], ["R1", "R2"], ["A"], ["A"])
        self.assertIn("different numbers of runs", str(caught.exception))

    def test_names_that_do_not_pair_up_are_refused(self):
        with self.assertRaises(DiffError) as caught:
            check_compatible(["R1", "R2"], ["R1", "R3"], ["A"], ["A"])
        self.assertIn("do not match one to one", str(caught.exception))
        self.assertIn("R2", str(caught.exception))

    def test_a_repeated_identifier_is_refused(self):
        """Two runs called the same thing have no one-to-one match."""
        with self.assertRaises(DiffError) as caught:
            check_compatible(["R1", "R1"], ["R1", "R1"], ["A"], ["A"])
        self.assertIn("more than once", str(caught.exception))

    def test_a_different_parameter_set_is_refused(self):
        with self.assertRaises(DiffError) as caught:
            check_compatible(["R1"], ["R1"], ["A", "B"], ["A"])
        self.assertIn("different parameters", str(caught.exception))
        self.assertIn("B", str(caught.exception))

    def test_runs_are_matched_by_name_and_not_by_position(self):
        """Two files from two builds hold their runs in whatever order the
        solver finished them."""
        def reader(file_index, run_index, name):
            if name == "TIME":
                return np.arange(4.0)
            # Run 0 of the first file is run 1 of the second, and they agree.
            return np.zeros(4) if (file_index, run_index) in (
                (0, 0), (1, 1)) else np.full(4, 10.0)

        rules = [Rule("G", "A", "TIME", 1.0)]
        report = compare_files(reader, 0, 1, [(0, "EARLY"), (1, "LATE")],
                               [(0, "LATE"), (1, "EARLY")], rules)
        self.assertEqual(report.flags, [], report.line_for("EARLY"))


class TestAlignment(unittest.TestCase):
    """Two files sampled at different instants, compared anyway."""

    def test_identical_grids_are_compared_as_they_are(self):
        x = np.arange(5.0)
        got = aligned(x, x * 2, x, x * 3)
        self.assertIsNotNone(got)
        out_x, a, b = got
        self.assertTrue(np.allclose(out_x, x))
        self.assertTrue(np.allclose(b - a, x))

    def test_the_second_file_is_read_at_the_first_instants(self):
        x_a = np.array([0.0, 1.0, 2.0])
        x_b = np.array([0.0, 2.0])
        out_x, a, b = aligned(x_a, np.zeros(3), x_b, np.array([0.0, 20.0]))
        self.assertTrue(np.allclose(out_x, x_a))
        self.assertTrue(np.allclose(b, [0.0, 10.0, 20.0]),
                        "the second series was not interpolated")

    def test_only_the_shared_range_is_compared(self):
        """No extrapolation: a difference outside the overlap is one nobody
        measured."""
        x_a = np.arange(0.0, 10.0)
        x_b = np.arange(4.0, 7.0)
        out_x, _a, _b = aligned(x_a, x_a, x_b, x_b)
        self.assertEqual((out_x[0], out_x[-1]), (4.0, 6.0))

    def test_no_overlap_at_all_is_no_comparison(self):
        self.assertIsNone(aligned(np.arange(0.0, 3.0), np.zeros(3),
                                  np.arange(10.0, 13.0), np.zeros(3)))

    def test_an_unsorted_second_grid_is_sorted_first(self):
        x_b = np.array([2.0, 0.0, 1.0])
        y_b = np.array([20.0, 0.0, 10.0])
        out_x, _a, b = aligned(np.arange(3.0), np.zeros(3), x_b, y_b)
        self.assertTrue(np.allclose(b, [0.0, 10.0, 20.0]))

    def test_an_empty_series_is_no_comparison(self):
        self.assertIsNone(aligned(np.array([]), np.array([]),
                                  np.arange(3.0), np.zeros(3)))


class TestTheWorkedExample(unittest.TestCase):
    """The one case the specification gives the answer to.

    File1 and File2 hold RUN1 and RUN2 with ALFA and BETA. ALFA is identical
    everywhere. In RUN2, BETA exceeds RED (threshold 4) on [2, 4] and BLACK
    (threshold 6) on [3, 5]. BLACK's threshold is the higher number, so BLACK
    has priority, and the shading must read RED on [2, 3) and BLACK on [3, 5].
    """

    TIME = np.arange(0.0, 7.0, 1.0)

    def reader(self, file_index, run_index, name):
        if name == "TIME":
            return self.TIME
        if name == "ALFA":
            return np.zeros_like(self.TIME)
        if name == "BETA":
            if file_index == 0 or run_index != 1:
                return np.zeros_like(self.TIME)
            out = np.zeros_like(self.TIME)
            out[2] = 5.0                 # over 4, under 6
            out[3:6] = 7.0               # over both
            return out
        return np.array([])

    def report(self):
        rules = [Rule("RED", "ALFA", "TIME", 4.0),
                 Rule("BLACK", "ALFA", "TIME", 6.0),
                 Rule("RED", "BETA", "TIME", 4.0),
                 Rule("BLACK", "BETA", "TIME", 6.0)]
        return compare_files(self.reader, 0, 1,
                             [(0, "RUN1"), (1, "RUN2")],
                             [(0, "RUN1"), (1, "RUN2")],
                             rules, "File1", "File2")

    def test_only_the_run_with_a_difference_is_reported(self):
        report = self.report()
        self.assertEqual([run for run, _flags in report.by_run()], ["RUN2"])

    def test_an_identical_parameter_raises_nothing(self):
        self.assertEqual([flag.param for flag in self.report().flags], ["BETA"])

    def test_the_line_names_every_pair(self):
        self.assertEqual(self.report().line_for("RUN2"),
                         "RUN2: BETA(BLACK), BETA(RED)")

    def test_the_worse_group_comes_first(self):
        """Higher threshold is worse news, and is what the eye should meet
        first."""
        flag = self.report().flags[0]
        self.assertEqual(flag.groups, ["BLACK", "RED"])

    def test_the_shading_is_the_specification_s(self):
        flag = self.report().flags[0]
        shading = {v.group: v.regions for v in flag.violations}
        self.assertEqual(shading["RED"], [(2.0, 3.0)])
        self.assertEqual(shading["BLACK"], [(3.0, 5.0)])

    def test_the_lower_group_is_not_drawn_under_the_higher_one(self):
        """Only the highest-priority violated group is shaded at a sample."""
        flag = self.report().flags[0]
        red = [v for v in flag.violations if v.group == "RED"][0]
        for low, high in red.regions:
            self.assertLessEqual(high, 3.0)

    def test_nothing_is_warned_about(self):
        self.assertEqual(self.report().warnings, [])


class TestViolationDetail(unittest.TestCase):
    """The rules at the edges of the arithmetic."""

    TIME = np.arange(0.0, 5.0)

    def run_one(self, a, b, thresholds=((("G", 1.0)),)):
        def reader(file_index, run_index, name):
            if name == "TIME":
                return self.TIME
            return np.asarray(a if file_index == 0 else b, dtype=float)

        rules = [Rule(group, "A", "TIME", value) for group, value in thresholds]
        return compare_run(reader, 0, 1, 0, 0, "A", rules)

    def test_equal_to_the_threshold_is_not_a_violation(self):
        """abs(A - B) > threshold, strictly."""
        flag, _warnings = self.run_one(np.zeros(5), np.full(5, 1.0))
        self.assertIsNone(flag)

    def test_just_over_the_threshold_is_a_violation(self):
        flag, _warnings = self.run_one(np.zeros(5), np.full(5, 1.0001))
        self.assertIsNotNone(flag)

    def test_the_sign_of_the_difference_does_not_matter(self):
        flag, _warnings = self.run_one(np.zeros(5), np.full(5, -2.0))
        self.assertIsNotNone(flag)

    def test_a_lone_sample_is_still_a_region(self):
        values = np.zeros(5)
        values[2] = 9.0
        flag, _warnings = self.run_one(np.zeros(5), values)
        self.assertEqual(flag.violations[0].regions, [(2.0, 2.0)])

    def test_two_separate_stretches_stay_separate(self):
        values = np.zeros(9)
        values[1] = 9.0
        values[5:7] = 9.0
        def reader(file_index, run_index, name):
            if name == "TIME":
                return np.arange(9.0)
            return np.zeros(9) if file_index == 0 else values
        flag, _warnings = compare_run(reader, 0, 1, 0, 0, "A",
                                      [Rule("G", "A", "TIME", 1.0)])
        self.assertEqual(flag.violations[0].regions, [(1.0, 1.0), (5.0, 6.0)])

    def test_a_group_outranked_everywhere_is_still_reported(self):
        """FR-3.3 is about violations, not about bands.

        A deviation large enough to cross the higher threshold crosses the
        lower one too, at every sample. The higher group owns all of them, so
        the lower has nothing to shade -- but it *was* exceeded, and a report
        that said otherwise would be hiding it.
        """
        flag, _warnings = self.run_one(
            np.zeros(5), np.full(5, 20.0),
            thresholds=(("LOW", 1.0), ("HIGH", 10.0)))
        self.assertEqual(flag.groups, ["HIGH", "LOW"])
        bands = {v.group: v.regions for v in flag.violations}
        self.assertEqual(bands["HIGH"], [(0.0, 4.0)])
        self.assertEqual(bands["LOW"], [],
                         "the lower group shaded over the higher one")

    def test_a_group_that_was_never_exceeded_is_not_reported(self):
        flag, _warnings = self.run_one(
            np.zeros(5), np.full(5, 2.0),
            thresholds=(("LOW", 1.0), ("HIGH", 10.0)))
        self.assertEqual(flag.groups, ["LOW"])

    def test_a_nan_sample_is_not_a_difference(self):
        """One file having no number there is not a difference anyone
        measured."""
        values = np.zeros(5)
        values[2] = np.nan
        flag, _warnings = self.run_one(np.zeros(5), values)
        self.assertIsNone(flag)

    def test_a_missing_x_channel_is_reported_and_nothing_is_flagged(self):
        def reader(file_index, run_index, name):
            if name == "TIME":
                return np.array([])
            return np.zeros(5)
        flag, warnings = compare_run(reader, 0, 1, 0, 0, "A",
                                     [Rule("G", "A", "TIME", 1.0)])
        self.assertIsNone(flag)
        self.assertTrue(warnings)
        self.assertIn("TIME", warnings[0])

    def test_no_shared_range_is_reported_and_nothing_is_flagged(self):
        def reader(file_index, run_index, name):
            if name == "TIME":
                return np.arange(0.0, 3.0) if file_index == 0 \
                    else np.arange(50.0, 53.0)
            return np.zeros(3) if file_index == 0 else np.full(3, 99.0)
        flag, warnings = compare_run(reader, 0, 1, 0, 0, "A",
                                     [Rule("G", "A", "TIME", 1.0)])
        self.assertIsNone(flag)
        self.assertTrue(any("share no" in note for note in warnings), warnings)

    def test_a_read_that_raises_is_a_warning_and_not_a_crash(self):
        def reader(file_index, run_index, name):
            raise RuntimeError("the column is not there")
        flag, warnings = compare_run(reader, 0, 1, 0, 0, "A",
                                     [Rule("G", "A", "TIME", 1.0)])
        self.assertIsNone(flag)
        self.assertTrue(warnings)

    def test_only_the_parameters_the_rules_name_are_compared(self):
        """A rule set is a statement about which channels matter; comparing
        the other four hundred would bury the report."""
        read = []

        def reader(file_index, run_index, name):
            read.append(name)
            if name == "TIME":
                return np.arange(3.0)
            return np.zeros(3)

        compare_files(reader, 0, 1, [(0, "R")], [(0, "R")],
                      [Rule("G", "A", "TIME", 1.0)])
        self.assertEqual(set(read), {"TIME", "A"})


class TestTheReportSurvivesAWorkspace(unittest.TestCase):
    def test_it_round_trips(self):
        def reader(file_index, run_index, name):
            if name == "TIME":
                return np.arange(4.0)
            return np.zeros(4) if file_index == 0 else np.full(4, 9.0)

        original = compare_files(reader, 0, 1, [(0, "R1")], [(0, "R1")],
                                 [Rule("G", "A", "TIME", 1.0)], "a", "b")
        copy = Report.from_dict(original.to_dict())

        self.assertEqual(copy.line_for("R1"), original.line_for("R1"))
        self.assertEqual(copy.flags[0].violations[0].regions,
                         original.flags[0].violations[0].regions)
        self.assertEqual((copy.name_a, copy.name_b), ("a", "b"))

    def test_an_empty_report_round_trips(self):
        copy = Report.from_dict(Report(0, 1, "a", "b", []).to_dict())
        self.assertEqual(copy.flags, [])
        self.assertEqual(copy.by_run(), [])


if __name__ == "__main__":      # pragma: no cover
    unittest.main()
