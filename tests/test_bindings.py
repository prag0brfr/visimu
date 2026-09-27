# Author: Paulo R. A. Gilz
"""Tests for the Python bindings and the core behind them.

Uses stdlib unittest rather than pytest on purpose: pytest is one more package
to obtain through the offline wheel process, and these tests must be runnable on
the target machine with nothing installed.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "packaging"))

import numpy as np

from aoeviewer import CoreError, Session
from gen_sample import generate


class SampleFileMixin:
    """Creates a sample AOE file once for the whole test class."""

    sims = 4
    rows = 500
    wrap = False

    @classmethod
    def setUpClass(cls):
        handle, cls.path = tempfile.mkstemp(suffix=".aoe", prefix="aoe_test_")
        os.close(handle)
        generate(cls.path, cls.sims, cls.rows, cls.wrap)

    @classmethod
    def tearDownClass(cls):
        try:
            os.remove(cls.path)
        except OSError:
            pass

    def setUp(self):
        self.session = Session()
        self.file = self.session.add_file(self.path)

    def tearDown(self):
        self.session.close()


class TestLoading(SampleFileMixin, unittest.TestCase):
    def test_file_metadata(self):
        self.assertEqual(self.session.file_count, 1)
        self.assertEqual(self.file.simulation_count, self.sims)
        self.assertEqual(
            self.file.headers, ["TIME", "ALPHA", "MACH", "NZ", "ALT", "THETA"]
        )
        self.assertEqual(self.file.simulations[0].sim_id, "RUN_001")

    def test_missing_file_raises(self):
        with self.assertRaises(CoreError):
            self.session.add_file("/definitely/not/here.aoe")

    def test_duplicate_names_are_disambiguated(self):
        second = self.session.add_file(self.path)
        # Both files have the same basename; the second must be distinguishable
        # or legends and simulation keys collide.
        self.assertNotEqual(self.file.name, second.name)

    def test_formulas_and_aliases_are_not_axis_candidates_on_their_own(self):
        """Swapping one study for another must not move an x axis.

        Removing a file leaves its formulas and aliases defined -- on purpose,
        so that closing a study and opening the next one does not throw away
        the work set up for it. `parameters` therefore comes back non-empty
        with nothing open at all, and the code that repairs tiles naming
        absent channels did not recognise a closed session: it "repaired"
        every tile by pointing its x axis at whichever alias sorted first, and
        the study opened next found that name perfectly valid and left it
        there. Remove sample1, add sample2, and every tile had lost TIME.
        """
        self.session.set_computed("CLIMB", "deriv(ALT, TIME)")
        self.session.set_alias("Angle of attack", "ALPHA")
        self.assertEqual(self.session.plottable_parameters,
                         self.session.parameters)
        self.assertIn("TIME", self.session.plottable_parameters)

        self.session.remove_file(0)
        # Still defined, and still offered wherever a name may be typed.
        self.assertEqual(sorted(self.session.parameters),
                         ["Angle of attack", "CLIMB"])
        # But nothing is open, so nothing is a candidate for an axis.
        self.assertEqual(self.session.plottable_parameters, [])

        self.session.add_file(self.path)
        self.assertIn("TIME", self.session.plottable_parameters)
        self.assertIn("CLIMB", self.session.plottable_parameters)

    def test_swapping_studies_keeps_the_x_axis(self):
        """The reported sequence, end to end."""
        from aoeviewer.model import AppState

        self.session.set_alias("Angle of attack", "ALPHA")
        state = AppState()
        state.tile.x_param = "TIME"
        state.tile.y_params = ["ALT"]

        for step in ("remove", "add"):
            if step == "remove":
                self.session.remove_file(0)
            else:
                self.session.add_file(self.path)
            available = self.session.plottable_parameters
            state.resolve_missing_parameters(available)
            state.apply_parameter_defaults(available)
            self.assertEqual(state.tile.x_param, "TIME", step)

    def test_reloading_keeps_the_keys_a_rename_did_not_change(self):
        """Renaming is a label; reloading re-reads the file. Neither is an
        identity change, so a run must come out of both with the key it went
        in with.

        A run's key is built from the file's name, and reload() re-opens the
        dataset and rebuilds every key -- so while that name was the display
        name, renaming a file and then reloading it renamed every run's key
        with it, and every colour, dash and marker the user had chosen came
        back attached to nothing. The key is built from the name the file was
        first given instead, which renaming does not touch.
        """
        before = [sim.key for sim in self.file.simulations]

        self.file.rename("Nominal climb")
        self.assertEqual([sim.key for sim in self.file.simulations], before,
                         "renaming must not move a run's identity")

        self.assertTrue(self.file.reload())
        self.assertEqual(self.file.name, "Nominal climb")
        self.assertEqual([sim.key for sim in self.file.simulations], before,
                         "reloading must not move a run's identity either")

    def test_reloading_re_reads_the_file(self):
        self.assertTrue(self.file.reload())
        self.assertEqual(self.file.simulation_count, self.sims)
        self.assertEqual(self.file.simulations[0].sim_id, "RUN_001")
        self.assertEqual(
            self.session.channel(0, 0, "TIME").size, self.rows)

    def test_channel_is_a_zero_copy_view(self):
        data = self.session.channel(0, 0, "TIME")
        self.assertEqual(data.dtype, np.float32)
        self.assertEqual(data.size, self.rows)
        # Read-only: writing through it would corrupt the core's cache.
        self.assertFalse(data.flags.writeable)
        self.assertFalse(data.flags.owndata)

    def test_unknown_channel_returns_empty(self):
        self.assertEqual(self.session.channel(0, 0, "NOPE").size, 0)

    def test_out_of_range_indices_are_safe(self):
        self.assertEqual(self.session.channel(99, 0, "TIME").size, 0)
        self.assertEqual(self.session.channel(0, 99, "TIME").size, 0)
        self.assertIsNone(self.session.file(99))

    def test_values_are_correct(self):
        time = self.session.channel(0, 0, "TIME")
        self.assertAlmostEqual(float(time[0]), 0.0, places=6)
        self.assertAlmostEqual(float(time[1]), 0.01, places=5)
        mach = self.session.channel(0, 1, "MACH")
        self.assertAlmostEqual(float(mach[0]), 0.63, places=5)

    def test_remove_file(self):
        self.session.remove_file(0)
        self.assertEqual(self.session.file_count, 0)
        self.assertEqual(self.session.parameters, [])


class TestWrappedRecords(SampleFileMixin, unittest.TestCase):
    wrap = True
    rows = 300

    def test_records_split_over_two_lines(self):
        # The column counter has to carry across the newline; if it resets per
        # line every channel comes out interleaved with its neighbour.
        time = self.session.channel(0, 0, "TIME")
        theta = self.session.channel(0, 0, "THETA")
        self.assertEqual(time.size, self.rows)
        self.assertEqual(theta.size, self.rows)
        self.assertAlmostEqual(float(time[1]), 0.01, places=5)


class TestFormulas(SampleFileMixin, unittest.TestCase):
    def test_simple_expression(self):
        self.session.set_computed("G_LOAD", "NZ * 9.80665")
        nz = np.array(self.session.channel(0, 0, "NZ"))
        g = self.session.channel(0, 0, "G_LOAD")
        self.assertEqual(g.size, nz.size)
        np.testing.assert_allclose(g, nz * 9.80665, rtol=1e-5)

    def test_lookup_syntax(self):
        self.session.set_computed("DRIFT", "ALPHA - ALPHA[TIME=0.0]")
        drift = self.session.channel(0, 0, "DRIFT")
        self.assertAlmostEqual(float(drift[0]), 0.0, places=5)

    def test_functions_and_precedence(self):
        cases = {
            "2 + 3 * 4": 14.0,
            "(2 + 3) * 4": 20.0,
            "2 ^ 3 ^ 2": 512.0,     # right-associative
            "-2 ^ 2": -4.0,          # unary minus binds looser than power
            "max(3, 7)": 7.0,
            "clamp(15, 0, 10)": 10.0,
            "if(1 > 2, 5, 9)": 9.0,
            "abs(-4) + sqrt(9)": 7.0,
            "10 % 3": 1.0,
        }
        for expression, expected in cases.items():
            name = "T{0}".format(abs(hash(expression)) % 100000)
            self.session.set_computed(name, expression)
            value = self.session.channel(0, 0, name)
            self.assertGreater(value.size, 0, expression)
            self.assertAlmostEqual(float(value[0]), expected, places=4, msg=expression)

    def test_division_by_zero_is_nan_not_inf(self):
        # An infinity would silently blow up every plot's y-range.
        self.session.set_computed("DZ", "1 / 0")
        self.assertTrue(math.isnan(float(self.session.channel(0, 0, "DZ")[0])))

    def test_invalid_expressions_are_rejected(self):
        for bad in ["NZ *", "(1 + 2", "nosuchfn(1)", "UNKNOWN_PARAM + 1", ""]:
            with self.assertRaises(CoreError, msg=bad):
                self.session.set_computed("BAD", bad)

    def test_validate_does_not_define(self):
        ok, message = self.session.validate_formula("NZ * 2")
        self.assertTrue(ok)
        self.assertEqual(message, "")
        self.assertNotIn("NZ * 2", self.session.parameters)

        ok, message = self.session.validate_formula("NZ *")
        self.assertFalse(ok)
        self.assertTrue(message)

    def test_cycles_are_refused(self):
        with self.assertRaises(CoreError):
            self.session.set_computed("SELF", "SELF + 1")

        self.session.set_computed("A", "NZ + 1")
        self.session.set_computed("B", "A * 2")
        with self.assertRaises(CoreError):
            self.session.set_computed("A", "B + 1")

    def test_removal_cascades_to_dependants(self):
        self.session.set_computed("BASE", "NZ + 1")
        self.session.set_computed("DERIVED", "BASE * 2")
        removed = self.session.remove_computed("BASE")
        self.assertIn("BASE", removed)
        self.assertIn("DERIVED", removed)
        self.assertFalse(self.session.is_computed("DERIVED"))

    def test_computed_appears_in_parameters(self):
        self.session.set_computed("EXTRA", "NZ * 2")
        self.assertIn("EXTRA", self.session.parameters)
        self.assertEqual(self.session.computed_source("EXTRA"), "NZ * 2")


class TestDecimation(SampleFileMixin, unittest.TestCase):
    rows = 20000

    def test_reduces_point_count(self):
        x = self.session.channel(0, 0, "TIME")
        y = self.session.channel(0, 0, "ALPHA")
        dx, dy = self.session.decimate(x, y, 400)
        self.assertLessEqual(dx.size, 800)
        self.assertEqual(dx.size, dy.size)
        self.assertGreater(dx.size, 0)

    def test_x_is_monotonic(self):
        x = self.session.channel(0, 0, "TIME")
        y = self.session.channel(0, 0, "ALPHA")
        dx, _ = self.session.decimate(x, y, 300)
        self.assertTrue(np.all(np.diff(dx) >= 0))

    def test_spikes_survive(self):
        # The whole reason for min/max instead of stride sampling.
        x = np.arange(50000, dtype=np.float32)
        y = np.zeros(50000, dtype=np.float32)
        y[31337] = 500.0
        dx, dy = self.session.decimate(x, y, 200)
        self.assertAlmostEqual(float(dy.max()), 500.0, places=3)

    def test_short_series_passes_through(self):
        x = np.arange(40, dtype=np.float32)
        y = x * 2
        dx, dy = self.session.decimate(x, y, 500)
        self.assertEqual(dx.size, 40)
        np.testing.assert_allclose(dy, y)

    def test_range_restriction(self):
        x = self.session.channel(0, 0, "TIME")
        y = self.session.channel(0, 0, "ALPHA")
        dx, _ = self.session.decimate(x, y, 200, x_range=(50.0, 60.0))
        self.assertGreater(dx.size, 0)
        self.assertGreaterEqual(float(dx.min()), 45.0)
        self.assertLessEqual(float(dx.max()), 65.0)

    def test_empty_input(self):
        empty = np.empty(0, dtype=np.float32)
        dx, dy = self.session.decimate(empty, empty, 100)
        self.assertEqual(dx.size, 0)
        self.assertEqual(dy.size, 0)

    def test_bounds_and_nearest(self):
        x = self.session.channel(0, 0, "TIME")
        low, high = Session.bounds(x)
        self.assertAlmostEqual(low, 0.0, places=5)
        index = Session.nearest_index(x, 10.0)
        self.assertAlmostEqual(float(x[index]), 10.0, places=3)

    def test_bounds_ignores_nan(self):
        values = np.array([np.nan, 1.0, np.nan, 5.0], dtype=np.float32)
        self.assertEqual(Session.bounds(values), (1.0, 5.0))
        self.assertIsNone(Session.bounds(np.array([np.nan], dtype=np.float32)))


class TestIso(SampleFileMixin, unittest.TestCase):
    def test_distinct_values(self):
        values = self.session.iso_distinct("MACH")
        self.assertEqual(values.size, self.sims)
        self.assertTrue(np.all(np.diff(values) > 0))

    def test_per_simulation_values(self):
        values = self.session.iso_values(0, "MACH")
        self.assertEqual(values.size, self.sims)
        self.assertAlmostEqual(float(values[0]), 0.60, places=5)

    def test_auto_grouping_covers_extremes(self):
        values = self.session.iso_distinct("MACH")
        groups = self.session.iso_auto_group(values, 2)
        self.assertEqual(len(groups), 2)
        # Both ends must land in a group; the maximum is what a half-open final
        # interval silently loses.
        self.assertGreaterEqual(self.session.iso_group_index(float(values[0]), groups), 0)
        self.assertGreaterEqual(self.session.iso_group_index(float(values[-1]), groups), 0)

    def test_format_trims_trailing_zeros(self):
        # Two decimals, trailing zeros trimmed, and a power of ten at either
        # end of the range where two decimals stop saying anything. These are
        # labels on a tree row and inside a plot legend, so every character is
        # one the data does not get. The rules are in core/include/aoe/iso.hpp.
        self.assertEqual(self.session.format_value(0.75), "0.75")
        self.assertEqual(self.session.format_value(3.0), "3")
        self.assertEqual(self.session.format_value(0.756), "0.76")
        self.assertEqual(self.session.format_value(0.01), "0.01")
        self.assertEqual(self.session.format_value(0.0012345), "1.23e-3")
        self.assertEqual(self.session.format_value(1234567.0), "1.23e6")
        self.assertEqual(self.session.format_value(-0.001), "-1e-3")
        self.assertEqual(self.session.format_value(0.0), "0")


class TestPreload(SampleFileMixin, unittest.TestCase):
    rows = 5000

    def test_preload_makes_channels_available(self):
        self.session.preload(0, [], ["TIME", "ALPHA", "NZ"])
        for sim in range(self.sims):
            self.assertGreater(self.session.channel(0, sim, "ALPHA").size, 0)

    def test_preload_with_unknown_names_is_harmless(self):
        self.session.preload(0, [0], ["TIME", "NOT_A_CHANNEL"])
        self.assertGreater(self.session.channel(0, 0, "TIME").size, 0)


class TestModel(unittest.TestCase):
    """The state model, which has no dependency on Qt or the core."""

    def test_workspace_round_trip(self):
        from aoeviewer.model import AppState, load_workspace, save_workspace

        state = AppState()
        state.page.rows, state.page.cols = 2, 2
        state.tile.x_param = "TIME"
        state.tile.y_params = ["ALPHA", "NZ"]
        state.set_selected(0, 0, True)
        state.set_selected(0, 2, True)
        state.add_page()

        handle, path = tempfile.mkstemp(suffix=".aoews")
        os.close(handle)
        try:
            save_workspace(path, state, ["/tmp/a.dat"], {"G": "NZ*9.81"}, {})
            data = load_workspace(path)

            restored = AppState()
            restored.load_dict(data)

            self.assertEqual(len(restored.pages), 2)
            self.assertEqual(restored.pages[0].rows, 2)
            self.assertEqual(restored.pages[0].tiles[0].y_params, ["ALPHA", "NZ"])
            self.assertEqual(restored.selected_sims, {(0, 0), (0, 2)})
            self.assertEqual(data["files"], ["/tmp/a.dat"])
            self.assertEqual(data["computed"], {"G": "NZ*9.81"})
        finally:
            os.remove(path)

    def test_drop_file_remaps_selection(self):
        from aoeviewer.model import AppState

        state = AppState()
        state.set_selected(0, 1, True)
        state.set_selected(1, 0, True)
        state.set_selected(2, 3, True)

        state.drop_file(1)

        # File 1's selections go; file 2 shifts down to 1.
        self.assertEqual(state.selected_sims, {(0, 1), (1, 3)})

    def test_styles_are_sticky(self):
        from aoeviewer.model import StyleRegistry

        styles = StyleRegistry()
        first = styles.color_for("sim::a")
        styles.color_for("sim::b")
        # Asking again must give the same colour, or curves change colour when
        # an unrelated file is loaded.
        self.assertEqual(styles.color_for("sim::a"), first)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestTileTitle(unittest.TestCase):
    """The 'PARAM = f(X)' heading each tile shows."""

    def _tile(self, x="", ys=()):
        from aoeviewer.model import TileState
        t = TileState()
        t.x_param = x
        t.y_params = list(ys)
        return t

    def test_single_parameter(self):
        from aoeviewer.model import tile_title
        self.assertEqual(tile_title(self._tile("TIME", ["ALPHA"])), "ALPHA = f(TIME)")

    def test_several_parameters(self):
        from aoeviewer.model import tile_title
        self.assertEqual(
            tile_title(self._tile("TIME", ["ALPHA", "NZ"])), "ALPHA, NZ = f(TIME)"
        )

    def test_long_lists_wrap_instead_of_truncating(self):
        from aoeviewer.model import tile_title
        title = tile_title(self._tile("TIME", ["A", "B", "C", "D", "E", "F"]))
        self.assertIn("\n", title)
        self.assertIn("A", title)
        self.assertIn("F", title)
        self.assertIn("f(TIME)", title)
        self.assertNotIn("more", title)

    def test_nothing_plotted_reads_sensibly(self):
        from aoeviewer.model import tile_title
        # Never "= f()", which looks like a bug rather than an empty tile.
        self.assertEqual(tile_title(self._tile()), "empty tile")
        self.assertNotIn("= f()", tile_title(self._tile("TIME", [])))
        self.assertIn("TIME", tile_title(self._tile("TIME", [])))
        self.assertIn("no x axis", tile_title(self._tile("", ["ALPHA"])))

    def test_y_axis_label(self):
        from aoeviewer.model import y_axis_label
        self.assertEqual(y_axis_label(self._tile("TIME", [])), "")
        self.assertEqual(y_axis_label(self._tile("TIME", ["A", "B"])), "A, B")
        self.assertTrue(y_axis_label(self._tile("TIME", ["A", "B", "C", "D"])).endswith("..."))


class TestTileArrangement(unittest.TestCase):
    """Moving, copying and clearing tiles."""

    def _state(self, rows=2, cols=3):
        from aoeviewer.model import AppState
        state = AppState()
        state.page.rows, state.page.cols = rows, cols
        for i in range(state.page.tile_count):
            state.page.tiles[i].x_param = "TIME"
            state.page.tiles[i].y_params = ["P{0}".format(i)]
        return state

    def test_move_right_swaps(self):
        state = self._state()
        state.active_tile = 0
        self.assertTrue(state.move_tile("right"))
        self.assertEqual(state.active_tile, 1)
        # The tile travelled with the selection, and its neighbour came back.
        self.assertEqual(state.page.tiles[1].y_params, ["P0"])
        self.assertEqual(state.page.tiles[0].y_params, ["P1"])

    def test_move_is_its_own_inverse(self):
        state = self._state()
        before = [t.y_params[0] for t in state.page.visible_tiles()]
        state.active_tile = 1
        state.move_tile("right")
        state.move_tile("left")
        after = [t.y_params[0] for t in state.page.visible_tiles()]
        self.assertEqual(before, after)

    def test_edges_refuse(self):
        state = self._state(rows=2, cols=3)
        state.active_tile = 0
        self.assertFalse(state.move_tile("left"))
        self.assertFalse(state.move_tile("up"))
        state.active_tile = 5           # bottom-right of a 2x3
        self.assertFalse(state.move_tile("right"))
        self.assertFalse(state.move_tile("down"))

    def test_vertical_move_uses_column_count(self):
        state = self._state(rows=2, cols=3)
        state.active_tile = 0
        self.assertTrue(state.move_tile("down"))
        self.assertEqual(state.active_tile, 3)

    def test_copy_paste_is_a_deep_copy(self):
        state = self._state()
        state.active_tile = 0
        state.copy_tile()
        state.active_tile = 2
        self.assertTrue(state.paste_tile())
        self.assertEqual(state.page.tiles[2].y_params, ["P0"])

        # Mutating the source must not reach through to the pasted copy.
        state.page.tiles[0].y_params.append("EXTRA")
        self.assertEqual(state.page.tiles[2].y_params, ["P0"])

    def test_cut_clears_the_source(self):
        state = self._state()
        state.active_tile = 1
        state.cut_tile()
        self.assertTrue(state.page.tiles[1].is_empty())
        state.active_tile = 4
        state.paste_tile()
        self.assertEqual(state.page.tiles[4].y_params, ["P1"])

    def test_paste_without_clipboard_does_nothing(self):
        state = self._state()
        self.assertFalse(state.paste_tile())


class TestPages(unittest.TestCase):
    def test_rename(self):
        from aoeviewer.model import AppState
        state = AppState()
        state.rename_page(0, "  Speed Plots  ")
        self.assertEqual(state.pages[0].name, "Speed Plots")
        # A blank name is ignored rather than leaving an unnamed page.
        state.rename_page(0, "   ")
        self.assertEqual(state.pages[0].name, "Speed Plots")

    def test_reorder(self):
        from aoeviewer.model import AppState
        state = AppState()
        state.rename_page(0, "First")
        state.add_page()
        state.rename_page(1, "Second")

        new_index = state.move_page(0, 1)
        self.assertEqual(new_index, 1)
        self.assertEqual([p.name for p in state.pages], ["Second", "First"])

        # Out-of-range moves are no-ops, not exceptions.
        self.assertEqual(state.move_page(0, -1), 0)
        self.assertEqual(state.move_page(1, 5), 1)

    def test_name_survives_a_workspace_round_trip(self):
        from aoeviewer.model import AppState, load_workspace, save_workspace
        state = AppState()
        state.rename_page(0, "Engine parameters")
        state.tile.legend_position = "bottom-right"

        handle, path = tempfile.mkstemp(suffix=".aoews")
        os.close(handle)
        try:
            save_workspace(path, state, [], {}, {})
            restored = AppState()
            restored.load_dict(load_workspace(path))
            self.assertEqual(restored.pages[0].name, "Engine parameters")
            self.assertEqual(restored.pages[0].tiles[0].legend_position, "bottom-right")
        finally:
            os.remove(path)


class TestStyleRegistry(unittest.TestCase):
    def test_keys_are_namespaced_per_mode(self):
        from aoeviewer.model import StyleRegistry
        styles = StyleRegistry()
        # A simulation and a parameter of the same name must not share a colour.
        a = styles.color_for("sim::run1")
        b = styles.color_for("param::run1")
        self.assertNotEqual(a, b)

    def test_dash_symbol_and_width(self):
        from aoeviewer.model import StyleRegistry
        styles = StyleRegistry()
        styles.set_dash("sim::a", "dash")
        styles.set_symbol("sim::a", "o")
        styles.set_width("sim::a", 3.0)
        self.assertEqual(styles.dash_for("sim::a"), "dash")
        self.assertEqual(styles.symbol_for("sim::a"), "o")
        self.assertEqual(styles.width_for("sim::a"), 3.0)

        # Unknown values are refused rather than silently stored.
        styles.set_dash("sim::a", "squiggly")
        self.assertEqual(styles.dash_for("sim::a"), "dash")

        # Anything without an override follows the global width.
        styles.line_width = 2.5
        self.assertEqual(styles.width_for("sim::b"), 2.5)
        styles.clear_width_overrides()
        self.assertEqual(styles.width_for("sim::a"), 2.5)

    def test_styles_survive_round_trip(self):
        from aoeviewer.model import StyleRegistry
        styles = StyleRegistry()
        styles.set_color("iso::MACH::0.75", "#123456")
        styles.set_symbol("iso::MACH::0.75", "s")
        restored = StyleRegistry()
        restored.load_dict(styles.to_dict())
        self.assertEqual(restored.color_for("iso::MACH::0.75"), "#123456")
        self.assertEqual(restored.symbol_for("iso::MACH::0.75"), "s")


class TestAnnotations(unittest.TestCase):
    """Shaded areas and vertical lines are global, keyed by x parameter."""

    def test_round_trip(self):
        from aoeviewer.model import AppState, AreaBand, Marker, load_workspace, save_workspace

        state = AppState()
        state.areas.append(AreaBand(x_param="TIME", x_min=2.0, x_max=5.0,
                                    color="#112233", alpha=0.4, visible=True))
        state.markers.append(Marker(x_param="TIME", position=12.5,
                                    color="#ff0000", visible=False))

        handle, path = tempfile.mkstemp(suffix=".aoews")
        os.close(handle)
        try:
            save_workspace(path, state, [], {}, {})
            restored = AppState()
            restored.load_dict(load_workspace(path))

            self.assertEqual(len(restored.areas), 1)
            area = restored.areas[0]
            self.assertEqual(area.x_param, "TIME")
            self.assertAlmostEqual(area.x_min, 2.0)
            self.assertAlmostEqual(area.alpha, 0.4)
            self.assertEqual(area.color, "#112233")

            self.assertEqual(len(restored.markers), 1)
            marker = restored.markers[0]
            self.assertAlmostEqual(marker.position, 12.5)
            self.assertFalse(marker.visible)
        finally:
            os.remove(path)


class TestScaleLocking(unittest.TestCase):
    """Axis locking, and applying a range across tiles sharing a variable."""

    def _state(self):
        from aoeviewer.model import AppState
        state = AppState()
        state.page.rows, state.page.cols = 2, 2
        for index, (x, ys) in enumerate(
            [("TIME", ["ALPHA"]), ("TIME", ["NZ"]), ("MACH", ["ALPHA"]), ("TIME", ["ALT"])]
        ):
            state.page.tiles[index].x_param = x
            state.page.tiles[index].y_params = ys
        return state

    def test_lock_values_round_trip(self):
        from aoeviewer.model import AppState, load_workspace, save_workspace
        state = AppState()
        tile = state.tile
        tile.lock_x = True
        tile.x_min, tile.x_max = 10.0, 20.0
        tile.lock_y = True
        tile.y_min, tile.y_max = -3.0, 3.0

        handle, path = tempfile.mkstemp(suffix=".aoews")
        os.close(handle)
        try:
            save_workspace(path, state, [], {}, {})
            restored = AppState()
            restored.load_dict(load_workspace(path))
            back = restored.pages[0].tiles[0]
            self.assertTrue(back.lock_x and back.lock_y)
            self.assertAlmostEqual(back.x_min, 10.0)
            self.assertAlmostEqual(back.y_max, 3.0)
        finally:
            os.remove(path)

    def test_apply_x_reaches_only_matching_tiles(self):
        # The "same X axis" rule is what makes plots comparable; it must not
        # rescale a tile plotted against a different variable.
        state = self._state()
        source = state.page.tiles[0]
        source.lock_x = True
        source.x_min, source.x_max = 5.0, 15.0

        for tile in state.page.visible_tiles():
            if tile is source:
                continue
            if tile.x_param == source.x_param:
                tile.lock_x = source.lock_x
                tile.x_min, tile.x_max = source.x_min, source.x_max

        self.assertTrue(state.page.tiles[1].lock_x)      # TIME
        self.assertTrue(state.page.tiles[3].lock_x)      # TIME
        self.assertFalse(state.page.tiles[2].lock_x)     # MACH, untouched
        self.assertAlmostEqual(state.page.tiles[1].x_min, 5.0)


class TestStatistics(SampleFileMixin, unittest.TestCase):
    rows = 2000

    def test_matches_numpy(self):
        x = self.session.channel(0, 0, "TIME")
        y = self.session.channel(0, 0, "ALPHA")
        stats = Session.stats(y, x)
        self.assertIsNotNone(stats)
        self.assertAlmostEqual(stats["min"], float(y.min()), places=4)
        self.assertAlmostEqual(stats["max"], float(y.max()), places=4)
        self.assertAlmostEqual(stats["mean"], float(y.mean()), places=4)
        self.assertAlmostEqual(stats["std"], float(y.std(ddof=1)), places=4)
        self.assertEqual(stats["count"], y.size)

    def test_restricted_to_a_window(self):
        x = self.session.channel(0, 0, "TIME")
        y = self.session.channel(0, 0, "ALPHA")
        window = Session.stats(y, x, x_range=(5.0, 10.0))
        self.assertIsNotNone(window)
        self.assertLess(window["count"], y.size)

        mask = (x >= 5.0) & (x <= 10.0)
        self.assertAlmostEqual(window["max"], float(y[mask].max()), places=3)

    def test_empty_and_all_nan(self):
        self.assertIsNone(Session.stats(np.empty(0, dtype=np.float32)))
        self.assertIsNone(Session.stats(np.array([np.nan] * 5, dtype=np.float32)))


class TestIsoFastPath(SampleFileMixin, unittest.TestCase):
    """The ISO scan reads only the first record of each block.

    It must agree exactly with the slow path that loads the whole column,
    because it is what the colour mapping and the filter both key on.
    """

    sims = 6
    rows = 3000

    def test_agrees_with_a_full_column_load(self):
        fast = self.session.iso_values(0, "MACH")

        # Force the column resident, then read sample zero the long way.
        self.session.preload(0, [], ["MACH"])
        slow = np.array([
            self.session.channel(0, sim, "MACH")[0]
            for sim in range(self.file.simulation_count)
        ], dtype=np.float32)

        np.testing.assert_array_equal(fast, slow)

    def test_does_not_load_the_column(self):
        # The whole point: scanning ISO values must not pull half a gigabyte
        # into memory.
        before = self.session.resident_bytes
        self.session.iso_values(0, "MACH")
        self.assertEqual(self.session.resident_bytes, before)

    def test_unknown_parameter_is_all_nan(self):
        # One NaN per run rather than an empty array: the length always matches
        # the run count, so callers can index by simulation without a special
        # case. iso_distinct() then filters the NaNs out.
        values = self.session.iso_values(0, "NOT_A_CHANNEL")
        self.assertEqual(values.size, self.file.simulation_count)
        self.assertTrue(np.all(np.isnan(values)))
        self.assertEqual(self.session.iso_distinct("NOT_A_CHANNEL").size, 0)

    def test_works_for_computed_channels(self):
        self.session.set_computed("MACH2", "MACH * 2")
        fast = self.session.iso_values(0, "MACH")
        computed = self.session.iso_values(0, "MACH2")
        np.testing.assert_allclose(computed, fast * 2, rtol=1e-5)


class TestMemoryManagement(SampleFileMixin, unittest.TestCase):
    rows = 5000

    def test_resident_bytes_tracks_loading(self):
        self.assertEqual(self.session.resident_bytes, 0)
        self.session.preload(0, [], ["TIME", "ALPHA"])
        self.assertGreater(self.session.resident_bytes, 0)

    def test_trim_frees_memory(self):
        self.session.preload(0, [], self.file.headers)
        loaded = self.session.resident_bytes
        self.assertGreater(loaded, 0)

        after = self.session.trim_memory(loaded // 4)
        self.assertLess(after, loaded)

    def test_evicted_data_is_identical_when_re_read(self):
        # Eviction must be invisible apart from timing: the whole design rests
        # on being able to drop anything and read it back.
        self.session.preload(0, [], ["TIME", "ALPHA"])
        original = np.array(self.session.channel(0, 0, "ALPHA"))

        self.session.trim_memory(0)
        self.assertLess(self.session.resident_bytes, original.nbytes)

        again = self.session.channel(0, 0, "ALPHA")
        np.testing.assert_array_equal(original, again)

    def test_trim_below_current_use_is_a_no_op(self):
        self.session.preload(0, [], ["TIME"])
        resident = self.session.resident_bytes
        self.assertEqual(self.session.trim_memory(resident * 10), resident)


class TestSeriesFunctions(SampleFileMixin, unittest.TestCase):
    """deriv, integ, filters and moving average."""

    rows = 4000

    def test_derivative_of_x_by_x_is_one(self):
        self.session.set_computed("ONE", "deriv(TIME, TIME)")
        values = self.session.channel(0, 0, "ONE")
        self.assertGreater(values.size, 0)
        np.testing.assert_allclose(values, 1.0, rtol=1e-4)

    def test_integral_of_one_reproduces_x(self):
        self.session.set_computed("LIN", "integ(1 + 0*TIME, TIME)")
        time = self.session.channel(0, 0, "TIME")
        integral = self.session.channel(0, 0, "LIN")
        np.testing.assert_allclose(integral, time - time[0], atol=1e-3)

    def test_derivative_then_integral_round_trips(self):
        # Fundamental theorem of calculus, within the error of the schemes.
        self.session.set_computed("D", "deriv(ALPHA, TIME)")
        self.session.set_computed("R", "integ(D, TIME)")
        alpha = self.session.channel(0, 0, "ALPHA")
        back = self.session.channel(0, 0, "R")
        np.testing.assert_allclose(back, alpha - alpha[0], atol=0.05)

    def test_delta_starts_at_zero(self):
        self.session.set_computed("DEL", "delta(ALPHA)")
        values = self.session.channel(0, 0, "DEL")
        self.assertAlmostEqual(float(values[0]), 0.0, places=6)
        alpha = self.session.channel(0, 0, "ALPHA")
        np.testing.assert_allclose(values, alpha - alpha[0], rtol=1e-4)

    def test_lowpass_reduces_variation(self):
        self.session.set_computed("SMOOTH", "lowpass(ALPHA, TIME, 0.02)")
        alpha = self.session.channel(0, 0, "ALPHA")
        smooth = self.session.channel(0, 0, "SMOOTH")
        self.assertEqual(smooth.size, alpha.size)
        # A low-pass must not amplify: its sample-to-sample change is smaller.
        self.assertLess(float(np.abs(np.diff(smooth)).mean()),
                        float(np.abs(np.diff(alpha)).mean()))

    def test_highpass_removes_the_mean(self):
        self.session.set_computed("AC", "highpass(ALPHA, TIME, 0.5)")
        alpha = self.session.channel(0, 0, "ALPHA")
        ac = self.session.channel(0, 0, "AC")
        # ALPHA sits around 2-3; its high-pass must sit around zero.
        self.assertGreater(abs(float(alpha.mean())), 1.0)
        self.assertLess(abs(float(ac.mean())), 0.2)

    def test_moving_average_of_a_constant_is_that_constant(self):
        self.session.set_computed("FLAT", "movavg(0*ALPHA + 7, 31)")
        values = self.session.channel(0, 0, "FLAT")
        np.testing.assert_allclose(values, 7.0, rtol=1e-5)

    def test_moving_average_smooths(self):
        self.session.set_computed("MA", "movavg(ALPHA, 51)")
        alpha = self.session.channel(0, 0, "ALPHA")
        smoothed = self.session.channel(0, 0, "MA")
        self.assertEqual(smoothed.size, alpha.size)
        self.assertLess(float(smoothed.std()), float(alpha.std()) + 1e-6)

    def test_wrong_arity_is_rejected(self):
        for bad in ["deriv(ALPHA)", "integ(ALPHA)", "lowpass(ALPHA, TIME)",
                    "movavg(ALPHA)", "delta(ALPHA, TIME)"]:
            with self.assertRaises(CoreError, msg=bad):
                self.session.set_computed("BAD", bad)

    def test_series_functions_compose(self):
        self.session.set_computed("STEP1", "lowpass(ALPHA, TIME, 0.1)")
        self.session.set_computed("STEP2", "deriv(STEP1, TIME)")
        values = self.session.channel(0, 0, "STEP2")
        self.assertGreater(values.size, 0)
        self.assertFalse(np.all(np.isnan(values)))


class TestRenaming(SampleFileMixin, unittest.TestCase):
    def test_file_and_run_can_be_renamed(self):
        self.file.rename("Flight 42")
        self.assertEqual(self.session.file(0).name, "Flight 42")

        simulation = self.file.simulations[0]
        original_id = simulation.sim_id
        simulation.rename("Takeoff roll")

        again = self.session.file(0).simulations[0]
        self.assertEqual(again.name, "Takeoff roll")
        # The file's own name for the run is still available.
        self.assertEqual(again.sim_id, original_id)

    def test_identity_key_survives_rename(self):
        # Styles are keyed on this, so a rename must not recolour anything.
        before = self.file.simulations[0].key
        self.file.rename("Renamed")
        self.file.simulations[0].rename("Also renamed")
        self.assertEqual(self.session.file(0).simulations[0].key, before)

    def test_blank_names_are_ignored(self):
        original = self.file.name
        self.file.rename("")
        self.assertEqual(self.session.file(0).name, original)


class TestMouseModeState(unittest.TestCase):
    def test_round_trip(self):
        from aoeviewer.model import AppState, MouseMode, load_workspace, save_workspace

        state = AppState()
        self.assertEqual(state.mouse_mode, MouseMode.SELECT)
        state.mouse_mode = MouseMode.PAN

        handle, path = tempfile.mkstemp(suffix=".aoews")
        os.close(handle)
        try:
            save_workspace(path, state, [], {}, {})
            restored = AppState()
            restored.load_dict(load_workspace(path))
            self.assertEqual(restored.mouse_mode, MouseMode.PAN)
        finally:
            os.remove(path)


class TestMultiIsoState(unittest.TestCase):
    def test_active_filters_are_listed_in_order(self):
        from aoeviewer.model import AppState

        state = AppState()
        state.iso_state("MACH").active = True
        state.iso_state("ALT").active = True
        state.iso_state("MASS").active = False

        names = [iso.param for iso in state.active_iso_filters()]
        self.assertEqual(names, ["ALT", "MACH"])

    def test_iso_state_creates_on_demand(self):
        from aoeviewer.model import AppState

        state = AppState()
        iso = state.iso_state("MACH")
        self.assertEqual(iso.param, "MACH")
        self.assertIs(state.iso_state("MACH"), iso)


class TestMarkerModel(unittest.TestCase):
    def test_describe_names_the_condition(self):
        # A marker has no caption of its own any more: pyqtgraph anchors an
        # InfiniteLine's label to the line, so it sat on the data with nothing
        # behind it, and editing one rebuilt the tile and deleted the item being
        # edited. Text notes do captions properly. What is left is the condition
        # the marker states, which is what the list needs to show.
        from aoeviewer.model import Marker

        marker = Marker(x_param="TIME", position=5.0)
        self.assertEqual(marker.describe(), "TIME = 5")
        self.assertFalse(hasattr(marker, "label"))

    def test_round_trip_keeps_the_dot_size(self):
        from aoeviewer.model import AppState, Marker, load_workspace, save_workspace

        state = AppState()
        state.markers.append(Marker(x_param="TIME", position=5.0, dot_size=12.0))
        handle, path = tempfile.mkstemp(suffix=".aoews")
        os.close(handle)
        try:
            save_workspace(path, state, [], {}, {})
            restored = AppState()
            restored.load_dict(load_workspace(path))
            self.assertAlmostEqual(restored.markers[0].dot_size, 12.0)
            self.assertEqual(restored.markers[0].x_param, "TIME")
        finally:
            os.remove(path)


# ==========================================================================
# Header dialects
# ==========================================================================


class TestIndexedHeaderDialect(unittest.TestCase):
    """*PARAMETRES written one parameter per line, with ordinals and units.

        <tab>1<tab>'TIME'<tab><tab>'s'
        <tab>2<tab>'PARAMETER001'<tab> <tab>''
        <tab>3<tab>'PARAMETER002'<tab> <tab>'deg'

    Reading every quoted token as a name is the obvious implementation and it
    is badly wrong: a 585-parameter file becomes 1156 parameters, 'deg' and
    'ft' appear in the channel list, and -- the part that actually costs you a
    day -- every column index shifts, so TIME quietly returns another
    channel's data. The file loads, the plots draw, and the numbers are
    somebody else's.
    """

    sims = 3
    rows = 40

    @classmethod
    def setUpClass(cls):
        handle, cls.path = tempfile.mkstemp(suffix=".aoe", prefix="aoe_indexed_")
        os.close(handle)
        generate(cls.path, cls.sims, cls.rows, wrap=True,
                 indexed=True, entity=True)

        handle, cls.packed_path = tempfile.mkstemp(suffix=".aoe", prefix="aoe_packed_")
        os.close(handle)
        generate(cls.packed_path, cls.sims, cls.rows)

    @classmethod
    def tearDownClass(cls):
        for path in (cls.path, cls.packed_path):
            try:
                os.remove(path)
            except OSError:
                pass

    def setUp(self):
        self.session = Session()
        self.file = self.session.add_file(self.path)

    def tearDown(self):
        self.session.close()

    def test_units_are_not_parameters(self):
        self.assertEqual(
            self.file.headers,
            ["TIME", "ALPHA", "MACH", "NZ", "ALT", "THETA"],
        )
        for unit in ("s", "deg", "ft"):
            self.assertNotIn(unit, self.file.headers)

    def test_units_are_captured(self):
        self.assertEqual(self.file.units, ["s", "deg", "", "", "ft", "deg"])
        self.assertEqual(len(self.file.units), len(self.file.headers))

    def test_unit_lookup_by_name(self):
        self.assertEqual(self.session.unit("TIME"), "s")
        self.assertEqual(self.session.unit("ALT"), "ft")
        # Dimensionless channels declare '' and must come back as "", not as
        # the unit of whichever parameter follows them.
        self.assertEqual(self.session.unit("MACH"), "")
        self.assertEqual(self.session.unit("NOT_A_CHANNEL"), "")

    def test_label_combines_name_and_unit(self):
        self.assertEqual(self.session.label("TIME"), "TIME [s]")
        self.assertEqual(self.session.label("MACH"), "MACH")

    def test_unit_follows_aliases(self):
        self.session.set_alias("Altitude", "ALT")
        self.assertEqual(self.session.unit("Altitude"), "ft")
        self.assertEqual(self.session.label("Altitude"), "Altitude [ft]")

    def test_columns_are_not_shifted(self):
        """The real damage of a miscounted header list."""
        time = self.session.channel(0, 0, "TIME")
        self.assertEqual(time.size, self.rows)
        # generate() writes t = row * 0.01, strictly increasing.
        self.assertAlmostEqual(float(time[0]), 0.0, places=6)
        self.assertAlmostEqual(float(time[1]), 0.01, places=6)
        self.assertTrue(bool(np.all(np.diff(time) > 0)))

        mach = self.session.channel(0, 0, "MACH")
        self.assertTrue(bool(np.all(mach == mach[0])))

    def test_entity_block_is_skipped(self):
        """Free text between the parameters and the first table is not data."""
        self.assertEqual(self.file.simulation_count, self.sims)
        for sim in range(self.sims):
            self.assertEqual(self.session.channel(0, sim, "TIME").size, self.rows)

    def test_tableau_num_simu_ids(self):
        ids = [sim.sim_id for sim in self.file.simulations]
        self.assertEqual(ids, ["1.2.0", "2.2.0", "3.2.0"])

    def test_packed_dialect_still_works(self):
        """The other dialect must not regress: names packed onto one line."""
        packed = self.session.add_file(self.packed_path)
        self.assertEqual(
            packed.headers, ["TIME", "ALPHA", "MACH", "NZ", "ALT", "THETA"]
        )
        # No units declared, but the list is still parallel to the headers.
        self.assertEqual(packed.units, [""] * 6)
        self.assertEqual(self.session.label("TIME"), "TIME [s]")

    def test_both_dialects_agree_on_the_data(self):
        packed = self.session.add_file(self.packed_path)
        for name in ("TIME", "ALPHA", "MACH", "NZ", "ALT", "THETA"):
            indexed_values = self.session.channel(0, 0, name)
            packed_values = self.session.channel(packed.index, 0, name)
            self.assertEqual(indexed_values.size, packed_values.size, name)
            self.assertTrue(
                bool(np.allclose(indexed_values, packed_values, atol=1e-5)), name
            )


class TestStridedArrays(unittest.TestCase):
    """Arrays the core reads as flat C buffers.

    The core takes a pointer and a count and knows nothing about NumPy strides.
    A strided view -- which is exactly what the 3D projection builds when it
    thins a long trajectory -- therefore had its first `count` *contiguous*
    samples read instead of every step-th one, so the bounds of a million-sample
    flight came from its first thousandth.
    """

    def setUp(self):
        self.values = np.arange(1000, dtype=np.float32)
        self.strided = self.values[::100]      # 0, 100, ... 900

    def test_bounds_of_a_strided_view(self):
        self.assertEqual(Session.bounds(self.strided), (0.0, 900.0))

    def test_bounds_match_an_explicit_copy(self):
        self.assertEqual(Session.bounds(self.strided),
                         Session.bounds(np.ascontiguousarray(self.strided)))

    def test_stats_of_a_strided_view(self):
        stats = Session.stats(self.strided)
        self.assertIsNotNone(stats)
        self.assertAlmostEqual(stats["min"], 0.0)
        self.assertAlmostEqual(stats["max"], 900.0)
        self.assertEqual(stats["count"], self.strided.size)

    def test_nearest_index_on_a_strided_view(self):
        # 550 sits between 500 (index 5) and 600 (index 6).
        self.assertIn(Session.nearest_index(self.strided, 550.0), (5, 6))
        self.assertEqual(Session.nearest_index(self.strided, 0.0), 0)
        self.assertEqual(Session.nearest_index(self.strided, 1e9), 9)

    def test_decimation_of_a_strided_view(self):
        x = np.arange(2000, dtype=np.float32)[::4]
        y = (x * 2.0).astype(np.float32)
        session = Session()
        try:
            dx, dy = session.decimate(x, y, 64)
            self.assertGreater(dx.size, 0)
            # The decimated envelope must span the same range as the input.
            self.assertAlmostEqual(float(dx.min()), float(x.min()), places=3)
            self.assertAlmostEqual(float(dx.max()), float(x.max()), places=3)
            self.assertAlmostEqual(float(dy.max()), float(y.max()), places=3)
        finally:
            session.close()

    def test_a_contiguous_array_is_not_copied(self):
        """The zero-copy path must stay zero-copy: this is the hot loop."""
        from aoeviewer.session import _c_floats

        contiguous = np.ascontiguousarray(self.values)
        self.assertIs(_c_floats(contiguous), contiguous)


# ---------------------------------------------------------------------------
# Regression tests for the defects listed in the post-flight review.
# Each names the defect it closes so a future reader can find the reasoning.
# ---------------------------------------------------------------------------


def _write_aoe(path: str, headers, rows) -> None:
    with open(path, "w") as handle:
        handle.write("*PARAMETRES\n " + " ".join("'%s'" % h for h in headers) + "\n")
        handle.write("*TABLEAU : R1\n")
        for row in rows:
            handle.write(" " + " ".join("%.6f" % v for v in row) + "\n")


class TestPinnedViews(unittest.TestCase):
    """AOE-08: eviction must not free memory a live view is reading."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "pinned.aoe")
        rows = [(i * 0.01, math.sin(i * 0.001), math.cos(i * 0.001))
                for i in range(20000)]
        _write_aoe(self.path, ["TIME", "A", "B"], rows)
        self.session = Session()
        self.session.add_file(self.path)

    def tearDown(self):
        self.session.close()

    def test_a_view_survives_a_full_trim(self):
        values = self.session.channel(0, 0, "A")
        expected = float(values[:100].sum())

        self.assertGreater(self.session.resident_bytes, 0)
        self.session.trim_memory(0)

        # The cache has let go, but the samples this view points at have not
        # moved: reading them is still safe and still correct.
        self.assertAlmostEqual(float(values[:100].sum()), expected, places=4)
        self.assertEqual(values.size, 20000)

    def test_a_dropped_view_releases_its_memory(self):
        values = self.session.channel(0, 0, "A")
        self.session.trim_memory(0)
        del values
        import gc
        gc.collect()
        self.assertEqual(self.session.resident_bytes, 0)

    def test_the_column_is_re_read_after_eviction(self):
        first = np.array(self.session.channel(0, 0, "A"))
        self.session.trim_memory(0)
        second = self.session.channel(0, 0, "A")
        self.assertTrue(np.array_equal(first, second))

    def test_a_view_carries_its_summary(self):
        from aoeviewer._core import pin_of

        time = self.session.channel(0, 0, "TIME")
        pin = pin_of(time)
        self.assertIsNotNone(pin)
        self.assertTrue(pin.sorted)
        low, high = pin.bounds()
        self.assertAlmostEqual(low, 0.0, places=4)
        self.assertAlmostEqual(high, 199.99, places=2)

    def test_the_cached_extent_matches_a_full_scan(self):
        for name in ("TIME", "A", "B"):
            values = self.session.channel(0, 0, name)
            cached = Session.bounds(values)
            scanned = Session.bounds(np.array(values))  # a plain copy: no pin
            self.assertIsNotNone(cached)
            self.assertEqual(cached, scanned, name)


class TestNonMonotonicX(unittest.TestCase):
    """AOE-12: a phase plane has x rising and then falling."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "phase.aoe")
        # A climb and descent: ALTITUDE goes up to 300 and back down, while
        # MACH rises throughout. Plotted as MACH = f(ALTITUDE) the curve
        # doubles back on itself.
        rows = []
        for i in range(101):
            altitude = 100.0 + 200.0 * (1.0 - abs(i - 50) / 50.0)
            rows.append((i * 1.0, altitude, 0.2 + i * 0.004))
        _write_aoe(self.path, ["TIME", "ALTITUDE", "MACH"], rows)
        self.session = Session()
        self.session.add_file(self.path)

    def tearDown(self):
        self.session.close()

    def test_both_branches_of_the_trajectory_survive(self):
        altitude = self.session.channel(0, 0, "ALTITUDE")
        mach = self.session.channel(0, 0, "MACH")

        dx, dy = self.session.decimate(altitude, mach, 200, (150.0, 350.0))

        # The window covers the climb and the descent. Both must be there,
        # which means MACH values from before and after the apex.
        self.assertGreater(float(dy.max()), 0.5)
        self.assertLess(float(dy.min()), 0.3)
        # And the x samples must go up and come back down.
        peak = int(np.argmax(dx))
        self.assertGreater(peak, 0)
        self.assertLess(peak, dx.size - 1)

    def test_statistics_use_the_whole_window(self):
        altitude = self.session.channel(0, 0, "ALTITUDE")
        mach = self.session.channel(0, 0, "MACH")

        stats = Session.stats(mach, altitude, (250.0, 350.0))
        self.assertIsNotNone(stats)
        self.assertTrue(stats["ranged"])

        # Compare against the answer worked out in NumPy, which knows nothing
        # about binary searches.
        mask = (np.asarray(altitude) >= 250.0) & (np.asarray(altitude) <= 350.0)
        self.assertEqual(stats["count"], int(mask.sum()))
        self.assertAlmostEqual(stats["max"], float(np.asarray(mach)[mask].max()),
                               places=5)
        self.assertAlmostEqual(stats["min"], float(np.asarray(mach)[mask].min()),
                               places=5)

    def test_a_sorted_channel_still_takes_the_fast_path(self):
        from aoeviewer._core import pin_of

        time = self.session.channel(0, 0, "TIME")
        altitude = self.session.channel(0, 0, "ALTITUDE")
        self.assertTrue(pin_of(time).sorted)
        self.assertFalse(pin_of(altitude).sorted)

    def test_decimation_agrees_with_the_sorted_path_on_sorted_data(self):
        time = self.session.channel(0, 0, "TIME")
        mach = self.session.channel(0, 0, "MACH")
        window = (20.0, 60.0)

        fast = self.session.decimate(time, mach, 500, window)
        # A copy has no pin, so the core has to work the ordering out itself.
        slow = self.session.decimate(np.array(time), np.array(mach), 500, window)
        self.assertTrue(np.array_equal(fast[0], slow[0]))
        self.assertTrue(np.array_equal(fast[1], slow[1]))


class TestFileNameReuse(unittest.TestCase):
    """AOE-16: removing and re-adding a file must not rename it."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "flight.aoe")
        _write_aoe(self.path, ["TIME", "A"], [(0.0, 1.0), (1.0, 2.0)])
        self.session = Session()

    def tearDown(self):
        self.session.close()

    def test_the_name_comes_back_after_a_removal(self):
        handle = self.session.add_file(self.path)
        self.assertEqual(handle.name, "flight.aoe")

        self.session.remove_file(0)
        handle = self.session.add_file(self.path)
        # Styles and saved workspaces are keyed on this name, so a suffix here
        # silently discarded every colour the user had chosen.
        self.assertEqual(handle.name, "flight.aoe")

    def test_a_genuine_duplicate_is_still_suffixed(self):
        first = self.session.add_file(self.path)
        second = self.session.add_file(self.path)
        self.assertEqual(first.name, "flight.aoe")
        self.assertEqual(second.name, "flight.aoe (2)")

    def test_the_gap_left_by_a_removal_is_reused(self):
        self.session.add_file(self.path)
        self.session.add_file(self.path)   # flight.aoe (2)
        self.session.add_file(self.path)   # flight.aoe (3)
        self.session.remove_file(1)        # frees "flight.aoe (2)"
        added = self.session.add_file(self.path)
        self.assertEqual(added.name, "flight.aoe (2)")
        names = sorted(f.name for f in self.session.files)
        self.assertEqual(len(names), len(set(names)))


class TestStatisticsScope(unittest.TestCase):
    """AOE-17: say so when the x window could not be applied."""

    def test_a_short_x_channel_reports_whole_channel(self):
        y = np.arange(100, dtype=np.float32)
        x = np.arange(50, dtype=np.float32)
        stats = Session.stats(y, x, (10.0, 20.0))
        self.assertIsNotNone(stats)
        self.assertFalse(stats["ranged"])
        self.assertEqual(stats["count"], 50)

    def test_a_full_length_x_channel_reports_ranged(self):
        y = np.arange(100, dtype=np.float32)
        x = np.arange(100, dtype=np.float32)
        stats = Session.stats(y, x, (10.0, 20.0))
        self.assertTrue(stats["ranged"])

    def test_no_range_reports_whole_channel(self):
        y = np.arange(100, dtype=np.float32)
        stats = Session.stats(y)
        self.assertFalse(stats["ranged"])


class TestLogScaleConversion(unittest.TestCase):
    """AOE-13: the view is in log units, the tile's range is in data units."""

    def test_a_round_trip_returns_the_original(self):
        from aoeviewer import scales

        for value in (1.0, 10.0, 1000.0, 0.001, 7.25):
            view = scales.to_view(value, True)
            self.assertIsNotNone(view)
            self.assertAlmostEqual(scales.from_view(view, True), value, places=4)

    def test_a_thousandfold_span_is_three_decades(self):
        from aoeviewer import scales

        low, high = scales.range_to_view(1.0, 1000.0, True)
        self.assertAlmostEqual(low, 0.0)
        self.assertAlmostEqual(high, 3.0)

    def test_linear_mode_is_the_identity(self):
        from aoeviewer import scales

        self.assertEqual(scales.to_view(-5.0, False), -5.0)
        self.assertEqual(scales.from_view(-5.0, False), -5.0)
        self.assertEqual(scales.range_to_view(-5.0, 5.0, False), (-5.0, 5.0))

    def test_non_positive_values_have_no_log(self):
        from aoeviewer import scales

        self.assertIsNone(scales.to_view(0.0, True))
        self.assertIsNone(scales.to_view(-1.0, True))

    def test_a_channel_crossing_zero_is_framed_on_what_can_be_drawn(self):
        from aoeviewer import scales

        framed = scales.range_to_view(-10.0, 100.0, True)
        self.assertIsNotNone(framed)
        low, high = framed
        self.assertAlmostEqual(high, 2.0)
        self.assertLess(low, high)

    def test_an_entirely_negative_channel_cannot_be_framed(self):
        from aoeviewer import scales

        self.assertIsNone(scales.range_to_view(-100.0, -1.0, True))

    def test_the_exponent_is_clamped(self):
        from aoeviewer import scales

        self.assertTrue(math.isfinite(scales.from_view(1e9, True)))


class TestFormulaExtensions(unittest.TestCase):
    """Reductions, and lookups at either end of a run."""

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "runs.aoe")
        with open(self.path, "w") as handle:
            handle.write("*PARAMETRES\n 'TIME' 'NZ'\n")
            # Two runs with different lengths and different peaks: what makes
            # "the end" impossible to write as a number, and what makes a
            # per-run reduction different from a global one.
            for name, rows in (("SHORT", [(0.0, 1.0), (1.0, 4.0), (2.0, 2.0)]),
                               ("LONG", [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0),
                                         (3.0, 8.0), (4.0, 5.0)])):
                handle.write("*TABLEAU : %s\n" % name)
                for t, nz in rows:
                    handle.write(" %.3f %.3f\n" % (t, nz))
        self.session = Session()
        self.session.add_file(self.path)

    def tearDown(self):
        self.session.close()

    def test_reductions_are_per_run(self):
        self.session.set_computed("PEAK", "max(NZ)")
        short = self.session.channel(0, 0, "PEAK")
        long_run = self.session.channel(0, 1, "PEAK")
        self.assertAlmostEqual(float(short[0]), 4.0, places=4)
        self.assertAlmostEqual(float(long_run[0]), 8.0, places=4)
        # Held flat across the run so it combines with per-sample channels.
        self.assertTrue(np.allclose(np.asarray(short), 4.0))

    def test_min_and_mean(self):
        self.session.set_computed("LOW", "min(NZ)")
        self.session.set_computed("AVG", "mean(NZ)")
        self.assertAlmostEqual(float(self.session.channel(0, 0, "LOW")[0]), 1.0,
                               places=4)
        self.assertAlmostEqual(float(self.session.channel(0, 0, "AVG")[0]),
                               7.0 / 3.0, places=4)

    def test_two_arguments_is_still_pointwise(self):
        self.session.set_computed("BOTH", "max(NZ, TIME)")
        values = self.session.channel(0, 0, "BOTH")
        self.assertTrue(np.array_equal(np.asarray(values), [1.0, 4.0, 2.0]))

    def test_normalising_against_the_peak(self):
        self.session.set_computed("NORM", "NZ / max(NZ)")
        self.assertAlmostEqual(float(self.session.channel(0, 0, "NORM").max()),
                               1.0, places=5)
        self.assertAlmostEqual(float(self.session.channel(0, 1, "NORM").max()),
                               1.0, places=5)

    def test_first_and_last_lookup(self):
        self.session.set_computed("START", "NZ[TIME=first]")
        self.session.set_computed("END", "NZ[TIME=last]")
        self.assertAlmostEqual(float(self.session.channel(0, 0, "START")[0]), 1.0)
        self.assertAlmostEqual(float(self.session.channel(0, 0, "END")[0]), 2.0)
        # The runs have different lengths, which is the point of the keywords.
        self.assertAlmostEqual(float(self.session.channel(0, 1, "END")[0]), 5.0)

    def test_a_numeric_lookup_still_works(self):
        self.session.set_computed("AT_ONE", "NZ[TIME=1.0]")
        self.assertAlmostEqual(float(self.session.channel(0, 0, "AT_ONE")[0]), 4.0)

    def test_the_error_names_the_accepted_argument_counts(self):
        ok, message = self.session.validate_formula("max(1, 2, 3)")
        self.assertFalse(ok)
        self.assertIn("1 or 2", message)

    def test_an_unknown_lookup_keyword_is_refused(self):
        ok, message = self.session.validate_formula("NZ[TIME=middle]")
        self.assertFalse(ok)
        self.assertIn("first", message)


class TestViewPersistence(unittest.TestCase):
    """A redraw must not throw away the user's zoom."""

    def test_a_new_tile_is_not_yet_framed(self):
        from aoeviewer.model import TileState

        self.assertFalse(TileState().view_initialised)

    def test_the_flag_survives_a_workspace_round_trip(self):
        from aoeviewer.model import TileState

        tile = TileState()
        tile.view_initialised = True
        tile.x_min, tile.x_max = 12.0, 34.0
        restored = TileState.from_dict(tile.to_dict())
        self.assertTrue(restored.view_initialised)
        self.assertEqual((restored.x_min, restored.x_max), (12.0, 34.0))

    def test_an_old_workspace_loads_without_the_flag(self):
        from aoeviewer.model import TileState

        data = TileState().to_dict()
        data.pop("view_initialised")
        restored = TileState.from_dict(data)
        # Absent means "never framed", so such a tile is fitted once on load
        # rather than opening at an arbitrary default range.
        self.assertFalse(restored.view_initialised)


class TestLegendDefaults(unittest.TestCase):
    """A legend row per run, times fifty runs, is not a legend."""

    def test_only_the_parameter_is_shown_by_default(self):
        from aoeviewer.model import TileState

        tile = TileState()
        self.assertTrue(tile.legend_show_param)
        self.assertFalse(tile.legend_show_file)
        self.assertFalse(tile.legend_show_sim)
        self.assertFalse(tile.legend_show_iso)

    def test_an_old_workspace_keeps_what_it_asked_for(self):
        from aoeviewer.model import TileState

        restored = TileState.from_dict({"legend_show_file": True,
                                        "legend_show_sim": True})
        self.assertTrue(restored.legend_show_file)
        self.assertTrue(restored.legend_show_sim)

    def test_the_withdrawn_style_option_still_loads(self):
        from aoeviewer.model import TileState

        restored = TileState.from_dict({"legend_show_style": True})
        self.assertTrue(restored.legend_show_style)


class TestParameterResolution(unittest.TestCase):
    """AOE-14: missing channels are settled once, not by a panel refresh."""

    def _state(self):
        from aoeviewer.model import AppState

        state = AppState()
        tile = state.tile
        tile.x_param = "GONE"
        tile.y_params = ["ALSO_GONE"]
        tile.z_param = "MISSING_Z"
        tile.color_iso_param = "MISSING_ISO"
        tile.style_iso_param = "MISSING_STYLE_ISO"
        return state, tile

    def test_a_missing_x_axis_is_re_pointed_and_reported(self):
        state, tile = self._state()
        notes = state.resolve_missing_parameters(["TIME", "ALPHA"])
        self.assertEqual(tile.x_param, "TIME")
        self.assertTrue(any("x axis" in n for n in notes))
        self.assertTrue(state.dirty)

    def test_missing_y_channels_are_left_alone(self):
        state, tile = self._state()
        state.resolve_missing_parameters(["TIME", "ALPHA"])
        # They come back by themselves when the file is reopened, which is the
        # friendlier behaviour when a study is loaded a file at a time.
        self.assertEqual(tile.y_params, ["ALSO_GONE"])

    def test_missing_z_and_colour_channels_are_cleared(self):
        state, tile = self._state()
        state.resolve_missing_parameters(["TIME", "ALPHA"])
        self.assertEqual(tile.z_param, "")
        self.assertEqual(tile.color_iso_param, "")

    def test_a_missing_style_iso_parameter_is_cleared_too(self):
        # Style-by-ISO arrived in 2.1.0 and this repair did not follow it. A
        # tile left naming a parameter the new files lack falls through to the
        # colour key, so its dashes stop distinguishing anything while the
        # STYLE tab still shows the old parameter as the answer.
        state, tile = self._state()
        notes = state.resolve_missing_parameters(["TIME", "ALPHA"])
        self.assertEqual(tile.style_iso_param, "")
        self.assertTrue(any("style-by" in n for n in notes))

    def test_both_iso_encodings_move_to_a_parameter_that_is_there(self):
        state, tile = self._state()
        iso = state.iso_state("MACH")
        iso.active = True
        state.resolve_missing_parameters(["TIME", "ALPHA", "MACH"])
        self.assertEqual(tile.color_iso_param, "MACH")
        self.assertEqual(tile.style_iso_param, "MACH")

    def test_nothing_happens_with_no_files_open(self):
        state, tile = self._state()
        self.assertEqual(state.resolve_missing_parameters([]), [])
        self.assertEqual(tile.x_param, "GONE")
        self.assertFalse(state.dirty)

    def test_a_valid_tile_is_untouched(self):
        from aoeviewer.model import AppState

        state = AppState()
        state.tile.x_param = "TIME"
        state.tile.y_params = ["ALPHA"]
        self.assertEqual(state.resolve_missing_parameters(["TIME", "ALPHA"]), [])
        self.assertFalse(state.dirty)


class TestTileDefaults(unittest.TestCase):
    """A tile that is plotting something must have an x axis.

    The panel shows a default in its combo but no longer writes it to the tile
    -- a refresh is not a user action. That left the two disagreeing: the combo
    said TIME, the tile said nothing, and the title read "no x axis selected".
    Committing the default belongs in one deliberate place, and this is it.
    """

    def _state(self):
        from aoeviewer.model import AppState

        return AppState()

    def test_a_tile_with_a_parameter_gets_the_default_x(self):
        state = self._state()
        state.tile.y_params = ["ALPHA"]
        self.assertTrue(state.apply_parameter_defaults(["ALPHA", "NZ", "TIME"]))
        self.assertEqual(state.tile.x_param, "TIME")

    def test_a_time_like_channel_is_preferred(self):
        state = self._state()
        state.tile.y_params = ["ALPHA"]
        state.apply_parameter_defaults(["ALPHA", "TIME_S", "NZ"])
        self.assertEqual(state.tile.x_param, "TIME_S")

    def test_without_a_time_channel_the_first_parameter_is_used(self):
        state = self._state()
        state.tile.y_params = ["NZ"]
        state.apply_parameter_defaults(["ALPHA", "MACH", "NZ"])
        self.assertEqual(state.tile.x_param, "ALPHA")

    def test_a_choice_the_user_made_is_never_overwritten(self):
        state = self._state()
        state.tile.y_params = ["ALPHA"]
        state.tile.x_param = "MACH"
        self.assertFalse(state.apply_parameter_defaults(["ALPHA", "MACH", "TIME"]))
        self.assertEqual(state.tile.x_param, "MACH")

    def test_an_empty_tile_is_left_empty(self):
        state = self._state()
        state.apply_parameter_defaults(["ALPHA", "TIME"])
        # Relabelling all twenty-five cells of a blank grid helps nobody.
        self.assertEqual(state.tile.x_param, "")

    def test_a_3d_tile_gets_one_too(self):
        state = self._state()
        state.tile.z_param = "ALTITUDE"
        state.apply_parameter_defaults(["ALTITUDE", "TIME"])
        self.assertEqual(state.tile.x_param, "TIME")

    def test_nothing_happens_before_any_file_is_open(self):
        state = self._state()
        state.tile.y_params = ["ALPHA"]
        self.assertFalse(state.apply_parameter_defaults([]))
        self.assertEqual(state.tile.x_param, "")

    def test_it_is_idempotent(self):
        state = self._state()
        state.tile.y_params = ["ALPHA"]
        params = ["ALPHA", "TIME"]
        self.assertTrue(state.apply_parameter_defaults(params))
        self.assertFalse(state.apply_parameter_defaults(params))

    def test_every_page_is_covered(self):
        state = self._state()
        state.add_page()
        for page in state.pages:
            page.tiles[0].y_params = ["ALPHA"]
        state.apply_parameter_defaults(["ALPHA", "TIME"])
        for page in state.pages:
            self.assertEqual(page.tiles[0].x_param, "TIME")

    def test_colour_by_iso_gets_a_parameter(self):
        from aoeviewer.model import ColorMode

        state = self._state()
        state.tile.y_params = ["ALPHA"]
        state.tile.color_mode = ColorMode.ISO
        state.apply_parameter_defaults(["ALPHA", "MASS", "TIME"])
        # Selecting the mode without a parameter drew every curve one colour
        # with nothing to say why.
        self.assertTrue(state.tile.color_iso_param)

    def test_a_configured_iso_parameter_is_preferred(self):
        from aoeviewer.model import ColorMode

        state = self._state()
        state.tile.y_params = ["ALPHA"]
        state.tile.color_mode = ColorMode.ISO
        state.iso_states["MASS"] = state.iso_states.get("MASS") or _IsoStub()
        state.apply_parameter_defaults(["ALPHA", "MASS", "TIME"])
        self.assertEqual(state.tile.color_iso_param, "MASS")

    def test_the_title_names_the_axis_once_the_default_is_applied(self):
        from aoeviewer.model import tile_title

        state = self._state()
        state.tile.y_params = ["ALPHA"]
        self.assertIn("no x axis", tile_title(state.tile))
        state.apply_parameter_defaults(["ALPHA", "TIME"])
        self.assertEqual(tile_title(state.tile), "ALPHA = f(TIME)")


class _IsoStub:
    """Stands in for whatever the ISO panel stores; only the key matters here."""

    grouped = False
    groups = ()


class TestThreeDGrid(unittest.TestCase):
    """Back-wall grid geometry for the 3D tiles.

    The whole point of the grid is that it stays *behind* the trajectory as the
    view turns. That is one line of sign logic which is invisible at any single
    rotation and wrong at some other one, so it is asserted across the sphere
    rather than eyeballed.
    """

    def test_ticks_are_round_numbers(self):
        from aoeviewer import projection

        self.assertEqual(projection.nice_ticks(0.0, 100.0, 5),
                         [0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
        self.assertEqual(projection.nice_ticks(0.0, 1.0, 4),
                         [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_ticks_stay_inside_the_range(self):
        from aoeviewer import projection

        for low, high in ((0.13, 0.87), (-45.0, 17.5), (1e4, 3.2e4),
                          (-1e-3, 4e-3), (999.4, 1000.6)):
            ticks = projection.nice_ticks(low, high, 5)
            self.assertTrue(ticks, (low, high))
            self.assertGreaterEqual(min(ticks), low - 1e-9)
            self.assertLessEqual(max(ticks), high + 1e-9)
            # Evenly spaced, or they are not a scale.
            steps = {round(b - a, 9) for a, b in zip(ticks, ticks[1:])}
            self.assertLessEqual(len(steps), 1)

    def test_a_degenerate_axis_gets_no_scale(self):
        from aoeviewer import projection

        # A constant channel. Inventing ticks for it would be a lie.
        self.assertEqual(projection.nice_ticks(5.0, 5.0), [])
        self.assertEqual(projection.nice_ticks(5.0, 1.0), [])
        self.assertEqual(projection.nice_ticks(float("nan"), 1.0), [])

    def test_labels_carry_the_precision_the_step_needs(self):
        from aoeviewer import projection

        self.assertEqual(projection.format_tick(0.25, 0.25), "0.25")
        self.assertEqual(projection.format_tick(1500.0, 500.0), "1500")
        self.assertEqual(projection.format_tick(0.0, 0.2), "0")
        self.assertEqual(projection.format_tick(-45.0, 15.0), "-45")
        # Very large and very small read better as exponents.
        self.assertIn("e", projection.format_tick(2.5e8, 1e8))

    def test_the_walls_are_always_behind_the_data(self):
        from aoeviewer import projection

        # The wall must sit at lower depth than the opposite face, at every
        # rotation -- that is precisely what "behind" means for this
        # projection, and getting the sign wrong draws the grid over the plot.
        for azimuth in range(0, 360, 7):
            for elevation in (-89, -60, -20, -1, 1, 20, 60, 89):
                walls = projection.back_walls(azimuth, elevation)
                for axis, wall in enumerate(walls):
                    point = [0.0, 0.0, 0.0]
                    point[axis] = wall
                    opposite = list(point)
                    opposite[axis] = -wall

                    _sx, _sy, near = projection.project(
                        np.array([point[0]]), np.array([point[1]]),
                        np.array([point[2]]), azimuth, elevation)
                    _ox, _oy, far = projection.project(
                        np.array([opposite[0]]), np.array([opposite[1]]),
                        np.array([opposite[2]]), azimuth, elevation)
                    self.assertLessEqual(
                        float(near[0]), float(far[0]) + 1e-9,
                        "azimuth {0}, elevation {1}, axis {2}".format(
                            azimuth, elevation, axis))

    def test_the_floor_is_below_when_looking_down(self):
        from aoeviewer import projection

        _x, _y, z = projection.back_walls(45.0, 30.0)
        self.assertEqual(z, -0.5)
        # ...and above when looking up from underneath.
        _x, _y, z = projection.back_walls(45.0, -30.0)
        self.assertEqual(z, 0.5)

    def test_the_walls_swap_as_the_view_turns(self):
        from aoeviewer import projection

        front = projection.back_walls(45.0, 25.0)
        behind = projection.back_walls(225.0, 25.0)
        self.assertEqual(front[0], -behind[0])
        self.assertEqual(front[1], -behind[1])

    def test_every_segment_lies_on_a_wall(self):
        from aoeviewer import projection

        fractions = [-0.5, -0.25, 0.0, 0.25, 0.5]
        walls = projection.back_walls(35.0, 20.0)
        segments = projection.wall_grid_segments(
            35.0, 20.0, fractions, fractions, fractions)
        self.assertTrue(segments)

        for start, end in segments:
            # Every point stays inside the box...
            for point in (start, end):
                for coordinate in point:
                    self.assertLessEqual(abs(coordinate), 0.5 + 1e-9)
            # ...and each segment is flat against one of the three back walls,
            # which is what stops it crossing in front of the trajectory.
            on_a_wall = any(
                abs(start[axis] - walls[axis]) < 1e-9
                and abs(end[axis] - walls[axis]) < 1e-9
                for axis in range(3)
            )
            self.assertTrue(on_a_wall, (start, end))

    def test_the_grid_has_lines_in_both_directions_on_each_wall(self):
        from aoeviewer import projection

        fractions = [-0.5, 0.0, 0.5]
        segments = projection.wall_grid_segments(
            45.0, 25.0, fractions, fractions, fractions)
        # Three walls, two directions, three ticks each.
        self.assertEqual(len(segments), 3 * 2 * 3)

    def test_tick_labels_sit_just_outside_the_box(self):
        from aoeviewer import projection

        fractions = [-0.5, 0.0, 0.5]
        anchors = projection.tick_anchors(45.0, 25.0, fractions, fractions,
                                          fractions)
        self.assertEqual(set(anchors), {"x", "y", "z"})
        for axis, points in anchors.items():
            self.assertEqual(len(points), 3)
            for point in points:
                # Outside on the two axes it is not measuring along, so a label
                # never sits on top of the line it belongs to.
                outside = sum(1 for c in point if abs(c) > 0.5 + 1e-9)
                self.assertEqual(outside, 2, (axis, point))

    def test_a_value_maps_to_the_same_fraction_as_the_array_form(self):
        from aoeviewer import projection

        bounds = (100.0, 900.0)
        for value in (100.0, 300.0, 900.0):
            self.assertAlmostEqual(
                projection.to_fraction(value, bounds),
                float(projection.normalise(np.array([value]), bounds)[0]),
                places=12)

    def test_a_degenerate_axis_maps_to_the_middle(self):
        from aoeviewer import projection

        self.assertEqual(projection.to_fraction(5.0, (5.0, 5.0)), 0.0)
        self.assertEqual(projection.to_fraction(5.0, None), 0.0)

    def test_the_option_survives_a_workspace_round_trip(self):
        from aoeviewer.model import TileState

        tile = TileState()
        self.assertTrue(tile.show_3d_grid)   # on by default
        tile.show_3d_grid = False
        self.assertFalse(TileState.from_dict(tile.to_dict()).show_3d_grid)

    def test_an_older_workspace_gets_the_grid(self):
        from aoeviewer.model import TileState

        data = TileState().to_dict()
        data.pop("show_3d_grid")
        self.assertTrue(TileState.from_dict(data).show_3d_grid)


class TestManyRunsScaling(unittest.TestCase):
    """A study of a thousand runs has to stay usable.

    1200 runs x 32 parameters, 0 to 15 s every 0.03 s, is 19 million samples --
    which the core handles in milliseconds. What did not scale was the front
    end: one plot item per curve meant 2400 Qt items for two parameters, and
    the cost of an item is a bounding rect, a paint call and a scene-graph node
    each, regardless of how few points it holds.
    """

    runs = 60
    params = 4
    rows = 51

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.mkdtemp()
        cls.path = os.path.join(cls.directory, "many.aoe")
        names = ["TIME"] + ["P%d" % i for i in range(1, cls.params)]
        with open(cls.path, "w") as handle:
            handle.write("*PARAMETRES\n " + " ".join("'%s'" % n for n in names) + "\n")
            for run in range(cls.runs):
                handle.write("*TABLEAU : RUN_%03d\n" % run)
                for row in range(cls.rows):
                    t = row * 0.03
                    values = [t] + [math.sin(t * (1 + c)) + run * 0.01
                                    for c in range(1, cls.params)]
                    handle.write(" " + " ".join("%.5f" % v for v in values) + "\n")

    def setUp(self):
        self.session = Session()
        self.file = self.session.add_file(self.path)

    def tearDown(self):
        self.session.close()

    def test_the_file_reads_correctly(self):
        self.assertEqual(self.file.simulation_count, self.runs)
        values = self.session.channel(0, 0, "TIME")
        self.assertEqual(values.size, self.rows)

    def test_run_handles_are_cached(self):
        first = self.file.simulations
        self.assertIs(first, self.file.simulations)
        # Building one handle costs three calls across the C boundary, and this
        # is read once per file per tile on every redraw.
        self.assertEqual(len(first), self.runs)

    def test_the_cache_is_dropped_when_the_files_change(self):
        first = self.file.simulations
        self.session.add_file(self.path)
        self.assertIsNot(first, self.file.simulations)

    def test_the_cache_is_dropped_on_reload(self):
        first = self.file.simulations
        self.file.reload()
        self.assertIsNot(first, self.file.simulations)

    def test_the_cache_survives_a_rename_of_a_run(self):
        handles = self.file.simulations
        handles[0].rename("first run")
        # Renaming a run updates the handle in place rather than invalidating
        # every handle of the file.
        self.assertIs(handles, self.file.simulations)
        self.assertEqual(self.file.simulations[0].name, "first run")

    def _grouped(self, color_mode):
        """Curves grouped by pen, the way the tile draws them."""
        from aoeviewer.model import AppState

        state = AppState()
        state.tile.color_mode = color_mode
        styles = state.styles
        groups = {}
        for run in self.file.simulations:
            for param in ("P1", "P2"):
                key = ("param::%s" % param if color_mode == 1
                       else "sim::%s::%s" % (run.key, param))
                pen = (styles.color_for(key), styles.dash_for(key),
                       styles.width_for(key))
                groups.setdefault(pen, []).append((run.index, param))
        return groups

    def test_curves_collapse_to_a_handful_of_pens(self):
        from aoeviewer.model import ColorMode, PALETTE

        by_sim = self._grouped(ColorMode.SIMULATION)
        drawn = self.runs * 2
        # The palette cycles, so a thousand runs cannot produce a thousand pens.
        self.assertLessEqual(len(by_sim), len(PALETTE) * 4)
        self.assertLess(len(by_sim), drawn // 4)

    def test_colouring_by_parameter_gives_one_pen_per_parameter(self):
        from aoeviewer.model import ColorMode

        by_param = self._grouped(ColorMode.PARAMETER)
        self.assertEqual(len(by_param), 2)

    def test_concatenating_a_group_preserves_every_sample(self):
        gap = np.array([np.nan], dtype=np.float32)
        pieces_x, pieces_y = [], []
        originals = []
        for index in range(5):
            x = self.session.channel(0, index, "TIME")
            y = self.session.channel(0, index, "P1")
            originals.append((np.array(x), np.array(y)))
            if pieces_x:
                pieces_x.append(gap)
                pieces_y.append(gap)
            pieces_x.append(x)
            pieces_y.append(y)

        cx = np.concatenate(pieces_x)
        cy = np.concatenate(pieces_y)

        # 5 runs, 4 separators.
        self.assertEqual(cx.size, 5 * self.rows + 4)

        # Each run comes back out intact, and a NaN sits between consecutive
        # runs so connect="finite" breaks the line rather than drawing a stroke
        # from the end of one run to the start of the next.
        position = 0
        for run_index, (ox, oy) in enumerate(originals):
            if run_index:
                self.assertTrue(np.isnan(cx[position]))
                self.assertTrue(np.isnan(cy[position]))
                position += 1
            np.testing.assert_array_equal(cx[position:position + self.rows], ox)
            np.testing.assert_array_equal(cy[position:position + self.rows], oy)
            position += self.rows

    def test_every_curve_stays_individually_addressable(self):
        # Merging is a drawing optimisation only. Click-to-inspect, the
        # crosshair readout, marker dots and the CSV export all still need to
        # answer "which run and which parameter is this", so the per-curve
        # model must survive the merge.
        keys = set()
        for run in self.file.simulations:
            for param in ("P1", "P2"):
                keys.add("{0}|{1}|{2}".format(0, run.index, param))
        self.assertEqual(len(keys), self.runs * 2)

    def test_loading_every_run_is_quick(self):
        import time

        sims = list(range(self.file.simulation_count))
        start = time.perf_counter()
        self.session.preload(0, sims, ["TIME", "P1", "P2"])
        elapsed = time.perf_counter() - start
        # Generous, because CI machines vary; it is a guard against an
        # accidental per-run file re-open, which is what this used to be.
        self.assertLess(elapsed, 5.0)

    def test_decimating_every_curve_is_quick(self):
        import time

        sims = list(range(self.file.simulation_count))
        self.session.preload(0, sims, ["TIME", "P1", "P2"])
        curves = [(self.session.channel(0, i, "TIME"),
                   self.session.channel(0, i, p))
                  for i in sims for p in ("P1", "P2")]
        start = time.perf_counter()
        for x, y in curves:
            self.session.decimate(x, y, 600, (0.0, 1.5))
        self.assertLess(time.perf_counter() - start, 5.0)


class TestCache(unittest.TestCase):
    """The `.aoec` sidecar, through the bindings the front end actually uses.

    The C++ suite covers the format itself in detail. What is worth checking
    from here is that the four entry points are wired up, that a cached session
    hands back the same numbers as an uncached one, and -- the part that would
    hurt a user -- that editing the file makes the cache stop being used.
    """

    sims = 4
    rows = 400

    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".aoe", prefix="aoe_cache_")
        os.close(handle)
        generate(self.path, self.sims, self.rows)
        self.sidecar = self.path + ".aoec"
        self._sessions = []

    def tearDown(self):
        for session in self._sessions:
            session.close()
        for path in (self.path, self.sidecar):
            try:
                os.remove(path)
            except OSError:
                pass

    def _open(self):
        session = Session()
        self._sessions.append(session)
        session.add_file(self.path)
        return session

    def _read_all(self, session):
        """Every sample of every channel, as a flat list of tuples."""
        out = []
        for sim in range(session.file(0).simulation_count):
            for name in session.parameters:
                values = session.channel(0, sim, name)
                out.append((sim, name, None if values is None else values.tolist()))
        return out

    def test_no_cache_until_one_is_built(self):
        session = self._open()
        self.assertFalse(session.cache_active(0))
        self.assertEqual(session.cache_path(0), "")
        self.assertFalse(os.path.exists(self.sidecar))

    def test_building_a_cache_makes_the_next_open_use_it(self):
        first = self._open()
        self.assertTrue(first.build_cache(0), first.last_error)
        self.assertTrue(first.cache_active(0))
        self.assertTrue(os.path.exists(self.sidecar))
        self.assertEqual(first.cache_path(0), self.sidecar)

        second = self._open()
        self.assertTrue(second.cache_active(0))

    def test_cached_values_are_identical(self):
        plain = self._open()
        before = self._read_all(plain)
        self.assertTrue(plain.build_cache(0), plain.last_error)

        cached = self._open()
        self.assertTrue(cached.cache_active(0))
        self.assertEqual(before, self._read_all(cached))

    def test_editing_the_file_retires_the_cache(self):
        session = self._open()
        self.assertTrue(session.build_cache(0), session.last_error)

        # Same length, one digit different: the case a size check alone misses.
        with open(self.path, "r+b") as handle:
            raw = bytearray(handle.read())
            for index in range(len(raw) // 2, len(raw)):
                if 0x30 <= raw[index] <= 0x38:
                    raw[index] += 1
                    break
            else:
                self.fail("no digit to edit in the sample file")
            handle.seek(0)
            handle.write(raw)
            handle.truncate()

        after = self._open()
        self.assertFalse(after.cache_active(0))
        # And it still reads, from the text, with the new value in place.
        self.assertIsNotNone(after.channel(0, 0, "TIME"))

    def test_truncated_sidecar_is_ignored(self):
        session = self._open()
        self.assertTrue(session.build_cache(0), session.last_error)

        with open(self.sidecar, "r+b") as handle:
            handle.truncate(os.path.getsize(self.sidecar) // 3)

        after = self._open()
        self.assertFalse(after.cache_active(0))
        self.assertIsNotNone(after.channel(0, 0, "ALPHA"))

    def test_discarding_removes_the_file(self):
        session = self._open()
        self.assertTrue(session.build_cache(0), session.last_error)
        self.assertTrue(os.path.exists(self.sidecar))

        self.assertTrue(session.discard_cache(0))
        self.assertFalse(session.cache_active(0))
        self.assertFalse(os.path.exists(self.sidecar))
        # Still perfectly usable without one.
        self.assertIsNotNone(session.channel(0, 0, "TIME"))

    def test_cache_enabled_is_reported(self):
        self.assertIsInstance(self._open().cache_enabled, bool)


class TestUiScale(unittest.TestCase):
    """One knob for the size of everything the window draws.

    The base type scale is a compromise chosen for a 1080-high screen, and
    screens are not all that. What must not change is that the knob moves
    *everything*: text set smaller inside unchanged padding looks shrunken
    rather than roomier.
    """

    def setUp(self):
        from aoeviewer import theme

        self.theme = theme
        self.addCleanup(theme.set_ui_scale, 1.0)
        theme.set_ui_scale(1.0)

    def test_it_starts_at_one(self):
        self.assertEqual(self.theme.ui_scale(), 1.0)
        self.assertEqual(self.theme.px(10), 10)
        self.assertEqual(self.theme.pt(7.5), 7.5)

    def test_it_scales_pixels_and_points(self):
        self.theme.set_ui_scale(1.4)
        self.assertEqual(self.theme.px(10), 14)
        self.assertEqual(self.theme.pt(7.5), 10.5)

    def test_a_pixel_figure_never_rounds_away_to_nothing(self):
        # A 1 px padding at 0.7 rounds to 0.7 -> 1, not to 0: a border or a
        # gap that disappears is a layout that looks broken rather than tight.
        self.theme.set_ui_scale(self.theme.UI_SCALE_MIN)
        self.assertGreaterEqual(self.theme.px(1), 1)
        self.assertGreaterEqual(self.theme.px(2), 1)

    def test_it_clamps_rather_than_refuses(self):
        # The buttons disable themselves at the ends, so clamping is only ever
        # reached by a workspace from another version -- which must still open.
        self.assertEqual(self.theme.set_ui_scale(99.0), self.theme.UI_SCALE_MAX)
        self.assertEqual(self.theme.set_ui_scale(0.01), self.theme.UI_SCALE_MIN)

    def test_the_stylesheet_moves_with_it(self):
        dark = self.theme.theme_for(True)
        base = self.theme.stylesheet(dark)
        self.assertIn("font-size: 10px", base)

        self.theme.set_ui_scale(1.4)
        bigger = self.theme.stylesheet(dark)
        self.assertIn("font-size: 14px", bigger)
        self.assertNotIn("font-size: 10px", bigger)

    def test_the_padding_moves_with_the_text(self):
        dark = self.theme.theme_for(True)
        base = self.theme.stylesheet(dark)
        self.theme.set_ui_scale(1.5)
        bigger = self.theme.stylesheet(dark)
        # A button's padding at 1.0 is 4px 9px; at 1.5 it must not still be.
        self.assertIn("padding: 4px 9px", base)
        self.assertNotIn("padding: 4px 9px", bigger)
        self.assertIn("padding: 6px 14px", bigger)

    def test_plot_point_sizes_move_with_it(self):
        # What plot_font() and the pyqtgraph size strings are both built from.
        # The font itself needs Qt, so the tile side of this is checked in
        # test_redraw, against the stubs.
        self.theme.set_ui_scale(1.2)
        self.assertAlmostEqual(self.theme.pt(self.theme.PLOT_TITLE_PT), 9.0)
        self.assertAlmostEqual(self.theme.pt(self.theme.PLOT_LEGEND_PT), 7.2)
        self.assertAlmostEqual(self.theme.pt(self.theme.RAIL_FONT_PT), 7.8)

    def test_the_scale_is_saved_in_the_workspace(self):
        from aoeviewer.model import AppState, load_workspace, save_workspace

        state = AppState()
        state.ui_scale = 1.3
        handle, path = tempfile.mkstemp(suffix=".aoews")
        os.close(handle)
        try:
            save_workspace(path, state, [], {}, {})
            data = load_workspace(path)
            self.assertAlmostEqual(data["ui_scale"], 1.3)
            restored = AppState()
            restored.load_dict(data)
            self.assertAlmostEqual(restored.ui_scale, 1.3)
        finally:
            os.remove(path)

    def test_a_workspace_without_it_opens_at_one(self):
        from aoeviewer.model import AppState

        state = AppState()
        state.load_dict({"dark_mode": True})
        self.assertEqual(state.ui_scale, 1.0)

        state.load_dict({"ui_scale": "not a number"})
        self.assertEqual(state.ui_scale, 1.0)


class TestBuildSamples(unittest.TestCase):
    """The build writes sample/ beside build/ and dist/.

    A few megabytes exercises none of what this program is built around: the
    sidecar is not written below 8 MB, decimation has nothing to decimate, and
    a six-run file has no sweep to filter. So the real ones are large, which is
    also why they are generated once and never rewritten.
    """

    def setUp(self):
        import shutil

        import build_exe

        self.build_exe = build_exe
        self.shutil = shutil
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.makedirs(os.path.join(self.tmp, "tools"), exist_ok=True)
        shutil.copy(os.path.join(root, "tools", "gen_flight_data.py"),
                    os.path.join(self.tmp, "tools"))

        self._project = build_exe.PROJECT
        self._samples = build_exe.SAMPLES
        self.addCleanup(setattr, build_exe, "PROJECT", self._project)
        self.addCleanup(setattr, build_exe, "SAMPLES", self._samples)
        build_exe.PROJECT = self.tmp
        # The same code path, at a size a test can afford.
        build_exe.SAMPLES = (
            ("tiny_a.aoe", {"runs": 3, "size_mb": 0.3, "seed": 7}),
            ("tiny_b.aoe", {"runs": 3, "size_mb": 0.3, "seed": 9}),
        )
        self.folder = os.path.join(self.tmp, "sample")

    def names(self):
        return sorted(os.listdir(self.folder)) if os.path.isdir(self.folder) else []

    def test_the_real_samples_are_big_enough_to_be_worth_having(self):
        # Guards the sizes rather than the mechanism: below the 8 MB sidecar
        # threshold these would exercise none of the machinery they exist for.
        sizes = [spec["size_mb"] for _name, spec in self._samples]
        self.assertGreaterEqual(max(sizes), 100.0)
        self.assertGreaterEqual(sum(1 for s in sizes if s >= 8.0), 2,
                                "at least two samples must cross the sidecar "
                                "threshold, or the pair has nothing to show")

    def test_it_writes_what_is_missing(self):
        self.build_exe.ensure_samples(quiet=True)
        self.assertEqual(self.names(), ["tiny_a.aoe", "tiny_b.aoe"])
        for name in self.names():
            self.assertGreater(os.path.getsize(os.path.join(self.folder, name)), 0)

    def test_the_files_are_readable_by_the_core(self):
        self.build_exe.ensure_samples(quiet=True)
        session = Session()
        try:
            handle = session.add_file(os.path.join(self.folder, "tiny_a.aoe"))
            self.assertEqual(handle.simulation_count, 3)
            self.assertIn("TIME", session.parameters)
        finally:
            session.close()

    def test_it_never_rewrites_a_file_that_is_there(self):
        self.build_exe.ensure_samples(quiet=True)
        marker = os.path.join(self.folder, "tiny_a.aoe")
        with open(marker, "w", encoding="ascii") as handle:
            handle.write("edited by hand\n")

        self.build_exe.ensure_samples(quiet=True)
        with open(marker, encoding="ascii") as handle:
            self.assertEqual(handle.read(), "edited by hand\n")

    def test_only_the_missing_one_is_written(self):
        self.build_exe.ensure_samples(quiet=True)
        os.remove(os.path.join(self.folder, "tiny_b.aoe"))
        kept = os.path.getmtime(os.path.join(self.folder, "tiny_a.aoe"))

        self.build_exe.ensure_samples(quiet=True)
        self.assertEqual(self.names(), ["tiny_a.aoe", "tiny_b.aoe"])
        self.assertEqual(os.path.getmtime(os.path.join(self.folder, "tiny_a.aoe")),
                         kept)

    def test_an_interrupted_run_leaves_nothing_that_looks_finished(self):
        # Ctrl+C halfway through must not leave a truncated file that the next
        # build counts as "already there" and the viewer then opens.
        import gen_flight_data

        real = gen_flight_data.write_file
        calls = []

        def explode(path, *args, **kwargs):
            calls.append(path)
            if len(calls) == 2:
                with open(path, "w", encoding="ascii") as handle:
                    handle.write("half")
                raise KeyboardInterrupt
            return real(path, *args, **kwargs)

        gen_flight_data.write_file = explode
        self.addCleanup(setattr, gen_flight_data, "write_file", real)
        try:
            self.build_exe.ensure_samples(quiet=True)
        except KeyboardInterrupt:
            self.fail("the build must not die with the generator")

        self.assertEqual(self.names(), ["tiny_a.aoe"])

    def test_a_generator_that_cannot_be_imported_does_not_fail_the_build(self):
        os.remove(os.path.join(self.tmp, "tools", "gen_flight_data.py"))
        import gen_flight_data

        self.addCleanup(sys.modules.__setitem__, "gen_flight_data", gen_flight_data)
        del sys.modules["gen_flight_data"]
        saved = list(sys.path)
        sys.path[:] = [p for p in sys.path if not p.endswith("tools")]
        self.addCleanup(sys.path.__setitem__, slice(None), saved)

        # A sample file is a convenience, not part of the product.
        self.build_exe.ensure_samples(quiet=True)
        self.assertEqual(self.names(), [])
