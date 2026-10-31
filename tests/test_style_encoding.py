# Author: Paulo R. A. Gilz
"""Dual encoding: colour key, line key, titles, PARAMS order, workspaces."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import visimu
from visimu.model import (
    COLOR_MODE_LABELS,
    AppState,
    ColorMode,
    StyleMode,
    StyleRegistry,
    TileState,
    curve_color_key,
    curve_line_key,
    load_workspace,
    order_parameters,
    save_workspace,
    tile_title,
)


def _tile(**kwargs):
    tile = TileState()
    for key, value in kwargs.items():
        setattr(tile, key, value)
    return tile


class TestCurveKeys(unittest.TestCase):
    def test_colour_by_parameter(self):
        tile = _tile(color_mode=ColorMode.PARAMETER)
        self.assertEqual(curve_color_key(tile, "ALPHA", "s1", "0.8"),
                         "param::ALPHA")

    def test_colour_by_iso(self):
        tile = _tile(color_mode=ColorMode.ISO, color_iso_param="MACH")
        self.assertEqual(curve_color_key(tile, "ALPHA", "s1", "0.8"),
                         "iso::MACH::0.8")

    def test_colour_by_parameter_is_the_default(self):
        # A tile is nearly always asked which quantity a line is, and colouring
        # by run answers a different question -- one a twenty-five-run study
        # answers with twenty-five colours that name nothing the user chose.
        tile = _tile()
        self.assertEqual(tile.color_mode, ColorMode.PARAMETER)
        self.assertEqual(curve_color_key(tile, "ALPHA", "s1", "0.8"),
                         "param::ALPHA")

    def test_style_by_file_is_independent_of_colour(self):
        tile = _tile(color_mode=ColorMode.PARAMETER, style_mode=StyleMode.FILE)
        self.assertEqual(curve_color_key(tile, "ALPHA", "s1", ""), "param::ALPHA")
        self.assertEqual(
            curve_line_key(tile, "ALPHA", "s1", "/data/a.aoe", "", ""),
            "file::/data/a.aoe")

    def test_style_none_follows_the_colour_key(self):
        tile = _tile(color_mode=ColorMode.PARAMETER, style_mode=StyleMode.NONE)
        self.assertEqual(
            curve_line_key(tile, "NZ", "s1", "/data/a.aoe", "0.8", "0.8"),
            "param::NZ")

    def test_style_by_iso_uses_its_own_parameter(self):
        tile = _tile(style_mode=StyleMode.ISO, style_iso_param="MASS")
        self.assertEqual(
            curve_line_key(tile, "ALPHA", "s1", "f", "0.8", "180"),
            "iso::MASS::180")


class TestLineAssignment(unittest.TestCase):
    def test_assign_is_sticky(self):
        styles = StyleRegistry()
        styles.assign_line_style("file::a", 0)
        styles.assign_line_style("file::b", 1)
        first = styles.dash_for("file::a")
        second = styles.dash_for("file::b")
        self.assertNotEqual(first, second)
        styles.assign_line_style("file::a", 4)
        self.assertEqual(styles.dash_for("file::a"), first)


class TestTileTitleWrap(unittest.TestCase):
    def test_short_title_stays_on_one_line(self):
        self.assertEqual(tile_title(_tile(x_param="TIME", y_params=["ALPHA"])),
                         "ALPHA = f(TIME)")

    def test_every_name_is_kept(self):
        title = tile_title(_tile(
            x_param="TIME",
            y_params=["ALPHA", "BETA", "NZ", "Q", "MACH", "ALT"]))
        for name in ("ALPHA", "BETA", "NZ", "Q", "MACH", "ALT"):
            self.assertIn(name, title)
        self.assertIn("\n", title)
        self.assertNotIn("more", title)


class TestOrderParameters(unittest.TestCase):
    def test_selected_come_first_in_selection_order(self):
        self.assertEqual(
            order_parameters(["TIME", "ALT", "MACH", "NZ"], ["NZ", "ALT"]),
            ["NZ", "ALT", "TIME", "MACH"])

    def test_unknown_selected_names_are_ignored(self):
        self.assertEqual(
            order_parameters(["A", "B"], ["Z", "B"]),
            ["B", "A"])

    def test_empty_selection_keeps_the_given_order(self):
        names = ["TIME", "ALT", "MACH"]
        self.assertEqual(order_parameters(names, []), names)


class TestWorkspaceStyleFields(unittest.TestCase):
    def test_style_encoding_round_trips(self):
        state = AppState()
        state.tile.color_mode = ColorMode.PARAMETER
        state.tile.style_mode = StyleMode.FILE
        state.tile.style_iso_param = "MACH"
        state.styles.assign_line_style("file::/a.aoe", 1)
        handle, path = tempfile.mkstemp(suffix=".aoews")
        os.close(handle)
        try:
            save_workspace(path, state, ["/a.aoe"], {}, {})
            data = load_workspace(path)
            self.assertEqual(data["version"], 3)
            # The schema is what a reader has to understand; app_version only
            # says which build wrote the file, so it tracks __version__ rather
            # than being pinned here.
            self.assertEqual(data["app_version"], visimu.__version__)
            restored = AppState()
            restored.load_dict(data)
            self.assertEqual(restored.tile.color_mode, ColorMode.PARAMETER)
            self.assertEqual(restored.tile.style_mode, StyleMode.FILE)
            self.assertEqual(restored.tile.style_iso_param, "MACH")
            self.assertTrue(restored.styles.has_dash("file::/a.aoe"))
        finally:
            os.unlink(path)

    def test_old_workspaces_default_style_to_none(self):
        state = AppState()
        payload = state.to_dict()
        for page in payload["pages"]:
            for tile in page["tiles"]:
                tile.pop("style_mode", None)
                tile.pop("style_iso_param", None)
        restored = AppState()
        restored.load_dict(payload)
        self.assertEqual(restored.tile.style_mode, StyleMode.NONE)
        self.assertEqual(restored.tile.style_iso_param, "")


if __name__ == "__main__":
    unittest.main()


class TestWorkspacesFromEveryEarlierVersion(unittest.TestCase):
    """A workspace file written by an older build must still open.

    These files are the user's work -- a grid of thirty tiles, every curve
    styled, the ISO filters set up -- so losing one to a schema change is
    losing an afternoon. The reader is tolerant by construction: it asks
    `data.get(name, default)` for everything and never reads the version
    number. This pins that down version by version, because tolerance is the
    kind of property that is true until someone adds a required field.

    Each dictionary below is what that version actually wrote, trimmed to the
    keys that matter. The assertions are about the fields those versions had
    never heard of: they must arrive at their documented defaults rather than
    None, and nothing may raise.
    """

    def _load(self, data):
        handle, path = tempfile.mkstemp(suffix=".aoews")
        os.close(handle)
        self.addCleanup(os.remove, path)
        import json

        with open(path, "w") as fh:
            json.dump(data, fh)
        state = AppState()
        state.load_dict(load_workspace(path))
        return state

    def test_a_version_1_workspace_loads(self):
        state = self._load({
            "version": 1,
            "pages": [{"name": "Page 1", "rows": 1, "cols": 1,
                       "tiles": [{"y_params": ["ALPHA"]}]}],
            "selected_sims": [[0, 0]],
        })
        tile = state.pages[0].tiles[0]
        self.assertEqual(tile.y_params, ["ALPHA"])
        self.assertEqual(state.selected_sims, {(0, 0)})

    def test_a_version_2_workspace_loads(self):
        state = self._load({
            "version": 2,
            "app_version": "2.0.0",
            "pages": [{"name": "Sweep", "rows": 2, "cols": 2,
                       "tiles": [{"y_params": ["ALPHA", "MACH"],
                                  "legend": True,
                                  "show_3d_box": True}]}],
            "colors": {"ALPHA": "#ff0000"},
            "active_page": 0,
        })
        page = state.pages[0]
        self.assertEqual(page.name, "Sweep")
        tile = page.tiles[0]
        self.assertEqual(tile.y_params, ["ALPHA", "MACH"])
        # The old key for the 3D reference, honoured rather than dropped.
        self.assertTrue(tile.show_3d_axes)

    def test_a_workspace_without_the_sync_fields_defaults_them_off(self):
        """SYNC arrived in 2.2.0; every workspace before it has no such keys.

        Off, with the condition and target at the values a fresh tile uses, so
        that opening an old workspace and ticking Align does the obvious thing
        rather than starting from a half-set state.
        """
        state = self._load({
            "version": 3,
            "app_version": "2.1.2",
            "pages": [{"name": "Page 1", "rows": 1, "cols": 1,
                       "tiles": [{"y_params": ["SPEED"]}]}],
        })
        tile = state.pages[0].tiles[0]
        self.assertFalse(tile.sync_enabled)
        self.assertEqual(tile.sync_param, "")
        self.assertEqual(tile.sync_condition, "above")
        self.assertEqual(tile.sync_level, 0.0)
        self.assertEqual(tile.sync_target, "zero")
        self.assertEqual(tile.sync_target_value, 0.0)

    def test_an_empty_workspace_is_not_an_error(self):
        state = self._load({})
        self.assertEqual(len(state.pages), 1)

    def test_an_unknown_future_field_is_ignored_rather_than_fatal(self):
        # The same tolerance in the other direction: a workspace written by a
        # later build should open here, minus whatever this build cannot do.
        state = self._load({
            "version": 99,
            "pages": [{"name": "Page 1", "rows": 1, "cols": 1,
                       "tiles": [{"y_params": ["ALPHA"],
                                  "something_invented_later": 42}]}],
            "a_whole_new_section": {"x": 1},
        })
        self.assertEqual(state.pages[0].tiles[0].y_params, ["ALPHA"])

    def test_what_this_build_writes_still_reads_back(self):
        state = AppState()
        tile = state.pages[0].tiles[0]
        tile.y_params = ["ALPHA"]
        tile.sync_enabled = True
        tile.sync_param = "SPEED"
        tile.sync_condition = "rises"
        tile.sync_level = 0.5

        handle, path = tempfile.mkstemp(suffix=".aoews")
        os.close(handle)
        self.addCleanup(os.remove, path)
        save_workspace(path, state, [], {}, {})

        back = AppState()
        back.load_dict(load_workspace(path))
        tile = back.pages[0].tiles[0]
        self.assertEqual(tile.y_params, ["ALPHA"])
        self.assertTrue(tile.sync_enabled)
        self.assertEqual(tile.sync_param, "SPEED")
        self.assertEqual(tile.sync_condition, "rises")
        self.assertEqual(tile.sync_level, 0.5)


class TestColourByFile(unittest.TestCase):
    """One colour per result file, shared by every parameter from it.

    The question a tile answers when several studies are open at once is
    often "which file is this line from", and until now that could only be
    said with the dash. Colour says it now as well.
    """

    def test_it_keys_on_the_file(self):
        tile = _tile(color_mode=ColorMode.FILE)
        self.assertEqual(
            curve_color_key(tile, "ALPHA", "s1", "0.8", "/data/a.aoe"),
            "file::/data/a.aoe")

    def test_every_parameter_of_a_file_shares_one_colour(self):
        tile = _tile(color_mode=ColorMode.FILE)
        first = curve_color_key(tile, "ALPHA", "s1", "", "/data/a.aoe")
        second = curve_color_key(tile, "NZ", "s2", "", "/data/a.aoe")
        self.assertEqual(first, second)

    def test_two_files_are_two_colours(self):
        tile = _tile(color_mode=ColorMode.FILE)
        self.assertNotEqual(
            curve_color_key(tile, "ALPHA", "s1", "", "/data/a.aoe"),
            curve_color_key(tile, "ALPHA", "s2", "", "/data/b.aoe"))

    def test_it_keys_on_the_path_not_the_name(self):
        """Two studies called results.aoe in different directories are two
        files, and must not collapse onto one colour."""
        tile = _tile(color_mode=ColorMode.FILE)
        self.assertNotEqual(
            curve_color_key(tile, "ALPHA", "s1", "", "/june/results.aoe"),
            curve_color_key(tile, "ALPHA", "s1", "", "/july/results.aoe"))

    def test_without_a_file_it_falls_back_to_the_run(self):
        """Keyed by the run, but still as a file identity.

        The same fallback Style-by-File uses -- a caller with no file to offer
        gets one identity per run rather than one for everything, and the two
        modes still produce the same key for the same curve.
        """
        tile = _tile(color_mode=ColorMode.FILE)
        self.assertEqual(curve_color_key(tile, "ALPHA", "s1", ""), "file::s1")
        styled = _tile(color_mode=ColorMode.PARAMETER,
                       style_mode=StyleMode.FILE)
        self.assertEqual(curve_line_key(styled, "ALPHA", "s1", "", "", ""),
                         "file::s1")

    def test_colour_and_style_by_file_agree(self):
        coloured = _tile(color_mode=ColorMode.FILE)
        styled = _tile(color_mode=ColorMode.PARAMETER,
                       style_mode=StyleMode.FILE)
        self.assertEqual(
            curve_color_key(coloured, "ALPHA", "s1", "", "/data/a.aoe"),
            curve_line_key(styled, "ALPHA", "s1", "/data/a.aoe", "", ""))

    def test_style_none_follows_a_file_colour_key(self):
        # Style by None means "dash follows the colour key", and the colour
        # key now has a file form it has to be able to reach.
        tile = _tile(color_mode=ColorMode.FILE, style_mode=StyleMode.NONE)
        self.assertEqual(
            curve_line_key(tile, "ALPHA", "s1", "/data/a.aoe", "", ""),
            "file::/data/a.aoe")

    def test_the_other_modes_ignore_the_file(self):
        for mode, expected in ((ColorMode.PARAMETER, "param::ALPHA"),
                               (ColorMode.SIMULATION, "sim::s1")):
            tile = _tile(color_mode=mode)
            self.assertEqual(
                curve_color_key(tile, "ALPHA", "s1", "0.8", "/data/a.aoe"),
                expected, mode)


class TestColourModeNumbersAreFixed(unittest.TestCase):
    """The mode is stored in workspaces as a number, so the numbers are an
    interface. A new one is appended; none is ever inserted or renumbered."""

    def test_the_existing_numbers_have_not_moved(self):
        self.assertEqual(int(ColorMode.SIMULATION), 0)
        self.assertEqual(int(ColorMode.PARAMETER), 1)
        self.assertEqual(int(ColorMode.ISO), 2)

    def test_file_was_appended(self):
        self.assertEqual(int(ColorMode.FILE), 3)

    def test_an_old_workspace_still_means_what_it_said(self):
        for number, mode in ((0, ColorMode.SIMULATION),
                             (1, ColorMode.PARAMETER),
                             (2, ColorMode.ISO)):
            tile = TileState.from_dict({"color_mode": number})
            self.assertEqual(tile.color_mode, mode)

    def test_a_file_coloured_tile_round_trips(self):
        tile = TileState()
        tile.color_mode = ColorMode.FILE
        back = TileState.from_dict(tile.to_dict())
        self.assertEqual(back.color_mode, ColorMode.FILE)

    def test_a_mode_this_version_does_not_know_loads_as_parameter(self):
        """A workspace from a later build can name a mode that does not exist
        here. The rest of the tile is still worth having, so it is not an
        error -- the same tolerance style_mode already had."""
        tile = TileState.from_dict({"color_mode": 99, "y_params": ["ALPHA"]})
        self.assertEqual(tile.color_mode, ColorMode.PARAMETER)
        self.assertEqual(tile.y_params, ["ALPHA"])

    def test_rubbish_is_tolerated_too(self):
        tile = TileState.from_dict({"color_mode": "banana"})
        self.assertEqual(tile.color_mode, ColorMode.PARAMETER)

    def test_every_mode_has_a_label(self):
        for mode in ColorMode:
            self.assertIn(mode, COLOR_MODE_LABELS)
            self.assertTrue(COLOR_MODE_LABELS[mode])
