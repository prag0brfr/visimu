# Author: Paulo R. A. Gilz
"""Dual encoding: colour key, line key, titles, PARAMS order, workspaces."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import aoeviewer
from aoeviewer.model import (
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
            self.assertEqual(data["app_version"], aoeviewer.__version__)
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
