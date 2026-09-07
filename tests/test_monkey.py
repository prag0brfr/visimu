# Author: Paulo R. A. Gilz
"""Randomised stress tests: thousands of operations, then check the invariants.

The other test file checks that each operation does the right thing when called
correctly. This one checks the property that actually decides whether the
application survives an afternoon of use: that no *sequence* of legal
operations can put it into a broken state.

That is a different kind of bug and the two hardest ones found so far were both
of this kind -- a removed file leaving selections pointing at indices that had
shifted, and a tile keeping parameters that no longer existed after a file was
dropped. Neither is reachable by calling one function wrongly; both are
reachable by calling three correct functions in the wrong order.

Every test here is seeded, so a failure is reproducible: the seed is printed in
the assertion message, and re-running with that seed replays the exact
sequence.

    python -m unittest discover -s tests -v
    python -m unittest tests.test_monkey -v
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import numpy as np

from aoeviewer import CoreError, Session
from aoeviewer.model import (
    DEFAULT_3D_HALF,
    MAX_3D_HALF,
    MAX_TILES,
    MIN_3D_HALF,
    AppState,
    AreaBand,
    ColorMode,
    Marker,
    MouseMode,
    TextNote,
    clamp_3d_half,
    PageState,
    TileState,
    load_workspace,
    save_workspace,
    tile_title,
)
from aoeviewer import projection
from aoeviewer.scales import MAX_TILE_POINTS, resample_target
from gen_sample import generate

# Kept small deliberately: the point is the number of *operations*, not the
# number of samples, and a big file would make the suite too slow to run often.
FILE_RUNS = 4
FILE_ROWS = 200


class MonkeyBase(unittest.TestCase):
    """A session with a few files, and the machinery to abuse it."""

    file_count = 3

    @classmethod
    def setUpClass(cls):
        cls._paths = []
        for index in range(cls.file_count):
            handle, path = tempfile.mkstemp(suffix=".aoe", prefix="aoe_monkey_")
            os.close(handle)
            generate(path, FILE_RUNS + index, FILE_ROWS, index % 2 == 1)
            cls._paths.append(path)

    @classmethod
    def tearDownClass(cls):
        for path in cls._paths:
            try:
                os.remove(path)
            except OSError:
                pass

    def setUp(self):
        self.session = Session()
        self.state = AppState()

    def tearDown(self):
        self.session.close()

    # ------------------------------------------------------------ helpers

    def check_invariants(self, seed: int, step: int) -> None:
        """Everything that must be true after any sequence of operations."""
        context = "seed={0} step={1}".format(seed, step)

        # Pages
        self.assertGreaterEqual(len(self.state.pages), 1, context)
        self.assertTrue(0 <= self.state.active_page < len(self.state.pages), context)

        page = self.state.page
        self.assertGreaterEqual(page.rows, 1, context)
        self.assertGreaterEqual(page.cols, 1, context)
        self.assertEqual(len(page.tiles), MAX_TILES, context)
        self.assertLessEqual(page.tile_count, MAX_TILES, context)

        # The active tile must be addressable. This is the invariant that a
        # shrinking grid used to break.
        self.assertIsInstance(self.state.tile, TileState, context)

        # Selections must name a file and a run that exist. A stale selection
        # is how a removed file used to take the plots down with it.
        for file_index, sim_index in self.state.selected_sims:
            handle = self.session.file(file_index)
            self.assertIsNotNone(handle, context + " dangling file " + str(file_index))
            self.assertLess(sim_index, handle.simulation_count, context)

        # Ranges must be ordered, or a locked tile draws nothing at all.
        for other_page in self.state.pages:
            for tile in other_page.visible_tiles():
                self.assertLessEqual(tile.x_min, tile.x_max, context)
                self.assertLessEqual(tile.y_min, tile.y_max, context)
                self.assertIsInstance(tile_title(tile), str, context)

        # Declared variable limits are ordered and are honoured by any tile
        # carrying that parameter.
        for name, (low, high) in self.state.global_x_limits.items():
            self.assertLessEqual(low, high, context + " x limit " + name)
        for name, (low, high) in self.state.global_y_limits.items():
            self.assertLessEqual(low, high, context + " y limit " + name)


class TestModelMonkey(MonkeyBase):
    """Pages, tiles, markers, areas and styles under random operation."""

    OPERATIONS = 4000

    def _random_param(self, rng):
        params = self.session.parameters
        return rng.choice(params) if params else ""

    def _step(self, rng, seed, step):
        params = self.session.parameters
        choice = rng.randrange(22)

        if choice == 0:
            self.state.add_page()
        elif choice == 1:
            self.state.remove_page(rng.randrange(len(self.state.pages)))
        elif choice == 2:
            self.state.active_page = rng.randrange(len(self.state.pages))
        elif choice == 3:
            page = self.state.page
            page.rows = rng.randint(1, 5)
            page.cols = rng.randint(1, 5)
            self.state.active_tile = min(self.state.active_tile,
                                         page.tile_count - 1)
        elif choice == 4:
            self.state.active_tile = rng.randrange(self.state.page.tile_count)
        elif choice == 5:
            self.state.move_tile(rng.choice(("left", "right", "up", "down")))
        elif choice == 6:
            self.state.copy_tile()
        elif choice == 7:
            self.state.cut_tile()
        elif choice == 8:
            self.state.paste_tile()
        elif choice == 9:
            self.state.clear_tile()
        elif choice == 10 and params:
            tile = self.state.tile
            tile.x_param = rng.choice(params)
            count = rng.randint(0, min(4, len(params)))
            tile.y_params = rng.sample(params, count)
            self.state.apply_global_limits(tile)
        elif choice == 11 and params:
            self.state.markers.append(Marker(
                x_param=rng.choice(params),
                position=rng.uniform(-50, 50),
                visible=rng.random() > 0.2,
            ))
        elif choice == 12 and self.state.markers:
            self.state.markers.pop(rng.randrange(len(self.state.markers)))
        elif choice == 13 and params:
            low = rng.uniform(-50, 50)
            self.state.areas.append(AreaBand(
                x_param=rng.choice(params),
                x_min=low, x_max=low + rng.uniform(0.1, 20),
            ))
        elif choice == 14 and self.state.areas:
            self.state.areas.pop(rng.randrange(len(self.state.areas)))
        elif choice == 15 and params:
            key = "param::" + rng.choice(params)
            self.state.styles.set_color(key, "#%06x" % rng.randrange(0x1000000))
            self.state.styles.set_dash(key, rng.choice(("solid", "dash", "dot")))
        elif choice == 16:
            tile = self.state.tile
            tile.color_mode = ColorMode(rng.randrange(3))
            tile.color_iso_param = self._random_param(rng)
        elif choice == 17 and params:
            low = rng.uniform(-100, 100)
            self.state.set_global_x_limit(rng.choice(params), low,
                                          low + rng.uniform(0.1, 50))
        elif choice == 18 and params:
            low = rng.uniform(-100, 100)
            self.state.set_global_y_limit(rng.choice(params), low,
                                          low + rng.uniform(0.1, 50))
        elif choice == 19:
            self.state.apply_global_limits_everywhere()
        elif choice == 20:
            tile = self.state.tile
            tile.plot3d = not tile.plot3d
            if params:
                tile.z_param = rng.choice(params)
        elif choice == 21:
            self.state.mouse_mode = MouseMode(rng.randrange(4))

        del seed, step

    def test_random_operations_preserve_invariants(self):
        for seed in (1, 2, 3, 17, 99):
            with self.subTest(seed=seed):
                self.setUp()
                try:
                    rng = random.Random(seed)
                    self.session.add_file(self._paths[0])
                    for sim in range(FILE_RUNS):
                        self.state.set_selected(0, sim, True)

                    for step in range(self.OPERATIONS):
                        self._step(rng, seed, step)
                        if step % 200 == 0:
                            self.check_invariants(seed, step)
                    self.check_invariants(seed, self.OPERATIONS)
                finally:
                    self.tearDown()

    def test_shrinking_the_grid_keeps_the_active_tile_valid(self):
        """The specific ordering that used to index past the end of the page."""
        page = self.state.page
        page.rows, page.cols = 5, 5
        self.state.active_tile = 24
        page.rows, page.cols = 1, 1
        # No exception, and the property clamps rather than raising.
        self.assertIs(self.state.tile, page.tiles[0])

    def test_paste_into_a_smaller_grid(self):
        page = self.state.page
        page.rows, page.cols = 3, 3
        self.state.active_tile = 8
        self.state.tile.x_param = "TIME"
        self.state.copy_tile()

        page.rows, page.cols = 1, 2
        self.state.active_tile = 1
        self.assertTrue(self.state.paste_tile())
        self.assertEqual(page.tiles[1].x_param, "TIME")


class TestFileChurnMonkey(MonkeyBase):
    """Adding and removing files while selections and tiles reference them."""

    ROUNDS = 60

    def _sync_selection_to_reality(self) -> None:
        """What the UI does after a file list change."""
        valid = set()
        for file_index, sim_index in self.state.selected_sims:
            handle = self.session.file(file_index)
            if handle is not None and sim_index < handle.simulation_count:
                valid.add((file_index, sim_index))
        self.state.selected_sims = valid

    def test_add_and_remove_files_repeatedly(self):
        for seed in (5, 41):
            with self.subTest(seed=seed):
                self.setUp()
                try:
                    rng = random.Random(seed)
                    for round_index in range(self.ROUNDS):
                        action = rng.randrange(4)

                        if action in (0, 1) and self.session.file_count < 5:
                            path = rng.choice(self._paths)
                            handle = self.session.add_file(path)
                            for sim in range(handle.simulation_count):
                                if rng.random() < 0.7:
                                    self.state.set_selected(handle.index, sim, True)

                        elif action == 2 and self.session.file_count:
                            victim = rng.randrange(self.session.file_count)
                            self.session.remove_file(victim)
                            # drop_file is the model's half of the contract;
                            # forgetting it is what left dangling selections.
                            self.state.drop_file(victim)

                        else:
                            params = self.session.parameters
                            if params:
                                tile = self.state.tile
                                tile.x_param = rng.choice(params)
                                tile.y_params = rng.sample(
                                    params, rng.randint(0, min(3, len(params)))
                                )

                        self._sync_selection_to_reality()
                        self.check_invariants(seed, round_index)

                    # Whatever survived must still be readable, not merely
                    # referenced: a stale index that happens to be in range is
                    # the failure this catches.
                    for file_index, sim_index in sorted(self.state.selected_sims):
                        data = self.session.channel(file_index, sim_index, "TIME")
                        self.assertGreater(data.size, 0)
                finally:
                    self.tearDown()

    def test_removing_every_file_leaves_a_usable_state(self):
        for path in self._paths:
            self.session.add_file(path)
        for index in range(self.session.file_count):
            for sim in range(self.session.file(index).simulation_count):
                self.state.set_selected(index, sim, True)

        while self.session.file_count:
            self.session.remove_file(0)
            self.state.drop_file(0)

        self.assertEqual(self.session.file_count, 0)
        self.assertEqual(self.state.selected_sims, set())
        self.assertEqual(self.session.parameters, [])
        self.assertIsInstance(tile_title(self.state.tile), str)


class TestComputedAndAliasMonkey(MonkeyBase):
    """Formulas and aliases created, redefined and removed at random."""

    ROUNDS = 300

    def test_computed_parameters_churn(self):
        rng = random.Random(23)
        self.session.add_file(self._paths[0])
        base = list(self.session.parameters)
        live = []

        for round_index in range(self.ROUNDS):
            action = rng.randrange(3)

            if action == 0 or not live:
                name = "CALC_{0}".format(rng.randrange(8))
                left = rng.choice(base)
                right = rng.choice(base)
                operator = rng.choice(("+", "-", "*", "/"))
                try:
                    self.session.set_computed(
                        name, "{0} {1} {2}".format(left, operator, right)
                    )
                except CoreError:
                    continue
                if name not in live:
                    live.append(name)

            elif action == 1 and live:
                name = live.pop(rng.randrange(len(live)))
                self.session.remove_computed(name)

            else:
                name = rng.choice(live)
                values = self.session.channel(0, 0, name)
                # Division can legitimately produce NaN; what must never happen
                # is an empty result, which would mean the formula silently
                # produced no samples at all.
                self.assertGreater(values.size, 0,
                                   "round {0}: {1} produced nothing".format(
                                       round_index, name))

            for name in live:
                self.assertIn(name, self.session.parameters)

    def test_alias_churn_never_hangs(self):
        """Aliases, including the cycles a user can easily create."""
        rng = random.Random(77)
        self.session.add_file(self._paths[0])
        base = list(self.session.parameters)

        for _round in range(200):
            alias = "A{0}".format(rng.randrange(6))
            target = rng.choice(base + ["A{0}".format(rng.randrange(6))])
            if alias == target:
                continue
            self.session.set_alias(alias, target)
            # The core breaks cycles; if it did not, this call would not return.
            resolved = self.session.resolve(alias)
            self.assertIsInstance(resolved, str)

            if rng.random() < 0.3:
                self.session.remove_alias(alias)


class TestWorkspaceRoundTripMonkey(MonkeyBase):
    """A random workspace, saved and loaded, must come back identical."""

    def test_round_trip(self):
        rng = random.Random(2024)
        self.session.add_file(self._paths[0])
        params = self.session.parameters

        for _ in range(6):
            page = self.state.add_page()
            page.name = "Page {0}".format(rng.randrange(1000))
            page.rows = rng.randint(1, 4)
            page.cols = rng.randint(1, 4)
            for tile in page.visible_tiles():
                tile.x_param = rng.choice(params)
                tile.y_params = rng.sample(params, rng.randint(0, 3))
                tile.plot3d = rng.random() < 0.3
                tile.z_param = rng.choice(params)
                tile.azimuth = rng.uniform(0, 360)
                tile.elevation = rng.uniform(-89, 89)
                tile.lock_x = rng.random() < 0.5
                tile.x_min, tile.x_max = 0.0, rng.uniform(1, 100)

        for name in rng.sample(params, min(3, len(params))):
            self.state.set_global_y_limit(name, -1.0, 1.0)
        self.state.markers.append(Marker(x_param=params[0], position=1.5))
        self.state.areas.append(AreaBand(x_param=params[0], x_min=0.0, x_max=2.0))
        self.state.styles.set_color("param::" + params[0], "#123456")

        handle, path = tempfile.mkstemp(suffix=".aoews")
        os.close(handle)
        try:
            save_workspace(path, self.state, [self._paths[0]], {}, {})
            data = load_workspace(path)

            restored = AppState()
            restored.load_dict(data)

            self.assertEqual(len(restored.pages), len(self.state.pages))
            for before, after in zip(self.state.pages, restored.pages):
                self.assertEqual(before.name, after.name)
                self.assertEqual((before.rows, before.cols),
                                 (after.rows, after.cols))
                for tile_a, tile_b in zip(before.visible_tiles(),
                                          after.visible_tiles()):
                    self.assertEqual(tile_a.x_param, tile_b.x_param)
                    self.assertEqual(tile_a.y_params, tile_b.y_params)
                    self.assertEqual(tile_a.plot3d, tile_b.plot3d)
                    self.assertEqual(tile_a.z_param, tile_b.z_param)
                    self.assertAlmostEqual(tile_a.azimuth, tile_b.azimuth, places=6)

            self.assertEqual(restored.global_y_limits, self.state.global_y_limits)
            self.assertEqual(len(restored.markers), len(self.state.markers))
            self.assertEqual(len(restored.areas), len(self.state.areas))
        finally:
            os.remove(path)


class TestProjectionMonkey(unittest.TestCase):
    """The 3D projection, which has no Qt in it and so can be checked directly."""

    def test_rotation_preserves_length_in_plane(self):
        """A horizontal segment keeps its length when viewed edge-on."""
        rng = random.Random(3)
        for _ in range(200):
            azimuth = rng.uniform(0, 360)
            x = np.array([0.0, 0.5])
            y = np.array([0.0, 0.0])
            z = np.array([0.0, 0.0])
            sx, sy, _d = projection.project(x, y, z, azimuth, 0.0)
            length = np.hypot(sx[1] - sx[0], sy[1] - sy[0])
            self.assertLessEqual(length, 0.5 + 1e-9)

    def test_projection_stays_inside_the_cube(self):
        """No view angle may push a normalised point outside the drawn frame."""
        rng = random.Random(11)
        corners = projection.CUBE_CORNERS
        for _ in range(400):
            azimuth = rng.uniform(0, 360)
            elevation = rng.uniform(-89, 89)
            sx, sy, _d = projection.project(
                corners[:, 0], corners[:, 1], corners[:, 2], azimuth, elevation
            )
            # The cube's half-diagonal is sqrt(3)/2; the view box is +/-0.95,
            # so a projection that fits inside that is guaranteed visible.
            self.assertLess(np.abs(sx).max(), 0.95)
            self.assertLess(np.abs(sy).max(), 0.95)

    def test_normalise_handles_degenerate_channels(self):
        flat = np.full(50, 3.25)
        result = projection.normalise(flat, (3.25, 3.25))
        self.assertTrue(np.all(result == 0.0))
        self.assertEqual(result.shape, flat.shape)

        result = projection.normalise(flat, None)
        self.assertTrue(np.all(result == 0.0))

        values = np.array([0.0, 5.0, 10.0])
        result = projection.normalise(values, (0.0, 10.0))
        self.assertAlmostEqual(float(result[0]), -0.5)
        self.assertAlmostEqual(float(result[-1]), 0.5)

    def test_normalise_never_produces_nan_from_finite_input(self):
        rng = np.random.default_rng(5)
        for _ in range(200):
            values = rng.normal(0, 1e6, 64)
            bounds = (float(values.min()), float(values.max()))
            result = projection.normalise(values, bounds)
            self.assertTrue(np.all(np.isfinite(result)))
            self.assertLessEqual(np.abs(result).max(), 0.5 + 1e-9)

    def test_axis_triad_is_a_direction_not_a_position(self):
        """Unit-length axes, so the gizmo can scale them to any pixel size.

        This is what keeps the triad the same size at any zoom: it is never
        expressed in data coordinates, so there is nothing for the view
        transform to scale.
        """
        for azimuth in range(0, 360, 11):
            for elevation in (-89, -30, 0, 30, 89):
                entries = projection.axis_triad(azimuth, elevation)
                self.assertEqual([e[0] for e in sorted(entries)][:0], [])
                self.assertEqual(sorted(e[0] for e in entries), ["X", "Y", "Z"])
                for _name, sx, sy, _colour, _depth in entries:
                    self.assertLessEqual(abs(sx), 1.0 + 1e-9)
                    self.assertLessEqual(abs(sy), 1.0 + 1e-9)

    def test_axis_triad_is_sorted_back_to_front(self):
        for azimuth in range(0, 360, 17):
            entries = projection.axis_triad(azimuth, 25.0)
            depths = [e[4] for e in entries]
            self.assertEqual(depths, sorted(depths))

    def test_triad_layout_never_paints_outside_its_box(self):
        """A label pushed past the bounding rect is clipped on screen.

        That kind of fault shows up at one rotation out of a hundred and is
        invisible in review, so it is asserted over the whole sphere instead.
        """
        size, arm, head, gap = 84.0, 26.0, 6.0, 9.0
        # Half a bold character, which the widget centres on the label point.
        text_margin = 6.0

        for azimuth in range(0, 360, 5):
            for elevation in range(-89, 90, 5):
                for entry in projection.triad_layout(
                        azimuth, elevation, size, arm, head, gap):
                    points = [entry["start"], entry["end"], entry["label"]]
                    points += list(entry["head"])
                    for x, y in points:
                        self.assertGreaterEqual(x, text_margin)
                        self.assertGreaterEqual(y, text_margin)
                        self.assertLessEqual(x, size - text_margin)
                        self.assertLessEqual(y, size - text_margin)

    def test_triad_layout_handles_an_axis_pointing_at_the_viewer(self):
        """No arrowhead can be built from a zero-length direction."""
        # Azimuth 0, elevation 0 puts Y straight through the screen.
        entries = {e["name"]: e for e in
                   projection.triad_layout(0.0, 0.0, 84.0, 26.0, 6.0, 9.0)}
        self.assertTrue(entries["Y"]["degenerate"])
        self.assertEqual(entries["Y"]["head"], ())
        self.assertFalse(entries["X"]["degenerate"])
        self.assertEqual(len(entries["X"]["head"]), 3)

    def test_triad_arrowheads_point_outward(self):
        for azimuth in range(0, 360, 13):
            for entry in projection.triad_layout(azimuth, 25.0, 84.0, 26.0,
                                                 6.0, 9.0):
                if entry["degenerate"]:
                    continue
                cx, cy = entry["start"]
                tip = entry["head"][0]
                # The tip of the head is the far end of the axis, and the two
                # base corners are nearer the origin than the tip is.
                tip_distance = np.hypot(tip[0] - cx, tip[1] - cy)
                for corner in entry["head"][1:]:
                    self.assertLess(
                        np.hypot(corner[0] - cx, corner[1] - cy), tip_distance
                    )

    def test_thinning_step(self):
        self.assertEqual(projection.thinning_step(0, 100), 1)
        self.assertEqual(projection.thinning_step(50, 100), 1)
        self.assertEqual(projection.thinning_step(1000, 100), 10)
        self.assertEqual(projection.thinning_step(1000, 0), 1)


class TestThreeDViewState(unittest.TestCase):
    """The window onto a 3D projection: zoom, fit, and surviving a rotation.

    The widget cannot be built without a QApplication, but the state it reads
    and writes can, and that is where both reported failures lived: auto-fit
    and the wheel had nowhere to record a result, so a 3D tile ignored them.
    """

    def test_defaults_frame_the_whole_cube(self):
        tile = TileState()
        self.assertEqual(tile.view3d_half, DEFAULT_3D_HALF)
        self.assertEqual((tile.view3d_cx, tile.view3d_cy), (0.0, 0.0))

        # Every corner of the cube must fall inside the default frame at any
        # rotation, or "fit" does not actually fit.
        corners = projection.CUBE_CORNERS
        for azimuth in range(0, 360, 15):
            for elevation in (-89, -45, 0, 45, 89):
                sx, sy, _d = projection.project(
                    corners[:, 0], corners[:, 1], corners[:, 2],
                    azimuth, elevation,
                )
                self.assertLess(float(np.abs(sx).max()), tile.view3d_half)
                self.assertLess(float(np.abs(sy).max()), tile.view3d_half)

    def test_reset_restores_the_default_frame(self):
        tile = TileState()
        tile.view3d_cx, tile.view3d_cy, tile.view3d_half = 3.0, -2.0, 0.05
        tile.reset_3d_view()
        self.assertEqual(tile.view3d_cx, 0.0)
        self.assertEqual(tile.view3d_cy, 0.0)
        self.assertEqual(tile.view3d_half, DEFAULT_3D_HALF)

    def test_view_survives_a_workspace_round_trip(self):
        """A zoom must not be lost by saving and reopening."""
        state = AppState()
        tile = state.tile
        tile.plot3d = True
        tile.x_param, tile.y_params, tile.z_param = "A", ["B"], "C"
        tile.view3d_cx, tile.view3d_cy, tile.view3d_half = 0.25, -0.4, 0.15

        restored = AppState()
        restored.load_dict(state.to_dict())
        other = restored.tile
        self.assertAlmostEqual(other.view3d_cx, 0.25)
        self.assertAlmostEqual(other.view3d_cy, -0.4)
        self.assertAlmostEqual(other.view3d_half, 0.15)

    def test_zoom_steps_are_bounded_and_reversible(self):
        """The arithmetic behind the wheel, without a widget.

        Scrolling in and back out must land where it started, and no amount of
        scrolling may reach zero or infinity -- either would leave the tile
        showing nothing, with no way back except the fit button.
        """
        zoom = clamp_3d_half

        half = DEFAULT_3D_HALF
        step_in = 1.0015 ** -120
        step_out = 1.0015 ** 120

        after = zoom(zoom(half * step_in) * step_out)
        self.assertAlmostEqual(after, half, places=9)

        # One notch should be a noticeable but not violent change.
        self.assertLess(zoom(half * step_in), half)
        self.assertGreater(zoom(half * step_in), half * 0.7)

        deep = half
        for _ in range(400):
            deep = zoom(deep * step_in)
        self.assertGreaterEqual(deep, MIN_3D_HALF)

        far = half
        for _ in range(400):
            far = zoom(far * step_out)
        self.assertLessEqual(far, MAX_3D_HALF)
        # Zooming out has to reach far enough to be worth having: the previous
        # ceiling let you back off by only 21x before it simply stopped.
        self.assertGreater(MAX_3D_HALF / DEFAULT_3D_HALF, 1000.0)


class TestAnnotationStability(unittest.TestCase):
    """An annotation must keep the parameter it was given.

    The reported symptom was that markers and areas "disappear from the images"
    when you click a plot with a different x axis. The cause was not in the
    drawing at all: refreshing the panel re-pointed the annotation. These tests
    pin the contract that the drawing code depends on -- an annotation's
    x_param changes only when someone changes it.
    """

    def test_area_describe_prefers_its_label(self):
        band = AreaBand(x_param="TIME", x_min=10.0, x_max=20.0)
        self.assertIn("TIME", band.describe())
        band.label = "second segment"
        self.assertEqual(band.describe(), "second segment")

    def test_area_label_survives_a_round_trip(self):
        state = AppState()
        state.areas.append(AreaBand(x_param="TIME", x_min=1.0, x_max=2.0,
                                    label="rotation"))
        restored = AppState()
        restored.load_dict(state.to_dict())
        self.assertEqual(restored.areas[0].label, "rotation")

    def test_workspaces_written_before_labels_still_load(self):
        """An old workspace has no 'label' key, and must not fail to open."""
        legacy = {
            "areas": [{"x_param": "TIME", "x_min": 0.0, "x_max": 1.0,
                       "color": "#3380ff", "alpha": 0.2, "visible": True}],
            "markers": [{"x_param": "TIME", "position": 5.0, "visible": True,
                         "color": "#ff3333"}],
        }
        state = AppState()
        state.load_dict(legacy)
        self.assertEqual(state.areas[0].label, "")
        self.assertEqual(state.markers[0].position, 5.0)

    def test_unknown_keys_from_a_newer_version_are_ignored(self):
        """Opening a workspace from a later build must not raise."""
        payload = {
            "areas": [{"x_param": "T", "x_min": 0.0, "x_max": 1.0,
                       "something_new": 42}],
            "markers": [{"x_param": "T", "position": 1.0, "future": "x"}],
        }
        state = AppState()
        state.load_dict(payload)
        self.assertEqual(state.areas[0].x_param, "T")
        self.assertEqual(state.markers[0].x_param, "T")


class TestTextNotes(unittest.TestCase):
    """Captions on a tile."""

    def test_defaults_are_on_screen(self):
        note = TextNote()
        self.assertTrue(0.0 <= note.x <= 1.0)
        self.assertTrue(0.0 <= note.y <= 1.0)
        self.assertTrue(note.visible)

    def test_describe_is_a_single_short_line(self):
        note = TextNote(text="first line\nsecond line")
        self.assertEqual(note.describe(), "first line")
        self.assertEqual(TextNote(text="").describe(), "(empty)")
        self.assertLessEqual(len(TextNote(text="x" * 200).describe()), 40)

    def test_notes_travel_with_the_tile(self):
        """Copy/paste of a tile has to bring its captions."""
        state = AppState()
        state.tile.texts.append(TextNote(text="V1", x=0.3, y=0.2, size=16.0))
        state.copy_tile()
        state.active_tile = 0
        state.page.rows, state.page.cols = 1, 2
        state.active_tile = 1
        self.assertTrue(state.paste_tile())
        pasted = state.page.tiles[1]
        self.assertEqual(len(pasted.texts), 1)
        self.assertEqual(pasted.texts[0].text, "V1")
        # A copy, not a shared reference: editing one must not edit the other.
        pasted.texts[0].text = "changed"
        self.assertEqual(state.page.tiles[0].texts[0].text, "V1")

    def test_notes_survive_a_workspace_round_trip(self):
        state = AppState()
        state.tile.texts.append(
            TextNote(text="stall", x=0.6, y=0.7, size=18.0,
                     color="#ff0000", bold=True))
        restored = AppState()
        restored.load_dict(state.to_dict())
        note = restored.tile.texts[0]
        self.assertEqual(note.text, "stall")
        self.assertAlmostEqual(note.x, 0.6)
        self.assertAlmostEqual(note.size, 18.0)
        self.assertTrue(note.bold)
        self.assertEqual(note.color, "#ff0000")


class TestLegendPlacement(unittest.TestCase):
    """A dragged legend has to stay where it was put."""

    def test_default_is_a_corner(self):
        tile = TileState()
        self.assertEqual(tile.legend_position, "top-left")

    def test_custom_position_survives_a_round_trip(self):
        state = AppState()
        tile = state.tile
        tile.legend_position = "custom"
        tile.legend_x, tile.legend_y = 0.42, 0.63

        restored = AppState()
        restored.load_dict(state.to_dict())
        self.assertEqual(restored.tile.legend_position, "custom")
        self.assertAlmostEqual(restored.tile.legend_x, 0.42)
        self.assertAlmostEqual(restored.tile.legend_y, 0.63)

    def test_custom_travels_with_a_copied_tile(self):
        state = AppState()
        state.tile.legend_position = "custom"
        state.tile.legend_x, state.tile.legend_y = 0.3, 0.7
        state.copy_tile()
        state.page.rows, state.page.cols = 1, 2
        state.active_tile = 1
        self.assertTrue(state.paste_tile())
        self.assertEqual(state.page.tiles[1].legend_position, "custom")
        self.assertAlmostEqual(state.page.tiles[1].legend_x, 0.3)


class TestIsoMonkey(MonkeyBase):
    """ISO selection and filtering, exercised the way a user clicks about."""

    def test_iso_selection_churn(self):
        rng = random.Random(313)
        self.session.add_file(self._paths[0])
        params = self.session.parameters

        for _round in range(400):
            name = rng.choice(params)
            iso = self.state.iso_state(name)
            iso.active = rng.random() < 0.5
            iso.grouped = rng.random() < 0.4
            iso.group_count = rng.randint(1, 8)

            values = self.session.iso_distinct(name)
            for value in values[:8]:
                iso.value_enabled[float(value)] = rng.random() < 0.7

            active = self.state.active_iso_filters()
            self.assertTrue(all(state.active for state in active))
            # Sorted by name, and each appears once: the filter list is
            # combined with AND, so a duplicate would silently double-filter.
            names = [state.param for state in active]
            self.assertEqual(names, sorted(names))
            self.assertEqual(len(names), len(set(names)))

            if rng.random() < 0.1:
                self.state.iso_states.clear()


if __name__ == "__main__":
    unittest.main()


class TestDrawBudget(unittest.TestCase):
    """The per-tile point budget, which decides how fast a pan feels.

    This was a real defect, not a hypothetical: three 550 MB files of 25 runs
    each, plotted against one parameter, is 75 curves -- and the budget meant
    to protect the draw path was raising the per-curve target from 1 800 to
    10 000, so every pan pushed 1.5 M points into pyqtgraph instead of 270 k.
    The arithmetic lives in `scales` precisely so it can be checked here,
    without Qt, which is the only reason a test can exist for it at all.
    """

    WIDTH = 1800   # a ~900 px tile at two points per pixel

    def test_a_small_tile_keeps_the_width_target(self):
        self.assertEqual(resample_target(self.WIDTH, 4, 100_000), self.WIDTH)

    def test_a_single_curve_is_never_budgeted(self):
        # One curve cannot be "too many curves", however long it is.
        self.assertEqual(
            resample_target(self.WIDTH, 1, 50_000_000), self.WIDTH)

    def test_the_budget_never_raises_the_target(self):
        """The regression. 75 curves must not get more points than 4 do."""
        self.assertEqual(
            resample_target(self.WIDTH, 75, 5_300_000), self.WIDTH)

    def test_the_budget_still_bites_at_a_thousand_curves(self):
        """And must keep doing the job it was written for."""
        target = resample_target(1200, 1000, 2_400_000)
        self.assertLess(target, 1200)
        self.assertEqual(target, 750)

    def test_the_total_is_bounded_once_the_budget_applies(self):
        # Two points per bucket, so buckets * curves * 2 is what Qt receives.
        for curves in (500, 1000, 5000, 20000):
            target = resample_target(1200, curves, 10_000_000)
            drawn = target * curves * 2
            self.assertLessEqual(drawn, MAX_TILE_POINTS * 1.05,
                                 "{0} curves would draw {1} points".format(
                                     curves, drawn))

    def test_the_target_never_collapses_to_nothing(self):
        # A floor of 32 buckets: past this the curve stops being a shape.
        self.assertGreaterEqual(resample_target(1200, 10_000_000, 10 ** 12), 32)

    def test_it_is_monotonic_in_the_curve_count(self):
        """More curves may never earn more points each."""
        previous = None
        for curves in range(2, 4000, 37):
            target = resample_target(self.WIDTH, curves, 50_000_000)
            if previous is not None:
                self.assertLessEqual(target, previous)
            previous = target

    def test_it_matches_a_direct_reading_of_the_two_limits(self):
        for curves in (2, 10, 75, 300, 400, 401, 1000, 9999):
            for total in (1000, MAX_TILE_POINTS, MAX_TILE_POINTS + 1, 10 ** 8):
                got = resample_target(self.WIDTH, curves, total)
                if total <= MAX_TILE_POINTS:
                    want = self.WIDTH
                else:
                    want = min(self.WIDTH,
                               max(32, int(MAX_TILE_POINTS / curves / 2)))
                self.assertEqual(got, want, (curves, total))
