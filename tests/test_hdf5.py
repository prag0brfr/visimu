# Author: Paulo R. A. Gilz
"""Tests for reading result files stored as HDF5.

Two halves. The first exercises the layout detection in `aoeviewer.hdf5`
against a stand-in for h5py, so it runs on a machine where h5py is not
installed -- which is every machine this project has been developed on, and
possibly the target one too. The second pushes a file through the core and
reads it back, which needs the compiled library but not h5py.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import h5py_stub

from aoeviewer import hdf5


CHANNELS = ["ALFA", "CG", "MACH", "MASSE", "Q1", "TETA", "TIME", "VC", "ZP"]
SAMPLES = 76


def make_channels(run_index):
    """One run's worth of plausible flight data, including two constants."""
    time = np.arange(SAMPLES) * 0.25
    return {
        "TIME": time,
        "ALFA": 2.0 + np.sin(time) + run_index,
        "MACH": 0.6 + 0.001 * time,
        "ZP": 9000.0 + 8.0 * time + run_index,
        "VC": 140.0 + 0.4 * time,
        "TETA": 3.0 + np.cos(time),
        "Q1": 0.1 * np.sin(2.0 * time),
        "MASSE": np.full(SAMPLES, 61000.0 - 100.0 * run_index),
        "CG": np.full(SAMPLES, 0.23),
    }


class Hdf5Mixin:
    """Installs the h5py stand-in for the duration of a test class."""

    @classmethod
    def setUpClass(cls):
        cls._real_h5py = sys.modules.get("h5py")
        sys.modules["h5py"] = h5py_stub
        cls._dir = tempfile.TemporaryDirectory(prefix="aoe_hdf5_test_")

    @classmethod
    def tearDownClass(cls):
        if cls._real_h5py is None:
            sys.modules.pop("h5py", None)
        else:
            sys.modules["h5py"] = cls._real_h5py
        cls._dir.cleanup()

    def path(self, name):
        return os.path.join(self._dir.name, name)

    def write_channel_groups(self, name, runs=("1_2_0", "2_2_0"), aux=True):
        """A group per channel, a dataset per run, the run in the leaf name.

        The convention of the file this was built for. The run identity is the
        last three underscore-separated tokens of the dataset name, and the
        same three appear under every channel group.
        """
        target = self.path(name)
        with h5py_stub.File(target, "w") as handle:
            group = handle.create_group("AVIONX/CASE_1/CHANNELS00")
            handle.create_group("AVIONX/CASE_1/METADATA001")
            for index, run in enumerate(runs):
                values = make_channels(index)
                for channel in CHANNELS:
                    holder = group.create_group(channel) \
                        if channel not in dict(group.items()) \
                        else dict(group.items())[channel]
                    dataset = holder.create_dataset(
                        "%s_%s" % (channel, run), data=values[channel])
                    if channel == "TIME":
                        dataset.attrs["units"] = "s"
                    elif channel == "ZP":
                        dataset.attrs["units"] = "m"
            if aux:
                group.create_dataset("AUX", data=np.arange(4.0))
        return target

    def write_run_groups(self, name, runs=("RUN_1", "RUN_2")):
        """A group per run, a dataset per channel: the other convention."""
        target = self.path(name)
        with h5py_stub.File(target, "w") as handle:
            for index, run in enumerate(runs):
                values = make_channels(index)
                group = handle.create_group(run)
                for channel in CHANNELS:
                    group.create_dataset(channel, data=values[channel])
        return target


class LayoutDetectionTests(Hdf5Mixin, unittest.TestCase):

    def test_reads_a_channel_group_per_channel_and_a_dataset_per_run(self):
        described = hdf5.inspect(self.write_channel_groups("channels.hdf5"))
        self.assertEqual(described.layout.rule, "leaf_tail_3")
        self.assertEqual(described.runs, ["1_2_0", "2_2_0"])
        self.assertEqual(described.channels, CHANNELS)
        self.assertEqual(described.layout.samples, SAMPLES)

    def test_reads_a_group_per_run_and_a_dataset_per_channel(self):
        described = hdf5.inspect(self.write_run_groups("runs.hdf5"))
        self.assertEqual(described.layout.rule, "parent_group")
        self.assertEqual(len(described.runs), 2)
        self.assertEqual(described.channels, CHANNELS)

    def test_the_transposed_reading_is_not_chosen(self):
        """The decisive case, and the one that was got wrong first.

        Reading the channel-group file as a single run gives eighteen channels
        called ALFA_1_2_0 and friends. It is a consistent reading -- nothing
        about the tree forbids it -- so only the evidence rules it out: the
        channel name would not match the group it sits in, and the run keys
        would not be shared across groups.
        """
        described = hdf5.inspect(self.write_channel_groups("transpose.hdf5"))
        self.assertNotEqual(described.layout.rule, "single_run")
        self.assertNotIn("ALFA_1_2_0", described.channels)
        self.assertEqual(len(described.runs) * len(described.channels), 18)

    def test_a_metadata_vector_of_another_length_is_ignored(self):
        described = hdf5.inspect(self.write_channel_groups("aux.hdf5"))
        self.assertNotIn("AUX", described.channels)

    def test_units_are_taken_from_the_dataset_attributes(self):
        described = hdf5.inspect(self.write_channel_groups("units.hdf5"))
        units = dict(zip(described.channels, described.units))
        self.assertEqual(units["TIME"], "s")
        self.assertEqual(units["ZP"], "m")
        self.assertEqual(units["MACH"], "")

    def test_runs_sort_naturally_not_asciibetically(self):
        described = hdf5.inspect(self.write_channel_groups(
            "many.hdf5", runs=("1_2_0", "2_2_0", "10_2_0")))
        self.assertEqual(described.runs, ["1_2_0", "2_2_0", "10_2_0"])

    def test_a_single_run_file_reads_as_one_run(self):
        target = self.path("single.hdf5")
        with h5py_stub.File(target, "w") as handle:
            values = make_channels(0)
            group = handle.create_group("results")
            for channel in CHANNELS:
                group.create_dataset(channel, data=values[channel])
        described = hdf5.inspect(target)
        self.assertEqual(len(described.runs), 1)
        self.assertEqual(described.channels, CHANNELS)

    def test_a_file_with_no_numeric_vectors_is_refused(self):
        target = self.path("empty.hdf5")
        with h5py_stub.File(target, "w") as handle:
            handle.create_group("nothing/here")
        with self.assertRaises(ValueError):
            hdf5.inspect(target)

    def test_the_signature_decides_not_the_extension(self):
        target = self.write_channel_groups("disguised.dat")
        self.assertTrue(hdf5.looks_like_hdf5(target))

        text = self.path("plain.txt")
        with open(text, "w", encoding="utf-8") as handle:
            handle.write("*PARAMETRES\n'TIME'\n")
        self.assertFalse(hdf5.looks_like_hdf5(text))

    def test_an_unreadable_path_with_an_hdf5_suffix_still_routes_here(self):
        # The router runs before the file is opened; a missing .hdf5 must not
        # be sent to the text parser, which would report the wrong problem.
        self.assertTrue(hdf5.looks_like_hdf5(self.path("absent.hdf5")))

    def test_columns_come_back_in_the_declared_order(self):
        target = self.write_channel_groups("order.hdf5")
        described = hdf5.inspect(target)
        seen = {}
        for run_index, channel_index, values in hdf5.read_columns(
                target, described.layout):
            seen.setdefault(run_index, []).append(channel_index)
            self.assertEqual(values.dtype, np.float32)
            self.assertEqual(values.size, SAMPLES)
        self.assertEqual(sorted(seen), [0, 1])
        for indices in seen.values():
            self.assertEqual(indices, list(range(len(CHANNELS))))


def core_available():
    try:
        from aoeviewer import Session  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(core_available(), "the core library is not loadable here")
class PushedFileTests(Hdf5Mixin, unittest.TestCase):
    """The array route: what the core makes of a file it did not parse."""

    def open_session(self, path):
        from aoeviewer import Session

        session = Session()
        self.addCleanup(session.close)
        return session, session.add_file(path)

    def test_a_pushed_file_looks_like_any_other_file(self):
        target = self.write_channel_groups("push.hdf5")
        session, handle = self.open_session(target)
        self.assertEqual(handle.simulation_count, 2)
        self.assertEqual([sim.name for sim in handle.simulations],
                         ["1_2_0", "2_2_0"])
        self.assertEqual(session.parameters, CHANNELS)
        self.assertEqual(handle.name, "push.hdf5")

    def test_every_sample_survives_the_push(self):
        target = self.write_channel_groups("values.hdf5")
        session, handle = self.open_session(target)
        for run_index in range(handle.simulation_count):
            expected = make_channels(run_index)
            for channel in CHANNELS:
                got = session.channel(handle.index, run_index, channel)
                want = expected[channel].astype(np.float32)
                self.assertEqual(got.shape, want.shape)
                self.assertTrue(np.array_equal(got, want), channel)

    def test_units_reach_the_session(self):
        target = self.write_channel_groups("pushunits.hdf5")
        session, _ = self.open_session(target)
        self.assertEqual(session.unit("TIME"), "s")
        self.assertEqual(session.unit("MACH"), "")

    def test_summaries_are_computed_for_pushed_columns(self):
        # Autoscaling asks for a curve's extent on every replot and expects the
        # answer to come from the summary computed at load, not from a scan.
        target = self.write_channel_groups("bounds.hdf5")
        session, handle = self.open_session(target)
        values = session.channel(handle.index, 0, "ZP")
        low, high = session.bounds(values)
        self.assertAlmostEqual(low, float(values.min()), places=3)
        self.assertAlmostEqual(high, float(values.max()), places=3)

    def test_decimation_works_on_a_pushed_file(self):
        target = self.write_channel_groups("decimate.hdf5")
        session, handle = self.open_session(target)
        x = session.channel(handle.index, 0, "TIME")
        y = session.channel(handle.index, 0, "ZP")
        dx, dy = session.decimate(x, y, 16)
        self.assertLessEqual(dx.size, SAMPLES)
        self.assertEqual(dx.size, dy.size)
        self.assertGreater(dx.size, 0)

    def test_formulas_work_on_a_pushed_file(self):
        target = self.write_channel_groups("formula.hdf5")
        session, handle = self.open_session(target)
        session.set_computed("ZP_FT", "ZP * 3.280839895")
        got = session.channel(handle.index, 0, "ZP_FT")
        want = session.channel(handle.index, 0, "ZP") * 3.280839895
        self.assertTrue(np.allclose(got, want, rtol=1e-5))

    def test_iso_values_work_on_a_pushed_file(self):
        target = self.write_channel_groups("iso.hdf5")
        session, handle = self.open_session(target)
        values = session.iso_values(handle.index, "MASSE")
        self.assertEqual(len(values), 2)
        self.assertAlmostEqual(values[0], 61000.0, places=1)
        self.assertAlmostEqual(values[1], 60900.0, places=1)

    def test_no_sidecar_is_written_for_a_pushed_file(self):
        # A .aoec describes a parse of the source. There is no parse here, and
        # one left beside the file would be claimed by a later open().
        target = self.write_channel_groups("nocache.hdf5")
        session, handle = self.open_session(target)
        self.assertTrue(session.is_pushed(target))
        self.assertFalse(session.cache_active(handle.index))
        self.assertFalse(session.build_cache(handle.index))
        self.assertFalse(os.path.exists(target + ".aoec"))

    def test_reload_runs_the_reader_again_and_keeps_the_index(self):
        target = self.write_channel_groups("reload.hdf5")
        session, handle = self.open_session(target)
        key_before = handle.simulations[0].key
        self.assertTrue(handle.reload())
        self.assertEqual(session.file_count, 1)
        again = session.file(0)
        self.assertEqual(again.simulation_count, 2)
        self.assertEqual(session.parameters, CHANNELS)
        # Styles are keyed on this; a reload that changed it would silently
        # detach every colour the user had set.
        self.assertEqual(again.simulations[0].key, key_before)

    def test_renaming_survives_a_reload(self):
        target = self.write_channel_groups("rename.hdf5")
        session, handle = self.open_session(target)
        handle.rename("My study")
        self.assertTrue(session.file(0).reload())
        self.assertEqual(session.file(0).name, "My study")

    def test_a_pushed_file_and_a_parsed_file_coexist(self):
        from gen_sample import generate

        text = self.path("mixed.aoe")
        generate(text, sims=3, rows=40)
        target = self.write_channel_groups("mixed.hdf5")

        session, _ = self.open_session(text)
        pushed = session.add_file(target)
        self.assertEqual(session.file_count, 2)
        self.assertTrue(session.is_pushed(target))
        self.assertFalse(session.is_pushed(text))
        self.assertEqual(pushed.simulation_count, 2)
        # The parameter list is the union of both files' headers.
        for channel in CHANNELS:
            self.assertIn(channel, session.parameters)

    def test_removing_a_pushed_file_forgets_it(self):
        target = self.write_channel_groups("forget.hdf5")
        session, handle = self.open_session(target)
        session.remove_file(handle.index)
        self.assertEqual(session.file_count, 0)
        self.assertFalse(session.is_pushed(target))


@unittest.skipUnless(core_available(), "the core library is not loadable here")

class LazyPushTests(Hdf5Mixin, unittest.TestCase):
    """What is read, and when.

    The whole point of the pull route: a file that opens without being read,
    and columns that arrive as they are drawn. So these tests count reads
    rather than inspect values -- the values are covered above, and the cost is
    what changed.
    """

    def setUp(self):
        self.reads = []
        self._real_read_selected = hdf5.read_selected
        self._real_read_first = hdf5.read_first_values

        def counting_selected(path, layout, wanted):
            pairs = sorted(set((int(r), int(c)) for r, c in wanted))
            self.reads.append(("columns", tuple(pairs)))
            return self._real_read_selected(path, layout, wanted)

        def counting_first(path, layout, column):
            self.reads.append(("first", int(column)))
            return self._real_read_first(path, layout, column)

        hdf5.read_selected = counting_selected
        hdf5.read_first_values = counting_first
        self.addCleanup(setattr, hdf5, "read_selected", self._real_read_selected)
        self.addCleanup(setattr, hdf5, "read_first_values", self._real_read_first)

    def open_session(self, name="lazy.hdf5"):
        from aoeviewer import Session

        target = self.write_channel_groups(name)
        session = Session()
        self.addCleanup(session.close)
        return session, session.add_file(target), target

    def column_reads(self):
        return [pairs for kind, pairs in self.reads if kind == "columns"]

    def test_opening_reads_no_samples_at_all(self):
        session, handle, _path = self.open_session()
        self.assertEqual(self.column_reads(), [])
        # And yet the file is fully described: this is what the DATA tree and
        # the parameter lists are built from.
        self.assertEqual(handle.simulation_count, 2)
        self.assertEqual(session.parameters, CHANNELS)

    def test_asking_for_one_channel_reads_one_channel(self):
        session, handle, _path = self.open_session()
        session.channel(handle.index, 0, "ALFA")
        column = CHANNELS.index("ALFA")
        self.assertEqual(self.column_reads(), [((0, column),)])

    def test_asking_twice_reads_once(self):
        session, handle, _path = self.open_session()
        session.channel(handle.index, 0, "ALFA")
        session.channel(handle.index, 0, "ALFA")
        self.assertEqual(len(self.column_reads()), 1)

    def test_each_run_is_fetched_separately(self):
        session, handle, _path = self.open_session()
        session.channel(handle.index, 0, "ALFA")
        session.channel(handle.index, 1, "ALFA")
        column = CHANNELS.index("ALFA")
        self.assertEqual(self.column_reads(),
                         [((0, column),), ((1, column),)])

    def test_a_preload_fetches_the_whole_batch_in_one_pass(self):
        session, handle, _path = self.open_session()
        session.preload(handle.index, [0, 1], ["TIME", "ALFA"])
        self.assertEqual(len(self.column_reads()), 1)
        self.assertEqual(len(self.column_reads()[0]), 4)

    def test_a_preloaded_channel_is_not_fetched_again_on_use(self):
        session, handle, _path = self.open_session()
        session.preload(handle.index, [0], ["ALFA"])
        before = len(self.column_reads())
        session.channel(handle.index, 0, "ALFA")
        self.assertEqual(len(self.column_reads()), before)

    def test_the_values_are_right_whichever_way_they_arrived(self):
        session, handle, _path = self.open_session()
        session.preload(handle.index, [0], ["ALFA"])          # batched
        batched = session.channel(handle.index, 0, "ALFA")
        single = session.channel(handle.index, 1, "MACH")     # on demand
        expected_a = make_channels(0)["ALFA"].astype(np.float32)
        expected_m = make_channels(1)["MACH"].astype(np.float32)
        self.assertTrue(np.array_equal(batched, expected_a))
        self.assertTrue(np.array_equal(single, expected_m))

    def test_an_unknown_channel_asks_the_reader_for_nothing(self):
        session, handle, _path = self.open_session()
        session.channel(handle.index, 0, "NOT_A_CHANNEL")
        self.assertEqual(self.column_reads(), [])

    def test_an_iso_scan_reads_one_element_per_run_not_the_column(self):
        session, handle, _path = self.open_session()
        values = session.iso_values(handle.index, "MASSE")
        self.assertEqual(len(values), 2)
        self.assertAlmostEqual(values[0], 61000.0, places=1)
        self.assertAlmostEqual(values[1], 60900.0, places=1)
        # Through the cheap path, and no column was pulled in to answer it.
        self.assertEqual(self.column_reads(), [])
        self.assertEqual([kind for kind, _ in self.reads], ["first"])

    def test_an_iso_scan_is_answered_from_memory_the_second_time(self):
        session, handle, _path = self.open_session()
        session.iso_values(handle.index, "MASSE")
        session.iso_values(handle.index, "MASSE")
        self.assertEqual(len(self.reads), 1)

    def test_distinct_iso_values_include_the_pushed_file(self):
        session, handle, _path = self.open_session()
        distinct = session.iso_distinct("MASSE")
        self.assertEqual(sorted(round(float(v), 1) for v in distinct),
                         [60900.0, 61000.0])

    def test_reloading_forgets_what_it_had_read(self):
        session, handle, _path = self.open_session()
        session.channel(handle.index, 0, "ALFA")
        self.assertTrue(handle.reload())
        before = len(self.column_reads())
        # The reload re-pushed the file with no samples, so the next ask has to
        # go back to the reader.
        session.channel(0, 0, "ALFA")
        self.assertEqual(len(self.column_reads()), before + 1)


class PushedSidecarTests(Hdf5Mixin, unittest.TestCase):
    """The sidecar written from pushed columns, and the open that uses it."""

    def session(self):
        from aoeviewer import Session

        session = Session()
        self.addCleanup(session.close)
        return session

    def test_a_sidecar_cannot_be_written_while_columns_are_missing(self):
        # There is no parser here to fill a gap, so a sidecar written with one
        # would record the file as holding no samples.
        target = self.write_channel_groups("partial.hdf5")
        session = self.session()
        handle = session.add_file(target)
        session.channel(handle.index, 0, "ALFA")      # one column of one run
        self.assertFalse(session.build_cache(handle.index))
        self.assertFalse(os.path.exists(target + ".aoec"))

    def test_filling_it_completely_lets_the_sidecar_be_written(self):
        target = self.write_channel_groups("full.hdf5")
        session = self.session()
        handle = session.add_file(target)
        self.assertTrue(session.fill_pushed_completely(handle.index))
        self.assertTrue(session.build_cache(handle.index))
        self.assertTrue(session.cache_active(handle.index))

    def test_a_later_open_is_served_by_the_sidecar(self):
        target = self.write_channel_groups("cached.hdf5")
        first = self.session()
        handle = first.add_file(target)
        first.fill_pushed_completely(handle.index)
        self.assertTrue(first.build_cache(handle.index))

        # A second session, as a later run of the program would be.
        second = self.session()
        again = second.add_file(target)
        self.assertTrue(second.cache_active(again.index))
        # And the columns are there without the reader being asked, which is
        # what makes the cached open the fast one.
        self.assertTrue(second.is_pushed_index(again.index))
        for run in range(again.simulation_count):
            for name in CHANNELS:
                got = second.channel(again.index, run, name)
                want = make_channels(run)[name].astype(np.float32)
                self.assertTrue(np.array_equal(got, want), name)

    def test_the_sidecar_is_refused_when_the_source_has_changed(self):
        target = self.write_channel_groups("changed.hdf5")
        session = self.session()
        handle = session.add_file(target)
        session.fill_pushed_completely(handle.index)
        self.assertTrue(session.build_cache(handle.index))

        # Rewrite the source with different numbers under the same name.
        with open(target, "r+b") as fh:
            fh.seek(0, os.SEEK_END)
            fh.write(b"\0" * 64)

        again = self.session()
        reopened = again.add_file(target)
        self.assertFalse(again.cache_active(reopened.index))

    def test_values_survive_the_sidecar_round_trip_exactly(self):
        # float32 both sides, so this is equality and not closeness.
        target = self.write_channel_groups("exact.hdf5")
        first = self.session()
        handle = first.add_file(target)
        first.fill_pushed_completely(handle.index)
        first.build_cache(handle.index)

        second = self.session()
        again = second.add_file(target)
        for run in range(again.simulation_count):
            expected = make_channels(run)
            for name in CHANNELS:
                self.assertTrue(np.array_equal(
                    second.channel(again.index, run, name),
                    expected[name].astype(np.float32)), name)

    def test_the_run_names_and_units_come_back_too(self):
        target = self.write_channel_groups("meta.hdf5")
        first = self.session()
        handle = first.add_file(target)
        first.fill_pushed_completely(handle.index)
        first.build_cache(handle.index)

        second = self.session()
        again = second.add_file(target)
        self.assertEqual([sim.name for sim in again.simulations],
                         ["1_2_0", "2_2_0"])
        self.assertEqual(second.unit("TIME"), "s")


@unittest.skipUnless(core_available(), "the core library is not loadable here")
class MissingDependencyTests(unittest.TestCase):

    def test_the_error_names_the_missing_package(self):
        real = sys.modules.pop("h5py", None)
        sys.modules["h5py"] = None  # an import of None raises ImportError
        try:
            with self.assertRaises(ValueError) as caught:
                hdf5.inspect("whatever.hdf5")
            self.assertIn("h5py", str(caught.exception))
        finally:
            if real is None:
                sys.modules.pop("h5py", None)
            else:
                sys.modules["h5py"] = real


if __name__ == "__main__":
    unittest.main()
