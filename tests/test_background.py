# Author: Paulo R. A. Gilz
"""Work that happens somewhere else: the out-of-process sidecar build.

Two things are being checked here, and only one of them is "does it work".

The other is that it stays harmless. This code reads files it does not own --
a solver's output, on a share, possibly being rewritten as we read it -- and
writes a derived file beside them. The failure that matters is not an absent
sidecar; it is a sidecar that mixes a file's before and after, or a source file
that was touched. So most of what follows is about restraint.

    python -m unittest tests.test_background -v
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np

import h5py_stub
from aoeviewer import cachebuild
# The project's own sample writer, rather than a hand-rolled one: the .aoe
# dialect has more to it than it looks (two *PARAMETRES spellings, optional
# *ENTITY text, wrapped tables), and a test file the core cannot read would be
# testing the fixture rather than the code.
from gen_sample import CHANNELS, UNITS, generate

SAMPLES = 400


class _Scratch(unittest.TestCase):
    """A directory of its own, removed afterwards."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="aoe-bg-")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def write_aoe(self, name="results.aoe", runs=2, rows=SAMPLES):
        path = os.path.join(self.dir, name)
        generate(path, runs, rows)
        return path

    def sidecar(self, path):
        return path + ".aoec"


class TheFingerprint(_Scratch):
    """What counts as the file having moved."""

    def test_a_missing_file_has_none(self):
        self.assertIsNone(cachebuild._fingerprint(
            os.path.join(self.dir, "nope.aoe")))

    def test_the_same_file_fingerprints_the_same(self):
        path = self.write_aoe()
        self.assertEqual(cachebuild._fingerprint(path),
                         cachebuild._fingerprint(path))

    def test_appending_to_the_file_changes_it(self):
        path = self.write_aoe()
        before = cachebuild._fingerprint(path)
        with open(path, "a") as handle:
            handle.write("# more\n")
        self.assertNotEqual(cachebuild._fingerprint(path), before)


class BuildingOne(_Scratch):
    """The .aoe route, which the core reads itself."""

    def test_a_sidecar_is_written_and_used(self):
        from aoeviewer import Session

        path = self.write_aoe()
        self.assertTrue(cachebuild.build_one(path))
        self.assertTrue(os.path.exists(self.sidecar(path)))

        session = Session()
        self.addCleanup(session.close)
        handle = session.add_file(path)
        self.assertTrue(session.cache_active(handle.index))

    def test_the_values_survive_the_round_trip(self):
        from aoeviewer import Session

        path = self.write_aoe()
        plain = Session()
        first = plain.add_file(path)
        expected = plain.channel(first.index, 0, "ALPHA").copy()
        plain.close()

        self.assertTrue(cachebuild.build_one(path))

        cached = Session()
        self.addCleanup(cached.close)
        again = cached.add_file(path)
        self.assertTrue(cached.cache_active(again.index))
        self.assertTrue(np.array_equal(
            cached.channel(again.index, 0, "ALPHA"), expected))

    def test_a_file_that_cannot_be_read_is_not_an_exception(self):
        self.assertFalse(cachebuild.build_one(
            os.path.join(self.dir, "absent.aoe")))

    def test_rubbish_is_not_an_exception_either(self):
        path = os.path.join(self.dir, "rubbish.aoe")
        with open(path, "wb") as handle:
            handle.write(os.urandom(4096))
        # Whatever the core makes of this, it must come back as False rather
        # than a traceback: this runs with nobody watching.
        self.assertIn(cachebuild.build_one(path), (True, False))

    def test_the_source_is_never_written(self):
        path = self.write_aoe()
        before = os.stat(path)
        with open(path, "rb") as handle:
            content = handle.read()

        self.assertTrue(cachebuild.build_one(path))

        after = os.stat(path)
        self.assertEqual(after.st_size, before.st_size)
        self.assertEqual(int(after.st_mtime), int(before.st_mtime))
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(), content)

    def test_the_source_survives_a_build_it_cannot_finish(self):
        # The sidecar is refused and the file it describes is left exactly as
        # it was. This is the one outcome that must hold whatever happens.
        path = self.write_aoe()
        with open(path, "rb") as handle:
            content = handle.read()

        real = cachebuild._fingerprint
        calls = []

        def moving(target):
            calls.append(target)
            answer = real(target)
            # Different on the second look: the file moved while we read it.
            if len(calls) > 1 and answer is not None:
                return (answer[0] + 1, answer[1])
            return answer

        cachebuild._fingerprint = moving
        try:
            self.assertFalse(cachebuild.build_one(path))
        finally:
            cachebuild._fingerprint = real

        self.assertFalse(os.path.exists(self.sidecar(path)),
                         "a sidecar was kept for a source that moved")
        self.assertTrue(os.path.exists(path), "the source was removed")
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(), content, "the source was rewritten")


class BuildingAPushedFile(_Scratch):
    """The HDF5 route, whose columns have to be fetched before they can be
    written out."""

    @classmethod
    def setUpClass(cls):
        cls._real_h5py = sys.modules.get("h5py")
        sys.modules["h5py"] = h5py_stub

    @classmethod
    def tearDownClass(cls):
        if cls._real_h5py is None:
            sys.modules.pop("h5py", None)
        else:
            sys.modules["h5py"] = cls._real_h5py

    def write_hdf5(self, name="results.hdf5", runs=2):
        path = os.path.join(self.dir, name)
        with h5py_stub.File(path, "w") as handle:
            group = handle.create_group("ROOT/CASE")
            base = np.arange(SAMPLES, dtype=np.float64) * 0.01
            for channel, unit in zip(CHANNELS, UNITS):
                child = group.create_group(channel)
                for run in range(runs):
                    dataset = child.create_dataset(
                        "{0}_{1}_0_0".format(channel, run + 1),
                        data=base + run)
                    dataset.attrs["unit"] = unit
        return path

    def test_every_column_is_fetched_and_the_sidecar_written(self):
        from aoeviewer import Session

        path = self.write_hdf5()
        self.assertTrue(cachebuild.build_one(path))
        self.assertTrue(os.path.exists(self.sidecar(path)))

        session = Session()
        self.addCleanup(session.close)
        handle = session.add_file(path)
        self.assertTrue(session.cache_active(handle.index),
                        "the sidecar was not picked up")
        for run in range(handle.simulation_count):
            for channel in CHANNELS:
                column = session.channel(handle.index, run, channel)
                self.assertTrue(np.array_equal(
                    column,
                    (np.arange(SAMPLES) * 0.01 + run).astype(np.float32)),
                    "{0} of run {1}".format(channel, run))

    def test_the_source_is_never_written(self):
        path = self.write_hdf5()
        before = os.stat(path)
        self.assertTrue(cachebuild.build_one(path))
        after = os.stat(path)
        self.assertEqual(after.st_size, before.st_size)
        self.assertEqual(int(after.st_mtime), int(before.st_mtime))


class BuildingSeveral(_Scratch):
    def test_each_one_gets_its_own_sidecar(self):
        paths = [self.write_aoe("a.aoe"), self.write_aoe("b.aoe")]
        self.assertEqual(cachebuild.build_many(paths), 2)
        for path in paths:
            self.assertTrue(os.path.exists(self.sidecar(path)))

    def test_one_failure_does_not_stop_the_rest(self):
        good = self.write_aoe("good.aoe")
        missing = os.path.join(self.dir, "missing.aoe")
        self.assertEqual(cachebuild.build_many([missing, good]), 1)
        self.assertTrue(os.path.exists(self.sidecar(good)))

    def test_main_reports_how_many_failed(self):
        good = self.write_aoe("good.aoe")
        missing = os.path.join(self.dir, "missing.aoe")
        self.assertEqual(cachebuild.main([good]), 0)
        self.assertEqual(cachebuild.main([missing]), 1)
        self.assertEqual(cachebuild.main([]), 0)

    def test_nothing_to_do_is_not_a_failure(self):
        self.assertEqual(cachebuild.main(["", None]), 0)


class HowTheSubprocessIsStarted(unittest.TestCase):
    """The command and the environment, without starting anything."""

    def test_a_checkout_runs_the_module(self):
        command = cachebuild.command_for(["/data/one.aoe"])
        self.assertEqual(command[1:], ["-m", "aoeviewer",
                                       cachebuild.FLAG, "/data/one.aoe"])

    def test_a_frozen_build_re_executes_itself(self):
        frozen = getattr(sys, "frozen", None)
        sys.frozen = True
        try:
            command = cachebuild.command_for(["/data/one.aoe"])
        finally:
            if frozen is None:
                del sys.frozen
            else:
                sys.frozen = frozen
        # No interpreter arguments: the executable *is* the interpreter, and
        # passing -m to it would make it look for a module it does not have.
        self.assertEqual(command, [sys.executable, cachebuild.FLAG,
                                   "/data/one.aoe"])

    def test_the_flag_is_one_the_entry_point_accepts(self):
        with open(os.path.join(ROOT, "python/aoeviewer/__main__.py")) as handle:
            source = handle.read()
        self.assertIn(cachebuild.FLAG, source,
                      "the subprocess would be started with a flag "
                      "__main__.py does not know")

    def _spawn_capturing(self, paths):
        """Call spawn() with Popen replaced, and return what it was given."""
        import subprocess

        seen = {}

        class FakePopen:
            pid = 4321

            def __init__(self, command, **kwargs):
                seen["command"] = command
                seen["kwargs"] = kwargs

            def poll(self):
                return None

        real = subprocess.Popen
        subprocess.Popen = FakePopen
        try:
            process = cachebuild.spawn(paths)
        finally:
            subprocess.Popen = real
        return process, seen

    def test_the_paths_are_passed_through(self):
        process, seen = self._spawn_capturing(["/data/a.aoe", "/data/b.hdf5"])
        self.assertIsNotNone(process)
        self.assertEqual(seen["command"][-2:], ["/data/a.aoe", "/data/b.hdf5"])

    def test_the_package_is_importable_by_the_child(self):
        _process, seen = self._spawn_capturing(["/data/a.aoe"])
        package_parent = seen["kwargs"]["env"]["PYTHONPATH"].split(os.pathsep)[0]
        self.assertTrue(
            os.path.isdir(os.path.join(package_parent, "aoeviewer")),
            "PYTHONPATH does not lead to the package: " + package_parent)

    def test_hdf5_locking_is_off_for_the_child(self):
        # Rule 2: reading a file must never stop the program that produces it
        # from writing to it.
        _process, seen = self._spawn_capturing(["/data/a.hdf5"])
        self.assertEqual(seen["kwargs"]["env"]["HDF5_USE_FILE_LOCKING"], "FALSE")

    def test_the_child_has_no_streams_to_block_on(self):
        import subprocess

        _process, seen = self._spawn_capturing(["/data/a.aoe"])
        for stream in ("stdin", "stdout", "stderr"):
            self.assertEqual(seen["kwargs"][stream], subprocess.DEVNULL,
                             stream + " was left connected")

    def test_a_frozen_build_is_given_no_import_path_or_directory(self):
        """Both would point inside the bundle, which may not outlive us.

        A one-file PyInstaller build unpacks itself into a temporary directory
        and removes it when the process exits. The sidecar build is detached
        and outlives the window on purpose, so handing it a PYTHONPATH or a
        working directory under that bundle points it at a directory that is
        about to disappear. The frozen executable carries its own imports and
        every path it is passed is absolute, so it needs neither.
        """
        frozen = getattr(sys, "frozen", None)
        sys.frozen = True
        try:
            _process, seen = self._spawn_capturing(["/data/a.aoe"])
        finally:
            if frozen is None:
                del sys.frozen
            else:
                sys.frozen = frozen
        self.assertNotIn("cwd", seen["kwargs"])
        self.assertEqual(seen["kwargs"]["env"].get("PYTHONPATH"),
                         os.environ.get("PYTHONPATH"),
                         "PYTHONPATH was changed for a frozen child")
        # The one thing it must still be given.
        self.assertEqual(seen["kwargs"]["env"]["HDF5_USE_FILE_LOCKING"], "FALSE")

    def test_a_checkout_is_given_both(self):
        _process, seen = self._spawn_capturing(["/data/a.aoe"])
        self.assertIn("cwd", seen["kwargs"])
        self.assertTrue(os.path.isdir(os.path.join(seen["kwargs"]["cwd"],
                                                   "aoeviewer")))

    def test_nothing_is_started_for_nothing(self):
        process, seen = self._spawn_capturing([])
        self.assertIsNone(process)
        self.assertEqual(seen, {})

    def test_a_refused_start_is_not_an_exception(self):
        import subprocess

        def refuse(*_args, **_kwargs):
            raise OSError("no")

        real = subprocess.Popen
        subprocess.Popen = refuse
        try:
            self.assertIsNone(cachebuild.spawn(["/data/a.aoe"]))
        finally:
            subprocess.Popen = real


if __name__ == "__main__":
    unittest.main()
