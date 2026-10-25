# Author: Paulo R. A. Gilz
"""Write a large HDF5 result file, in the layout the real ones use.

    python tools/gen_hdf5_sample.py big.hdf5 --runs 50 --channels 1000 --samples 7500

That is about 3 GB. By default it writes the structure the real files use --
a session holding a PARAMETERS group that names the channels and a SIMULATIONS
group that names the runs:

    /DOMAIN/OSMA_1/PARAMETERS/ALFA              @unit=deg
    /DOMAIN/OSMA_1/PARAMETERS/ALFA/ALFA_1_2_0   the samples
    /DOMAIN/OSMA_1/PARAMETERS/ALFA/ALFA_2_2_0
    /DOMAIN/OSMA_1/SIMULATIONS/SIMULATION_1_2_0 the run, with its sweep
    /DOMAIN/OSMA_1/SIMULATIONS/SIMULATION_2_2_0

so a file written here exercises the same reader, the same lazy column reads
and the same sidecar path as a real one.

``--sessions N`` writes several sessions under DOMAIN, which is worth testing:
every session repeats the same run names, and a reader that keys on the
dataset name alone cannot tell them apart.

``--flat`` writes the other convention instead -- a bare group per channel,
with nothing declaring what is a run -- which is what the layout inference is
for.

Needs h5py. On Windows that is the interpreter you build the executable
from -- the one whose ``pip list`` shows h5py -- not necessarily the one on
PATH.

Storage matters as much as size, which is why it is an option. A solver
writing contiguous datasets and one writing chunked, compressed ones produce
files of very different read cost, and if an open is slow it is worth knowing
which you have:

    --chunked            store each dataset chunked rather than contiguous
    --compress N         gzip level N (implies --chunked); slower to read

Use ``--quick`` for a small one to check the thing works before committing to
a multi-gigabyte write.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

CHANNEL_NAMES = ("TIME", "ALPHA", "MACH", "NZ", "ALT", "THETA", "SPEED",
                 "MASSE", "THRUST", "PHI", "PSI", "BETA")
UNITS = {"TIME": "s", "ALPHA": "deg", "MACH": "", "NZ": "", "ALT": "ft",
         "THETA": "deg", "SPEED": "m/s", "MASSE": "kg", "THRUST": "N",
         "PHI": "deg", "PSI": "deg", "BETA": "deg"}


def channel_names(count):
    """Readable names first, then CH0012-style ones to make up the number."""
    names = list(CHANNEL_NAMES[:count])
    names += ["CH%04d" % i for i in range(len(names), count)]
    return names


def column(numpy, name, run, samples):
    """One channel of one run. Flight-shaped rather than noise, so a plot of
    it looks like something and a decimated curve is visibly right."""
    t = numpy.arange(samples, dtype=numpy.float64) * 0.02
    if name == "TIME":
        return t
    # Each run starts its manoeuvre at a different moment, which is what the
    # SYNC panel is for.
    start = 2.0 + 0.37 * run
    ramp = 1.0 / (1.0 + numpy.exp(-(t - start)))
    if name == "MASSE":
        return 61000.0 - 100.0 * run - 0.8 * t
    if name == "ALT":
        return 1000.0 + 300.0 * run + 8000.0 * ramp
    if name == "SPEED":
        return 100.0 * ramp
    if name == "MACH":
        return 0.70 + 0.05 * (run % 3) + 0.25 * ramp
    phase = 0.1 * (hash(name) % 17)
    return numpy.sin(t * 0.5 + phase) * (1.0 + 0.1 * run) + ramp


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Write a large HDF5 result file for testing.")
    parser.add_argument("output")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--channels", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=7500)
    parser.add_argument("--chunked", action="store_true",
                        help="store chunked rather than contiguous")
    parser.add_argument("--compress", type=int, default=0, metavar="N",
                        help="gzip level 1-9; implies --chunked")
    parser.add_argument("--sessions", type=int, default=1,
                        help="how many sessions under DOMAIN (default 1)")
    parser.add_argument("--flat", action="store_true",
                        help="write a bare group-per-channel tree instead, "
                             "with no PARAMETERS or SIMULATIONS groups, so "
                             "the layout has to be inferred")
    parser.add_argument("--quick", action="store_true",
                        help="a small file: 4 runs, 12 channels, 2000 samples")
    args = parser.parse_args(argv)

    if args.quick:
        args.runs, args.channels, args.samples = 4, 12, 2000

    try:
        import numpy
    except ImportError:
        print("needs numpy")
        return 1
    try:
        import h5py
    except ImportError:
        print("needs h5py. On Windows, run this with the interpreter you "
              "build the executable from -- the one whose 'pip list' shows "
              "h5py -- rather than whichever python is on PATH.")
        return 1

    datasets = args.runs * args.channels
    expected = datasets * args.samples * 8 * max(1, args.sessions)
    print("%d session(s) x %d runs x %d channels x %d samples" % (
        max(1, args.sessions), args.runs, args.channels, args.samples))
    print("%d datasets, about %.2f GB%s" % (
        datasets * max(1, args.sessions), expected / 1e9,
        " before compression" if args.compress else ""))

    names = channel_names(args.channels)
    # Run keys of the X_Y_Z form real files use; the reader takes the last
    # three tokens of the dataset name as the run.
    runs = ["%d_%d_%d" % (1 + run // 100, 1 + (run // 10) % 10, run % 10)
            for run in range(args.runs)]
    if len(set(runs)) != len(runs):
        print("too many runs for the X_Y_Z naming (max 1000)")
        return 1

    options = {}
    if args.compress:
        options["compression"] = "gzip"
        options["compression_opts"] = args.compress
        options["chunks"] = True
    elif args.chunked:
        options["chunks"] = True

    started = time.time()
    written = 0
    sessions = max(1, args.sessions)
    total = datasets * sessions

    with h5py.File(args.output, "w") as handle:
        if args.flat:
            holders = [handle.create_group("ROOT/CASE")]
        else:
            domain = handle.create_group("DOMAIN")
            holders = []
            for number in range(sessions):
                session = domain.create_group("OSMA_%d" % (number + 1))
                session.attrs["name"] = "OSMA_%d" % (number + 1)
                session.attrs["kernel"] = "V7.4"
                holders.append(session.create_group("PARAMETERS"))

                # The runs, named and described, which is what makes the file
                # readable without guessing.
                simulations = session.create_group("SIMULATIONS")
                for run_index, run_key in enumerate(runs):
                    entry = simulations.create_group("SIMULATION_%s" % run_key)
                    entry.attrs["name"] = "SIMULATION_%s" % run_key
                    entry.attrs["simulationName"] = "SIMULATION_%s" % run_key
                    entry.attrs["sweepParameters"] = ["sw_MASSE", "sw_CG"]
                    entry.attrs["sweepUnits"] = ["", ""]
                    entry.attrs["sweepValues"] = [
                        "%f" % (210.0 + run_index), "%f" % (17.0 + run_index)]
                    entry.attrs["WEIGHT(KG)"] = "%.5E" % (210000.0 + run_index)

        for holder in holders:
            for index, name in enumerate(names):
                group = holder.create_group(name)
                if not args.flat:
                    # The channel declares itself here, once, rather than once
                    # per run on every dataset.
                    group.attrs["name"] = name
                    group.attrs["unit"] = UNITS.get(name, "")
                    group.attrs["type"] = "REAL"
                    group.attrs["dimension"] = 0
                for run_index, run_key in enumerate(runs):
                    values = column(numpy, name, run_index, args.samples)
                    dataset = group.create_dataset(
                        "%s_%s" % (name, run_key), data=values, **options)
                    dataset.attrs["unit"] = UNITS.get(name, "")
                    dataset.attrs["units"] = UNITS.get(name, "")
                    dataset.attrs["name"] = "%s_%s" % (name, run_key)
                    written += 1
                if index % 50 == 0 or index == len(names) - 1:
                    print("  %5.1f%%  %d/%d datasets  %.0f s"
                          % (100.0 * written / total, written, total,
                             time.time() - started), flush=True)

    size = os.path.getsize(args.output)
    print("wrote %s -- %.2f GB in %.0f s"
          % (args.output, size / 1e9, time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
