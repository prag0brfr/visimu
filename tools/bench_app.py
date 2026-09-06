#!/usr/bin/env python3
# Author: Paulo R. A. Gilz
"""End-to-end benchmark over large files.

Measures what the application actually does, not synthetic micro-operations:
opening files, the batched preload a tile rebuild issues, the per-curve fetch and
decimation, ISO scanning, and formula evaluation. Resident memory is sampled at
each step, because on half-gigabyte files memory is as much a limit as time.

    python tools/bench_app.py /path/to/*.aoe
"""

from __future__ import annotations

import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import numpy as np

from aoeviewer import Session
from aoeviewer import session as session_module


def rss_mb() -> float:
    """Resident set size in MB, or 0 where /proc is unavailable."""
    try:
        with open("/proc/self/status", "r") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


class Timer:
    def __init__(self, label: str, scale_mb: float = 0.0):
        self.label = label
        self.scale_mb = scale_mb

    def __enter__(self):
        self.start = time.perf_counter()
        self.rss_before = rss_mb()
        return self

    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self.start
        rss_after = rss_mb()
        throughput = ""
        if self.scale_mb and elapsed > 0:
            throughput = "   {0:7.1f} MB/s".format(self.scale_mb / elapsed)
        print(
            "  {0:<44} {1:8.3f} s{2}   RSS {3:7.0f} MB  (+{4:.0f})".format(
                self.label, elapsed, throughput, rss_after,
                rss_after - self.rss_before,
            )
        )
        self.elapsed = elapsed


def main(paths: List[str]) -> int:
    if not paths:
        print(__doc__)
        return 1

    total_mb = sum(os.path.getsize(p) for p in paths) / (1024.0 * 1024.0)
    print("aoeViewer end-to-end benchmark")
    print("  core {0}, {1} worker threads".format(
        session_module.VERSION, session_module.WORKER_COUNT))
    print("  {0} file(s), {1:.0f} MB total".format(len(paths), total_mb))
    print("  RSS at start: {0:.0f} MB\n".format(rss_mb()))

    session = Session()

    # ---------------------------------------------------------- opening
    print("OPEN  (index scan only, no sample data read)")
    for path in paths:
        size_mb = os.path.getsize(path) / (1024.0 * 1024.0)
        with Timer(os.path.basename(path), size_mb):
            handle = session.add_file(path)
        print("      {0} runs, {1} channels".format(
            handle.simulation_count, len(handle.headers)))

    files = session.files
    total_runs = sum(f.simulation_count for f in files)
    print("\n  {0} runs across {1} files\n".format(total_runs, len(files)))

    # ------------------------------------------------- first plot (cold)
    # What a tile rebuild does: one batched preload per file, then a fetch and
    # a decimation per curve.
    params = ["TIME", "ALPHA", "NZ", "THETA"]
    print("FIRST PLOT  (4 channels x every run, cold)")
    with Timer("preload across all files", total_mb):
        for handle in files:
            available = [p for p in params if p in handle.headers]
            session.preload(handle.index, [], available)

    with Timer("fetch every curve (channel views)"):
        curves = []
        for handle in files:
            for sim in range(handle.simulation_count):
                x = session.channel(handle.index, sim, "TIME")
                for param in params[1:]:
                    y = session.channel(handle.index, sim, param)
                    if x.size and y.size:
                        curves.append((x, y))
    samples = sum(min(x.size, y.size) for x, y in curves)
    print("      {0} curves, {1:.1f} M samples".format(len(curves), samples / 1e6))

    with Timer("decimate every curve to 1200 px"):
        drawn = 0
        for x, y in curves:
            dx, _dy = session.decimate(x, y, 600)
            drawn += dx.size
    print("      {0} points reach the plot widget "
          "({1:.0f}x reduction)".format(drawn, samples / max(1, drawn)))

    # ------------------------------------------------------------ pan/zoom
    print("\nINTERACTION  (what one pan or zoom step costs)")
    if curves:
        x0 = curves[0][0]
        low = float(x0[int(x0.size * 0.40)])
        high = float(x0[int(x0.size * 0.45)])
        with Timer("re-decimate every curve, zoomed to 5%"):
            for x, y in curves:
                session.decimate(x, y, 600, x_range=(low, high))

        with Timer("cursor readout (nearest sample x40)"):
            for _ in range(40):
                for x, _y in curves[:10]:
                    Session.nearest_index(x, low)

    # ---------------------------------------------------------------- ISO
    print("\nISO")
    # Pick ISO candidates that actually exist in these files rather than
    # hard-coding names: reporting "0 distinct values" for a channel the file
    # never had reads like a failure of the ISO scan, which it is not.
    available = session.parameters
    iso_candidates = [
        name for name in ("INITIAL_MASS", "CG_REF", "MACH", "ALT", "CONFIGURATION")
        if name in available
    ][:2] or available[:2]

    for name in iso_candidates:
        with Timer("distinct values of {0} across all files".format(name)):
            values = session.iso_distinct(name)
        print("      {0} distinct values".format(values.size))

    # ------------------------------------------------------------ formula
    print("\nFORMULAS")
    session.set_computed("G_LOAD", "NZ * 9.80665")
    session.set_computed("DRIFT", "ALPHA - ALPHA[TIME=0.0]")
    session.set_computed("NESTED", "G_LOAD / 9.80665 + DRIFT * 0")

    for name in ("G_LOAD", "DRIFT", "NESTED"):
        with Timer("evaluate {0} for every run (cold)".format(name)):
            for handle in files:
                for sim in range(handle.simulation_count):
                    session.channel(handle.index, sim, name)
        with Timer("re-read {0} (cached)".format(name)):
            for handle in files:
                for sim in range(handle.simulation_count):
                    session.channel(handle.index, sim, name)

    # ------------------------------------------------------- worst case
    print("\nWORST CASE  (every channel of every run resident)")
    with Timer("load all channels", total_mb):
        for handle in files:
            session.preload(handle.index, [], handle.headers)
    print("      RSS now {0:.0f} MB for {1:.0f} MB of source text".format(
        rss_mb(), total_mb))

    print("\nDone.  Peak RSS {0:.0f} MB".format(rss_mb()))
    session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
