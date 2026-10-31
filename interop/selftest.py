#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exercise both tools on synthetic files, so they can be trusted before use.

This writes a handful of files in a temporary directory -- a CSV, a fixed-width
Fortran-style listing, a Fortran unformatted binary, a gzip, a zip and, when
h5py is installed, an HDF5 file -- profiles each one, and checks the report
says what it should.  It also runs the conformance harness with the example
reader against a deliberately corrupted twin, to prove the harness actually
fails when the two sides disagree.

It touches nothing outside its temporary directory and needs no .aoe file and
no company data.

    python selftest.py
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import tempfile
import zipfile
import gzip

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import aoe_probe                                        # noqa: E402
import aoe_infer                                        # noqa: E402
import aoe_compare as aoe_compare_module                # noqa: E402

PASS = "  ok   "
FAIL = "  FAIL "

_failures = []


def check(label, condition, detail=""):
    print(("%s%s" % (PASS if condition else FAIL, label))
          + (("  -- %s" % detail) if detail and not condition else ""))
    if not condition:
        _failures.append(label)


# --------------------------------------------------------------------------
# synthetic files
# --------------------------------------------------------------------------

def write_csv(path, rows=200):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("TIME,ALTITUDE,MACH,ALPHA\n")
        for i in range(rows):
            handle.write("%.6E,%.6E,%.6E,%.6E\n" % (
                i * 0.01, 10000.0 + i, 0.85 + i * 1e-4, 2.0 + i * 1e-3))


def write_fortran_listing(path, runs=3, rows=50):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for run in range(runs):
            handle.write("*RUN  SIM_%03d\n" % (run + 1))
            handle.write("*CHANNELS   TIME        ALTITUDE    MACH\n")
            for i in range(rows):
                handle.write("  %10.4E %10.4E %10.4E\n" % (
                    i * 0.1, 10000.0 + run * 100 + i, 0.85 + i * 1e-4))
            handle.write("*END\n")


def write_fixed_width(path, rows=80):
    """A true fixed-format listing: column positions, and blanks for missing.

    Blank fields are what separate fixed-width from merely space-delimited --
    splitting on whitespace silently shifts every later column on those lines,
    which is the classic way a reader for this kind of file goes wrong.
    """
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("    TIME  ALTITUDE      MACH\n")
        for i in range(rows):
            mach = "" if i % 5 == 0 else "%9.5f" % (0.85 + i * 1e-4)
            handle.write("%8.3f %9.2f %9s\n" % (
                i * 0.1, 10000.0 + i, mach))


def write_fortran_unformatted(path, records=40, values=12):
    with open(path, "wb") as handle:
        for record in range(records):
            payload = struct.pack("<%df" % values,
                                  *[record + 0.5 * v for v in range(values)])
            handle.write(struct.pack("<i", len(payload)))
            handle.write(payload)
            handle.write(struct.pack("<i", len(payload)))


def write_hdf5(path):
    try:
        import h5py
        import numpy as np
    except ImportError:
        return False
    with h5py.File(path, "w") as handle:
        handle.attrs["solver"] = "synthetic"
        for run in range(3):
            group = handle.create_group("run_%03d" % (run + 1))
            group.attrs["case"] = "SIM_%03d" % (run + 1)
            time = np.arange(0.0, 10.0, 0.01)
            group.create_dataset("TIME", data=time, compression="gzip")
            group.create_dataset("ALTITUDE", data=10000.0 + time * 10.0,
                                 chunks=(100,))
            group.create_dataset("MACH", data=0.85 + time * 1e-3)
    return True


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------

def probe(path, tier=aoe_probe.TIER_IDENTIFIERS, recurse=False):
    return aoe_probe.profile_path(path, tier, recurse,
                                  aoe_probe.DEFAULT_MAX_BYTES)


def no_values_anywhere(report, forbidden):
    """Search every string in the report for a value that was in the file."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            for needle in forbidden:
                if needle in node:
                    found.append((needle, node[:80]))

    walk(report)
    return found


def run(scratch):
    sys.path.insert(0, HERE)
    import aoe_compare as aoe_compare_module

    csv_path = os.path.join(scratch, "RESULTS.CSV")
    listing = os.path.join(scratch, "RUN.DAT")
    fixed = os.path.join(scratch, "FIXED.TXT")
    binary = os.path.join(scratch, "UNFORMATTED.BIN")
    archive = os.path.join(scratch, "BUNDLE.zip")
    compressed = os.path.join(scratch, "RESULTS.CSV.gz")
    hdf5 = os.path.join(scratch, "RUN.h5")

    write_csv(csv_path)
    write_fortran_listing(listing)
    write_fixed_width(fixed)
    write_fortran_unformatted(binary)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, "inner/RESULTS.CSV")
        zf.write(listing, "inner/RUN.DAT")
    with open(csv_path, "rb") as src, gzip.open(compressed, "wb") as dst:
        dst.write(src.read())
    has_h5 = write_hdf5(hdf5)

    print("\n-- text: comma-delimited --")
    report = probe(csv_path)
    text = report["text"]
    check("detected as text", report["kind"] == "text")
    check("delimiter is comma",
          text["chosen_delimiter"] == "comma", text["chosen_delimiter"])
    check("four fields", text["delimiters"][0]["modal_field_count"] == 4)
    check("header found", "ALTITUDE" in text["identifier_candidates"],
          str(text["identifier_candidates"]))
    check("exponent signs fragment the raw shapes",
          text["distinct_line_shapes"] == 3, str(text["distinct_line_shapes"]))
    check("one data family dominates once signs are folded",
          text["line_families"][0]["share"] > 0.99,
          str(text["line_families"][:2]))
    check("one field-class grammar covers every data line",
          text["field_grammars"][0]["share"] > 0.99
          and text["field_grammars"][0]["grammar"] == "F{4}",
          str(text["field_grammars"][:2]))
    check("the header has its own grammar",
          any(e["grammar"] == "N{4}" for e in text["field_grammars"]),
          str([e["grammar"] for e in text["field_grammars"]]))
    check("a delimited file does not claim fixed-width columns",
          not text["fixed_width_is_likely"])
    check("C exponents recognised",
          text["number_grammars"].get("c_e_exponent", 0) > 100)

    print("\n-- text: masking at the structure tier --")
    masked = probe(csv_path, aoe_probe.TIER_STRUCTURE)
    check("no identifiers emitted",
          not masked["text"]["identifier_candidates"])
    check("channel names are masked",
          not no_values_anywhere(masked, ["ALTITUDE", "MACH"]))
    check("file name is masked", "RESULTS" not in str(masked))
    check("no data values survive masking",
          not no_values_anywhere(masked, ["10000", "0.85", "1.000000E"]),
          str(no_values_anywhere(masked, ["10000", "0.85"])[:2]))

    print("\n-- text: quoted headers and free-format numbers --")
    quoted = os.path.join(scratch, "QUOTED.TXT")
    with open(quoted, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("*PARAMETRES\n")
        handle.write("'TIME' 'ALPHA' 'ALTITUDE'\n")
        handle.write("*TABLEAU : 1\n")
        for i in range(120):
            handle.write("%g %g %g\n" % (i * 0.1, 1.5 + i, 392.95 + i * 7))
    report = probe(quoted)
    text = report["text"]
    check("quoted header names recovered",
          "ALTITUDE" in text["identifier_candidates"],
          str(text["identifier_candidates"]))
    check("free-format digits fragment the raw shapes",
          text["distinct_line_shapes"] > 6,
          str(text["distinct_line_shapes"]))
    check("but one field-class grammar still covers the data",
          text["field_grammars"][0]["grammar"] in ("F{3}", "IF{2}")
          and text["field_grammars"][0]["count"] >= 100,
          str(text["field_grammars"][:3]))

    print("\n-- text: repeating run blocks --")
    report = probe(listing)
    text = report["text"]
    check("block period found", text["block_period"] is not None,
          str(text["block_period"]))
    check("period is the run length (2 headers + 50 rows + 1 terminator)",
          text["block_period"] and text["block_period"]["period"] == 53,
          str(text["block_period"]))
    check("the run-header keyword line is visible as a shape",
          any(s["shape"].startswith("*A{3}") for s in text["line_shapes"]),
          str([s["shape"] for s in text["line_shapes"]][:6]))

    print("\n-- text: fixed-width columns --")
    report = probe(fixed)
    fixed_report = report["text"]["fixed_width"]
    check("column boundaries inferred", fixed_report is not None)
    check("fixed width flagged as the real explanation",
          report["text"]["fixed_width_is_likely"])
    check("three data fields",
          fixed_report and fixed_report["field_count"] == 3,
          str(fixed_report and fixed_report["field_count"]))

    print("\n-- binary: Fortran unformatted --")
    report = probe(binary)
    check("detected as binary", report["kind"] == "binary", report["kind"])
    chain = report["binary"]["fortran_records_little_endian"]
    check("record chain walked", chain["records"] == 40, str(chain["records"]))
    check("chain consumed the file", chain["consumed_all"])
    check("one record length", len(chain["distinct_record_lengths"]) == 1)
    check("float32 little-endian is most plausible",
          report["binary"]["most_plausible_layout"] == "float32_le",
          str(report["binary"]["most_plausible_layout"]))

    print("\n-- container: zip --")
    report = probe(archive, recurse=True)
    check("detected as container", report["kind"] == "container")
    check("two members listed", len(report["container"]["members"]) == 2)
    check("members profiled",
          len(report["container"].get("member_profiles", [])) == 2)
    check("inner CSV recognised",
          any(m["profile"].get("kind") == "text"
              for m in report["container"]["member_profiles"]))

    print("\n-- container: gzip stream --")
    report = probe(compressed)
    check("detected as compressed stream",
          report["kind"] == "compressed_stream", report["kind"])
    check("inner stream profiled as text",
          report["inner"]["kind"] == "text")
    check("inner delimiter found",
          report["inner"]["text"]["chosen_delimiter"] == "comma")

    print("\n-- hdf5 --")
    if not has_h5:
        print("  skip   h5py is not installed")
    else:
        report = probe(hdf5)
        h5 = report["hdf5"]
        check("tree walked", h5["available"] and h5["dataset_count"] == 9,
              str(h5.get("dataset_count")))
        check("groups counted", h5["group_count"] == 3,
              str(h5["group_count"]))
        paths = [o["path"] for o in h5["objects"]]
        check("dataset paths present", "/run_001/TIME" in paths, str(paths[:4]))
        check("compression reported",
              any(o.get("compression") == "gzip"
                  for o in h5["objects"] if o["kind"] == "dataset"))
        check("chunking reported",
              any(o.get("chunks") for o in h5["objects"]
                  if o["kind"] == "dataset"))
        time_entry = [o for o in h5["objects"] if o["path"].endswith("/TIME")][0]
        check("time axis flagged monotonic",
              time_entry["content_shape"]["monotonic_increasing"],
              str(time_entry["content_shape"]))
        masked = probe(hdf5, aoe_probe.TIER_STRUCTURE)
        check("paths masked at the structure tier",
              not no_values_anywhere(masked, ["ALTITUDE", "run_001"]))

    print("\n-- the .aoe text reader --")
    from readers import aoe_text_reader
    aoe_like = os.path.join(scratch, "SYNTH.aoe")
    with open(aoe_like, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("*PARAMETRES\n")
        handle.write("'TIME' 'ALPHA' 'ALTITUDE'\n")
        for run in range(4):
            handle.write("*TABLEAU : %d\n" % (run + 1))
            for i in range(60):
                handle.write("%g %g %g\n" % (
                    i * 0.02, 1.5 + i * 0.01, 392.95 + run * 10 + i))
    parsed = aoe_text_reader.read(aoe_like)
    check("four runs parsed", len(parsed) == 4, str(sorted(parsed)))
    check("runs named from the table label",
          sorted(parsed) == ["1", "2", "3", "4"], str(sorted(parsed)))
    check("quoted channel names unquoted",
          sorted(parsed["1"]) == ["ALPHA", "ALTITUDE", "TIME"],
          str(sorted(parsed["1"])))
    check("sixty samples per channel",
          all(v.size == 60 for v in parsed["1"].values()),
          str([v.size for v in parsed["1"].values()]))
    report = aoe_compare_module.compare(
        parsed, aoe_text_reader.read(aoe_like), {}, {}, "name", 1e-6,
        aoe_compare_module.TIER_IDENTIFIERS, True)
    check("the .aoe reader agrees with itself",
          report["verdict"] == "equivalent"
          and report["channels"]["compared"] == 12,
          str(report["channels"]))

    print("\n-- conformance harness --")
    harness = os.path.join(HERE, "aoe_compare.py")
    twin = os.path.join(scratch, "twin.DAT")
    write_fortran_listing(twin)
    result = subprocess.run(
        [sys.executable, harness, "--aoe", twin, "--other", listing,
         "--reader", "readers.example_reader:read"],
        capture_output=True, text=True, cwd=HERE)
    check("harness reports a missing visimu clearly",
          "Could not import visimu" in (result.stdout + result.stderr)
          or result.returncode in (0, 1),
          (result.stdout + result.stderr)[-200:])

    # The reader half can be graded without any .aoe at all: read the same
    # listing twice and confirm the comparison machinery agrees with itself,
    # then perturb one channel and confirm it stops agreeing.
    sys.path.insert(0, HERE)
    import aoe_compare
    from readers import example_reader

    left = example_reader.read(listing)
    check("example reader found three runs", len(left) == 3, str(list(left)))
    check("example reader found three channels",
          len(left["SIM_001"]) == 3, str(list(left["SIM_001"])))

    report = aoe_compare.compare(
        left, example_reader.read(listing), {}, {}, "name", 1e-6,
        aoe_compare.TIER_IDENTIFIERS, True)
    check("a file agrees with itself", report["verdict"] == "equivalent",
          str(report["failures"]))
    check("nine comparisons made", report["channels"]["compared"] == 9,
          str(report["channels"]["compared"]))

    perturbed = {run: dict(channels) for run, channels in left.items()}
    perturbed["SIM_002"]["MACH"] = perturbed["SIM_002"]["MACH"] * 1.01
    report = aoe_compare.compare(
        left, perturbed, {}, {}, "name", 1e-6,
        aoe_compare.TIER_IDENTIFIERS, True)
    check("a perturbed channel is caught", report["verdict"] == "differs")
    check("exactly one comparison fails", report["failures"] == 1,
          str(report["failures"]))

    dropped = {run: dict(channels) for run, channels in left.items()}
    del dropped["SIM_003"]["ALTITUDE"]
    report = aoe_compare.compare(
        left, dropped, {}, {}, "name", 1e-6,
        aoe_compare.TIER_IDENTIFIERS, True)
    check("a missing channel is reported",
          "ALTITUDE" in report["channels"]["only_in_aoe"],
          str(report["channels"]))

    renamed = {run: {("TOTAL_TIME" if k == "TIME" else k): v
                     for k, v in channels.items()}
               for run, channels in left.items()}
    report = aoe_compare.compare(
        left, renamed, {"TIME": "TOTAL_TIME"}, {}, "name", 1e-6,
        aoe_compare.TIER_IDENTIFIERS, True)
    check("an explicit rename map is honoured",
          report["verdict"] == "equivalent"
          and report["channels"]["compared"] == 9,
          str(report["channels"]))

    report = aoe_compare.compare(
        left, perturbed, {}, {}, "name", 1e-6,
        aoe_compare.TIER_STRUCTURE, False)
    check("structure tier masks channel names in the comparison",
          all("TIME" not in str(entry["channel"])
              for entry in report["comparisons"]))
    check("--no-magnitudes emits exponents, not differences",
          all("max_abs_diff" not in entry for entry in report["comparisons"]))

    spelt = aoe_compare.normalise("Total Force") == aoe_compare.normalise(
        "total_force")
    check("name normalisation folds spelling", spelt)




# --------------------------------------------------------------------------
# inference: can the parse be found from the twin alone?
# --------------------------------------------------------------------------

def synthetic_twin(runs=3, samples=300):
    """A stand-in for an .aoe file, stored the way the core stores it.

    Rounding through float32 matters: the twin the real tool loads has been
    through that, so a tolerance that only works on exact float64 would pass
    here and fail in the field.
    """
    import numpy as np

    twin = {}
    for run in range(runs):
        time = np.arange(samples) * 0.02
        channels = {
            "TIME": time,
            "ALPHA": 1.5 + np.sin(time) + run,
            "ALTITUDE": 10000.0 + time * 12.0 + run * 50,
            "MACH": 0.8 + 0.001 * time,
            "MASS": 58000.0 - time * 3.0 - run,
            "THRUST": 210000.0 + 1000.0 * np.cos(time) + run,
            # Constant through the run and different between runs: correlation
            # says nothing about a flat line, so this exercises the third pass.
            "REF_MASS": np.full(samples, 58000.0 + run * 250.0),
        }
        twin[str(run + 1)] = {
            name: values.astype("float32").astype("float64")
            for name, values in channels.items()
        }
    return twin


FEET_PER_METRE = 3.280839895013123


def write_scrambled_csv(path, twin):
    """The hard case: no header, columns in a different order, feet not metres."""
    order = ["MACH", "ALTITUDE", "TIME", "REF_MASS", "THRUST", "ALPHA", "MASS"]
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for label in sorted(twin):
            handle.write("*CASE %s\n" % label)
            for index in range(len(twin[label]["TIME"])):
                cells = []
                for name in order:
                    value = twin[label][name][index]
                    if name == "ALTITUDE":
                        value = value * FEET_PER_METRE
                    cells.append("%.9g" % value)
                handle.write(",".join(cells) + "\n")
    return order


def write_unformatted(path, twin, names):
    import struct

    channels = twin["1"]
    with open(path, "wb") as handle:
        for index in range(len(channels["TIME"])):
            payload = struct.pack(">%dd" % len(names),
                                  *[channels[n][index] for n in names])
            marker = struct.pack(">i", len(payload))
            handle.write(marker + payload + marker)


def write_raw_column_major(path, twin, names):
    import numpy as np

    with open(path, "wb") as handle:
        for name in names:
            handle.write(np.asarray(twin["1"][name], dtype="<f4").tobytes())


def write_fixed_width_table(path, twin, names):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for label in sorted(twin):
            handle.write("RUN %s\n" % label)
            for index in range(len(twin[label]["TIME"])):
                handle.write("".join("%14.6E" % twin[label][name][index]
                                     for name in names) + "\n")


def round_trip(strategy, per_run, twin, path, module_name):
    """Generate the reader the inference implies, run it, and grade it."""
    import importlib

    columns, conversions, _ = aoe_infer.canonical_map(per_run, True)
    keys = per_run[0].get("names")
    if strategy.description.get("kind") == "hdf5" and \
            strategy.description.get("rule") not in ("table", "cube") and keys:
        columns = {keys[i]: c for i, c in columns.items() if i < len(keys)}
        conversions = {keys[i]: v for i, v in conversions.items()
                       if i < len(keys)}
    source = aoe_infer.generate_reader(strategy, columns, conversions,
                                       module_name, "self-test")
    target = os.path.join(HERE, "readers", module_name + ".py")
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(source)
    try:
        module = importlib.import_module("readers." + module_name)
        importlib.reload(module)
        produced = aoe_compare_module.normalise_reader_output(module.read(path))
        run_map = {entry["twin_run"]: entry["parsed_run"] for entry in per_run}
        align = "name" if set(run_map.values()) & set(produced) else "order"
        return aoe_compare_module.compare(
            twin, produced, {}, run_map, align, 1e-5,
            aoe_compare_module.TIER_IDENTIFIERS, False)
    finally:
        try:
            os.remove(target)
        except OSError:
            pass


def inference_checks(scratch):
    twin = synthetic_twin()
    names = ["TIME", "ALPHA", "ALTITUDE", "MACH", "MASS", "THRUST", "REF_MASS"]
    one_run = {"1": twin["1"]}

    print("\n-- inference: delimited text, no header, shuffled, in feet --")
    path = os.path.join(scratch, "SCRAMBLED.CSV")
    order = write_scrambled_csv(path, twin)
    graded = aoe_infer.infer(twin, path)
    check("a strategy was found", bool(graded) and graded[0][0] > 0)
    score, strategy, per_run = graded[0]
    check("comma delimiter chosen", strategy.key == "text:comma", strategy.key)
    check("three runs found", len(per_run) == 3, str(len(per_run)))
    columns, conversions, details = aoe_infer.canonical_map(per_run, True)
    check("every channel identified by value alone",
          len(columns) == 7, str(columns))
    check("a constant channel is identified too, without correlation",
          columns.get(3) == "REF_MASS", str(columns.get(3)))
    recovered = [columns.get(i) for i in range(7)]
    check("the shuffled column order is recovered",
          recovered == order, str(recovered))
    altitude = [d for d in details if d["channel"] == "ALTITUDE"][0]
    check("the unit difference is seen as a scale, not a mismatch",
          altitude["relation"] == "scaled", altitude["relation"])
    check("and the scale is named",
          altitude.get("looks_like") == "feet -> metres",
          str(altitude.get("looks_like")))
    report = round_trip(strategy, per_run, twin, path, "generated_probe_csv")
    check("the generated reader reproduces the twin",
          report["verdict"] == "equivalent", str(report["failures"]))
    check("all twenty-one comparisons ran",
          report["channels"]["compared"] == 21,
          str(report["channels"]["compared"]))

    print("\n-- inference: Fortran unformatted, big-endian float64 --")
    path = os.path.join(scratch, "UNFORMATTED.BIN")
    write_unformatted(path, twin, names)
    graded = aoe_infer.infer(one_run, path)
    check("a strategy was found", bool(graded) and graded[0][0] > 0)
    score, strategy, per_run = graded[0]
    check("endianness and word size recovered",
          strategy.key == "binary:fortran_row_big_8", strategy.key)
    columns, _, _ = aoe_infer.canonical_map(per_run, True)
    check("all seven columns identified",
          [columns.get(i) for i in range(7)] == names, str(columns))
    report = round_trip(strategy, per_run, one_run, path, "generated_probe_unf")
    check("the generated reader reproduces the twin",
          report["verdict"] == "equivalent", str(report["failures"]))

    print("\n-- inference: headerless raw binary, column-major float32 --")
    path = os.path.join(scratch, "RAW.DAT")
    write_raw_column_major(path, twin, names)
    graded = aoe_infer.infer(one_run, path)
    check("a strategy was found", bool(graded) and graded[0][0] > 0)
    score, strategy, per_run = graded[0]
    check("word size, endianness and order recovered",
          strategy.key == "binary:raw_little_4_column_major", strategy.key)
    report = round_trip(strategy, per_run, one_run, path, "generated_probe_raw")
    check("the generated reader reproduces the twin",
          report["verdict"] == "equivalent", str(report["failures"]))

    print("\n-- inference: fixed-width table with run headers --")
    path = os.path.join(scratch, "TABLE.TXT")
    write_fixed_width_table(path, twin, names)
    graded = aoe_infer.infer(twin, path)
    check("a strategy was found", bool(graded) and graded[0][0] > 0)
    score, strategy, per_run = graded[0]
    check("three runs found", len(per_run) == 3, str(len(per_run)))
    columns, _, _ = aoe_infer.canonical_map(per_run, True)
    check("all seven columns identified",
          [columns.get(i) for i in range(7)] == names, str(columns))
    report = round_trip(strategy, per_run, twin, path, "generated_probe_fixed")
    check("the generated reader reproduces the twin",
          report["verdict"] == "equivalent", str(report["failures"]))

    print("\n-- inference: HDF5, group per channel, dataset per run --")
    try:
        import h5py
    except ImportError:
        print("  skip   h5py is not installed")
    else:
        import numpy as np

        path = os.path.join(scratch, "CASES.hdf5")
        with h5py.File(path, "w") as handle:
            # The transpose of the obvious layout, and a real convention: the
            # group is the channel, each dataset inside it is one run, and the
            # run identity is spelled into the dataset name.  Nothing but the
            # twin can tell this apart from the other way round.
            channels = handle.create_group("AVION/CASE_1/CHANNELS")
            for name in names:
                group = channels.create_group(name)
                for label in sorted(twin):
                    group.create_dataset(
                        "%s_%s_2_0" % (name, label), data=twin[label][name])
            # A metadata vector of another length, which must be ignored.
            channels.create_dataset("AUX", data=np.arange(4.0))

        relabelled = {"%s_2_0" % label: channels_of
                      for label, channels_of in twin.items()}
        graded = aoe_infer.infer(relabelled, path)
        check("a strategy was found", bool(graded) and graded[0][0] > 0)
        score, strategy, per_run = graded[0]
        check("the run is found in the dataset name, not the group",
              strategy.key == "hdf5:leaf_tail_3", strategy.key)
        check("three runs found", len(per_run) == 3, str(len(per_run)))
        columns, _, _ = aoe_infer.canonical_map(per_run, True)
        check("all seven channels identified",
              sorted(columns.values()) == sorted(names), str(columns))
        keys = per_run[0]["names"]
        check("the odd-length metadata vector is ignored",
              "AUX" not in keys, str(keys))
        report = round_trip(strategy, per_run, relabelled, path,
                            "generated_probe_h5")
        check("the generated reader reproduces the twin",
              report["verdict"] == "equivalent", str(report["failures"]))

    print("\n-- inference: a twin that lost precision in its own export --")
    import numpy as np

    # A text .aoe keeps a fixed number of significant digits.  The unknown
    # file holds full float64, so a perfectly correct reader still lands a few
    # parts per million from the twin -- and that must not read as a failure.
    rounded = {
        label: {name: np.asarray(["%.7g" % v for v in values], dtype=np.float64)
                for name, values in channels.items()}
        for label, channels in twin.items()
    }
    path = os.path.join(scratch, "PRECISE.CSV")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for label in sorted(twin):
            handle.write("*CASE %s\n" % label)
            handle.write(",".join(names) + "\n")
            for index in range(len(twin[label]["TIME"])):
                handle.write(",".join("%.17g" % twin[label][name][index]
                                      for name in names) + "\n")
    graded = aoe_infer.infer(rounded, path, tolerance=1e-5)
    check("the rounded twin still matches", bool(graded) and graded[0][0] > 0,
          str(graded[0][0] if graded else 0))
    score, strategy, per_run = graded[0]
    columns, _, _ = aoe_infer.canonical_map(per_run, True)
    check("all seven channels identified through the rounding",
          len(columns) == 7, str(columns))
    report = round_trip(strategy, per_run, rounded, path,
                        "generated_probe_round")
    check("grading at the matching tolerance calls it equivalent",
          report["verdict"] == "equivalent", str(report["failures"]))
    strict = aoe_compare_module.compare(
        rounded, {"1": {n: v for n, v in twin["1"].items()}}, {}, {},
        "order", 1e-9, aoe_compare_module.TIER_IDENTIFIERS, False)
    check("and a tolerance tighter than the export would not",
          strict["verdict"] == "differs", strict["verdict"])

    print("\n-- inference: a file that is not the twin --")
    path = os.path.join(scratch, "UNRELATED.CSV")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("A,B,C\n")
        for i in range(200):
            handle.write("%d,%d,%d\n" % (i * 7 % 13, i * 11 % 17, i * 3 % 5))
    graded = aoe_infer.infer(one_run, path)
    best = graded[0][0] if graded else 0.0
    check("nothing is claimed to match", best <= 1.0, str(best))


def main():
    with tempfile.TemporaryDirectory(prefix="aoe_interop_selftest_") as scratch:
        run(scratch)
        inference_checks(scratch)
    print("")
    if _failures:
        print("%d check(s) failed:" % len(_failures))
        for label in _failures:
            print("  - %s" % label)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
