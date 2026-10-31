# interop — describing a format nobody may send you

This folder is a self-contained toolkit for the situation where visimu has
to read a proprietary result format, but no sample of that format may leave the
site where it lives.

Nothing here is part of the application. One script imports
`../python/visimu` read-only, for one purpose (reading an `.aoe` file the way
the app reads it). No application file, build script or test refers to this
folder, and deleting it changes nothing about the app.

| File | What it does |
| --- | --- |
| `aoe_infer.py` | **Start here when you have both files.** Works out how to read the unknown format by verifying candidate parses against the `.aoe` twin, identifies every column by its values, and writes a working reader. One run, no iteration. |
| `aoe_probe.py` | Reads one unknown file and writes a report describing its *structure*, never its contents. This is the report that travels. |
| `aoe_compare.py` | Grades a candidate reader for the unknown format against an `.aoe` export of the same simulation. Runs entirely on site; its report is abstract. |
| `readers/example_reader.py` | The reader protocol, worked through on a Fortran-style listing. Copy this to write a real one. |
| `readers/aoe_text_reader.py` | A reader for the `.aoe` text format itself. Grading it against the core's own parser is the end-to-end proof that the harness works. |
| `selftest.py` | Exercises all three tools on synthetic files. Needs no company data and no `.aoe`. Run it first. |

```
python selftest.py          # confirm the tools work here
```

## The short version

If you can export one simulation twice — once in the ad-hoc format, once as
`.aoe` — this is the whole procedure:

```
python aoe_infer.py --aoe RUN.aoe --other RUN.DAT -o inferred.json
```

That one command finds the parse, identifies every column, writes
`readers/generated_run_reader.py`, runs it, and grades it against the twin.
Send `inferred.json`. Everything below is the longer road, and the detail
behind what that command does.

Python 3.8+ and the standard library. `numpy` is required by `aoe_compare.py`
only. `h5py` is used by `aoe_probe.py` when present, and its absence only means
an HDF5 file is described at the container level instead of tree level.

---

## Step 1 — profile the unknown file (no twin available)

```
python aoe_probe.py  RESULTS.DAT                     -o profile.json
python aoe_probe.py  RUN.h5   --tier identifiers     -o profile.json
python aoe_probe.py  BUNDLE.zip --recurse            -o profile.json
```

The script decides for itself what it is looking at — delimited text,
fixed-width text, packed binary, Fortran unformatted, HDF5, zip, tar, gzip,
bzip2, xz — and profiles accordingly, recursing into containers and
transparently through single-stream compression. Large files are sampled in
three windows (head, middle, tail), so a layout that changes part-way through
still shows up.

What comes out, by kind of file:

**Text.** The encoding, the byte-order mark, the mix of line endings. The field
grammars and line shapes described below. The delimiter candidates ranked by how
consistent their field count is. The inferred fixed-width column boundaries when
there is no delimiter at all. The numeric dialects in use — C `1.0E+03`, Fortran
`1.0D+03`, and the Fortran habit of dropping the exponent letter entirely
(`1.0+03`), which `float()` rejects and a reader must handle. The repeating block
period, which is where one run ends and the next begins. Quoted headers
(`'TIME' 'ALPHA'`) are unquoted before anything else happens to them.

**HDF5.** The group and dataset tree with shape, dtype, item size, chunk shape,
compression filter and options, shuffle and checksum flags, compound field
layout, dimension-scale membership, and attribute names. Per dataset it also
reports three facts that identify an axis without disclosing one: how many NaNs
and infinities there are, whether every entry is a whole number, and whether the
first column rises monotonically.

**Binary.** The magic bytes, the null-byte fraction, and a walk of the Fortran
unformatted record-marker chain in both endiannesses — `[int32 n][n bytes][int32
n]`; if the chain consumes the whole file, the layout is settled and the record
length gives the field count directly. Then, for each of float32/float64 ×
little/big endian, what fraction of words decode as finite numbers in an
engineering magnitude range. The layout with the highest fraction is almost
always the real one.

**Containers.** Member names, sizes, compressed sizes and compression method,
with `--recurse` profiling the members themselves.

### Field grammars and line shapes

Two views of the same lines, because they answer different questions.

A **field grammar** replaces each field with one letter for its *class* — `I`
integer, `F` float, `S` NaN/Inf/overflow fill, `N` name, `?` unparseable, `_`
empty — and collapses runs. This is the view that groups: `F{23}IFIF{10}II` is
the same record whether a value needed three digits or nine. It is the first
thing to read in a report.

A **line shape** replaces each *character*: every digit becomes `9`, every
capital `A`, every lowercase `a`. Punctuation, spaces and alignment survive,
because those are the grammar of a fixed-width format. Runs of three or more
collapse.

```
*TABLEAU : SIM_001            ->   *A{7} : A{3}_9{3}
   0.0000   1.2340E+02        ->      9.9{4}   9.9{4}A+99
```

Shapes are what fixed-width formats are made of, and they are useless for
free-format numbers — a real `.aoe` file gives 25 404 distinct shapes and 321
distinct grammars for the same 26 557 lines. The report carries both, each with
counts and a first-line number, and never the lines themselves. Shapes differing
only in an exponent's sign are also folded into one "family".

When a delimiter already explains the file, the inferred fixed-width columns are
flagged as incidental — a whitespace-delimited file will always appear to have
column boundaries, and splitting is simpler and safer than slicing.

### Redaction tiers

`--tier structure` is the default and is the safe one. Every name in the file —
column headers, HDF5 dataset paths, attribute names, archive member names, and
the profiled file's own name — is masked exactly as a data line is. The report is
a skeleton.

`--tier identifiers` adds the real names, and string attribute values, which are
typically units. Names are usually engineering vocabulary rather than data, and
they are what makes a reader writable rather than guessable — but they are the
one thing that could be argued to be disclosure, so they are opt-in, and the
text report prints them in a section headed `IDENTIFIERS (review before
sending)` so they can be read or deleted before the report goes anywhere.

There is no tier that emits values. Every number in a report is one of: a byte
count, an array dimension, an occurrence count, or a derived flag with no
magnitude. `selftest.py` asserts this by searching the produced report for
values it knows are in the source file.

A sensible sequence when in doubt: run at `structure`, read the report, and only
then decide whether the names need to come too.

---

## The fast path — infer the reader from the twin

`aoe_probe.py` describes a file; `aoe_infer.py` works out how to *read* it. The
difference is the twin: with ground truth on hand a parse is either right or
wrong, so the machine can search instead of asking.

```
python aoe_infer.py --aoe RUN.aoe --other RUN.DAT
python aoe_infer.py --aoe RUN.aoe --other RUN.h5   -o inferred.json
python aoe_infer.py --aoe RUN.aoe --other RUN.BIN  --no-constants
```

What it does in one run:

1. Profiles the unknown file structurally, as `aoe_probe.py` would.
2. Enumerates every plausible parse — each delimiter, fixed-width columns, each
   HDF5 layout convention, each binary word size, endianness, record layout and
   row/column order. HDF5 is a container rather than a layout, so the
   conventions are enumerated explicitly: the run identified by the parent
   group (`/RUN_1/ALTITUDE`), by a suffix of the dataset name
   (`/ALTITUDE/ALTITUDE_1_2_0` — a group per channel, a dataset per run), by an
   ancestor further up the tree, or not at all; plus one 2-D table per run with
   the channels along either axis, and a single 3-D block. Nothing but the twin
   can tell the first two apart.
3. Parses the file each way and matches the resulting columns against the
   twin's channels **by their values**. This is the part that removes the
   guesswork: a column is identified as `ALTITUDE` because it *is* `ALTITUDE` —
   no header needed, no agreement on spelling, and a unit difference is detected
   as an affine relationship and named (`scaled x0.3048 (feet -> metres)`)
   instead of being read as a wrong column. Channels that stay constant through
   a run are matched on the value they hold, since a flat line correlates with
   nothing.
4. Keeps the strategy that explains the most of the twin, writes a reader for it
   into `readers/generated_<name>_reader.py`, runs that reader, and grades it —
   so the last line of the report is a verdict on working code, not on a theory.

On a real file it looks like this. `sample/quick.aoe` was re-exported as
semicolon-delimited text with no header, its 38 columns shuffled into a random
order, and altitude converted to feet:

```
twin      6 runs, 38 channels
tried     1 parse strategies
  text:semicolon                     score 226.5

winner    text:semicolon
  delimiter        semicolon
  field_count      38
  header_present   False
  runs_found       6

runs
  1          -> twin 1          38 matched (37 identical)
  ...

column channel                            relation   agreement
0      P                                  identical  1e-9
18     ALTITUDE                           scaled x0.3048 (feet -> metres) 1e-9
27     CONFIGURATION                      identical  exact
...

reader    readers/generated_export_reader.py
graded    228 comparisons, 0 failing

verdict   EQUIVALENT
```

The same data exported as big-endian float64 Fortran unformatted records, and
again as a headerless little-endian float32 column-major dump, both come back
`EQUIVALENT` as well, with the word size, endianness and storage order recovered
from nothing but the twin.

When it cannot find a parse it says so rather than inventing one:

```
verdict   no strategy reproduced the twin
```

That is the honest outcome for a container it does not understand, a file that
is not the same simulation, or a convention outside the enumeration. Send
`aoe_probe.py`'s profile in that case and the enumeration can be extended.

### What the report carries

Structure, the column-to-channel map, agreement exponents, and the strategy that
won — no sample of anyone's data. The one exception is the gain and offset of a
detected unit conversion, which are constants of the format rather than values
from it; `--no-constants` omits them if that distinction matters where you are.
The structural profile inside the report is always masked; `--tier structure`
masks the matching half of it too — channel names, run labels and dataset
keys — for when the report has to leave the site.

---

## Step 2 — grade a reader against the `.aoe` twin

The one thing that makes this tractable is that the same simulation can be
exported twice: once in the ad-hoc format, once as `.aoe`. The `.aoe` file is
then ground truth, and a candidate reader can be graded against it on site.

```
python aoe_compare.py --aoe RUN.aoe --other RUN.DAT --reader csv
python aoe_compare.py --aoe RUN.aoe --other RUN.DAT \
    --reader readers.example_reader:read -o conformance.json
```

`--reader csv` is a built-in delimited-text reader: it sniffs the delimiter,
takes the last non-numeric line before the data as the header, and treats the
file as one run. It exists so the harness is runnable on day one. `--reader
module:function` uses a real reader instead.

A reader is one function. It takes a path and returns

```python
{run_name: {channel_name: array_of_floats}}
```

The `.aoe` side is read through `visimu.Session`, not through a second
parser, so what is being graded is agreement with what the application will
actually display.

Runs are paired by name, or by position with `--align-runs order`. Channels are
paired by a normalised name — `Total Force`, `TOTAL_FORCE` and `total-force` are
one channel — and anything the two exporters spell irreconcilably differently is
paired explicitly:

```json
{
  "channels": { "TIME": "TOTAL_TIME", "ALT": "ALTITUDE_MSL" },
  "runs":     { "SIM_001": "CASE1" }
}
```

```
python aoe_compare.py ... --map channel_map.json
```

The report says, per channel and run: whether both sides have it, whether the
arrays are the same length, how many elements disagree about being NaN, and one
scalar — the worst relative difference — with a flag for whether it falls inside
float32 epsilon. Exit status is 0 on agreement and 1 on disagreement, so this
can be wired straight into a test run.

Two switches control what the report may carry, for when the report is the thing
being sent rather than read locally:

```
--tier structure     mask the channel and run names, as aoe_probe.py does
--no-magnitudes      report the exponent of each difference (-8) instead of
                     the difference itself
```

Together they produce a report that says "these two files agree to within 1e-8
on all 38 channels of all 6 runs, and neither has a channel the other lacks"
while naming nothing and quantifying nothing.

### The harness, proven

Reading one real `.aoe` file both ways — through the core, and through
`readers/aoe_text_reader.py` — must come out equivalent, because both are
reading the same bytes:

```
python aoe_compare.py --aoe ../sample/quick.aoe --other ../sample/quick.aoe \
    --reader readers.aoe_text_reader:read
```

```
runs      6 in .aoe, 6 in the other file, 6 matched (name)
channels  228 comparisons over 38 distinct channels
verdict   EQUIVALENT (0 failing comparisons)
```

The residual agreement is about 6e-8 on every channel, and that number is worth
recognising: it is float32 round-off. The core stores samples as float32, the
text reader parses them as float64, so 6e-8 is the two representations of the
same number, not a bug in either. A real reader that comes back at this level is
correct; one that comes back at 1e-3 is reading a different column.

---

## Step 3 — what to send

With a twin: `inferred.json` and the generated reader. That is the whole
package — it already carries the parse, the column map and the proof that it
works, and there is nothing to iterate on.

Without a twin:

1. `profile.json` from `aoe_probe.py` for **one file of each distinct kind** the
   format produces — a small run and a large one if they differ, and a
   "difficult" one (missing channels, a failed case, a restart) if such a thing
   exists. Structure tier is enough to start; the identifier tier saves a
   round-trip.
2. The number of runs, channels and samples the real files hold, roughly, and
   the largest file size that has to open. This sizes the reader; it does not
   describe anyone's data.
3. Whatever the format's own documentation says, if any exists and may be
   shared. A one-page spec beats a hundred profiles.
4. Later, `conformance.json` from `aoe_compare.py`, which is what turns "this
   reader parses" into "this reader is correct".

## How this fits the application

The reader protocol here — `{run: {channel: array}}` — is the same shape a
push-based entry point into the C++ core would want: one contiguous array per
channel, named, grouped by run. A reader that satisfies this harness in Python
is therefore most of the way to being wired in, whether the final version stays
in Python and pushes arrays down, or is rewritten natively in the core behind
the same four `Dataset` methods the `.aoe` reader uses.

Nothing in this folder writes to the application, the build, or `sample/`.
Reports and `__pycache__` are ignored by the local `.gitignore`; the scripts
themselves are worth committing.
