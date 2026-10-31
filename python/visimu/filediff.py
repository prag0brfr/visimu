# Author: Paulo R. A. Gilz
"""Comparing two files run by run: what differs, by how much, and where.

A non-regression test asks one question of two result files -- did anything
move by more than it is allowed to -- and the answer has to be a list of
places to look rather than a number. So this module produces flags: (run,
parameter, group) triples, each meaning "the threshold for that group was
exceeded at least once here", plus the x intervals to shade on a plot.

Three decisions are worth stating, because each of them could reasonably have
gone the other way and the plots would then lie:

Runs are matched by identifier, never by position. Two files from two builds
of a model hold the same runs in whatever order the solver finished them, and
comparing run 3 of one with run 3 of the other is how a non-regression test
reports differences that are not there.

Samples outside the overlapping x range are not compared at all. Where the
two files sample a run at different instants, one series is read at the
other's instants by interpolation -- but only inside the range they share.
Beyond it there is nothing to interpolate from, and an extrapolated
difference is a difference nobody measured.

Priority is the numeric threshold, and nothing else: the group with the
*larger* threshold wins, because exceeding a larger allowance is the worse
news. Group names carry no meaning here -- they are labels the user colours.

This module knows nothing about Qt, and nothing about the session beyond the
two small callables it is handed. That is what makes the arithmetic testable
on its own, which for a tool whose whole output is "these runs changed" is
the part that has to be right.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .diffconfig import DEFAULT_X, Rule

# A reader: (file_index, run_index, name) -> samples, as Session.channel is.
Reader = Callable[[int, int, str], np.ndarray]


class DiffError(ValueError):
    """The two files cannot be compared, with the reason the user needs."""


class Violation:
    """One group exceeded on one (run, parameter), and where.

    `regions` are (low, high) in x, already merged and already resolved
    against the other groups of the same parameter: what is in here is what
    gets shaded, so the priority rule is applied once, here, rather than by
    whatever happens to draw it.
    """

    __slots__ = ("group", "threshold", "regions")

    def __init__(self, group: str, threshold: float,
                 regions: Sequence[Tuple[float, float]]):
        self.group = group
        self.threshold = float(threshold)
        self.regions = [(float(low), float(high)) for low, high in regions]

    def to_dict(self) -> dict:
        return {"group": self.group, "threshold": self.threshold,
                "regions": [list(region) for region in self.regions]}

    @staticmethod
    def from_dict(data: dict) -> "Violation":
        return Violation(str(data.get("group", "")),
                         float(data.get("threshold", 0.0)),
                         [tuple(region[:2]) for region in
                          data.get("regions", []) if len(region) >= 2])


class Flag:
    """Everything found for one (run, parameter), worst group first."""

    __slots__ = ("run_id", "param", "x_param", "violations")

    def __init__(self, run_id: str, param: str, x_param: str,
                 violations: Sequence[Violation]):
        self.run_id = run_id
        self.param = param
        self.x_param = x_param or DEFAULT_X
        # Highest threshold first: that is the priority order, and it is the
        # order the report reads best in.
        self.violations = sorted(violations, key=lambda v: -v.threshold)

    @property
    def groups(self) -> List[str]:
        return [violation.group for violation in self.violations]

    def to_dict(self) -> dict:
        return {"run_id": self.run_id, "param": self.param,
                "x_param": self.x_param,
                "violations": [v.to_dict() for v in self.violations]}

    @staticmethod
    def from_dict(data: dict) -> "Flag":
        return Flag(str(data.get("run_id", "")), str(data.get("param", "")),
                    str(data.get("x_param", "") or DEFAULT_X),
                    [Violation.from_dict(entry)
                     for entry in data.get("violations", [])])


class Report:
    """The flags of one comparison, and what to say about it."""

    def __init__(self, file_a: int, file_b: int, name_a: str, name_b: str,
                 flags: Sequence[Flag], warnings: Sequence[str] = ()):
        self.file_a = int(file_a)
        self.file_b = int(file_b)
        self.name_a = name_a
        self.name_b = name_b
        self.flags = list(flags)
        self.warnings = list(warnings)

    # Runs, in the order the files hold them, each with its own flags. The
    # report shows one line per run, and a run with nothing to say is not a
    # line.
    def by_run(self) -> List[Tuple[str, List[Flag]]]:
        order: List[str] = []
        grouped: Dict[str, List[Flag]] = {}
        for flag in self.flags:
            if flag.run_id not in grouped:
                order.append(flag.run_id)
                grouped[flag.run_id] = []
            grouped[flag.run_id].append(flag)
        return [(run_id, grouped[run_id]) for run_id in order]

    def line_for(self, run_id: str) -> str:
        """RUN2: ALFA(BLACK), TETA(BLACK), TETA(RED) -- as plain text."""
        for candidate, flags in self.by_run():
            if candidate != run_id:
                continue
            parts = []
            for flag in flags:
                for group in flag.groups:
                    parts.append("{0}({1})".format(flag.param, group))
            return "{0}: {1}".format(run_id, ", ".join(parts))
        return run_id

    def to_dict(self) -> dict:
        return {"file_a": self.file_a, "file_b": self.file_b,
                "name_a": self.name_a, "name_b": self.name_b,
                "flags": [flag.to_dict() for flag in self.flags],
                "warnings": list(self.warnings)}

    @staticmethod
    def from_dict(data: dict) -> "Report":
        return Report(int(data.get("file_a", -1)), int(data.get("file_b", -1)),
                      str(data.get("name_a", "")), str(data.get("name_b", "")),
                      [Flag.from_dict(entry) for entry in data.get("flags", [])],
                      [str(note) for note in data.get("warnings", [])])


# ------------------------------------------------------------- validation


def check_compatible(runs_a: Sequence[str], runs_b: Sequence[str],
                     params_a: Sequence[str], params_b: Sequence[str]) -> None:
    """Raise DiffError unless the two files can be compared at all.

    Counted, then matched, then named: three messages rather than one, because
    "the files have 24 and 25 runs" and "these three run names are not in the
    other file" send the user to different places.
    """
    if len(runs_a) != len(runs_b):
        raise DiffError(
            "The files hold different numbers of runs: {0} and {1}."
            .format(len(runs_a), len(runs_b)))

    only_a = [run for run in runs_a if run not in set(runs_b)]
    only_b = [run for run in runs_b if run not in set(runs_a)]
    if only_a or only_b:
        parts = []
        if only_a:
            parts.append("only in the first: " + ", ".join(sorted(only_a)[:10]))
        if only_b:
            parts.append("only in the second: " + ", ".join(sorted(only_b)[:10]))
        raise DiffError(
            "The run identifiers do not match one to one.\n" + "\n".join(parts))

    seen = {}
    for run in runs_a:
        seen[run] = seen.get(run, 0) + 1
    repeated = sorted(run for run, count in seen.items() if count > 1)
    if repeated:
        raise DiffError(
            "These run identifiers appear more than once, so there is no "
            "one-to-one match: " + ", ".join(repeated[:10]))

    missing_b = [name for name in params_a if name not in set(params_b)]
    missing_a = [name for name in params_b if name not in set(params_a)]
    if missing_a or missing_b:
        parts = []
        if missing_b:
            parts.append("not in the second file: "
                         + ", ".join(sorted(missing_b)[:10]))
        if missing_a:
            parts.append("not in the first file: "
                         + ", ".join(sorted(missing_a)[:10]))
        raise DiffError(
            "The files carry different parameters.\n" + "\n".join(parts))


# ---------------------------------------------------------------- compare


def aligned(x_a, y_a, x_b, y_b):
    """(x, a, b) on the first file's instants, over the shared x range.

    The first file is the reference, as it is in the run-to-run diff: its
    instants are where the difference is reported, because those are instants
    something was actually recorded at. The second series is read at them by
    linear interpolation, and only inside the range the two share -- outside
    it nothing is compared, which is the specification's rule and the honest
    one.

    Returns None when there is no shared range to speak of.
    """
    x_a = np.asarray(x_a, dtype=np.float64)
    y_a = np.asarray(y_a, dtype=np.float64)
    x_b = np.asarray(x_b, dtype=np.float64)
    y_b = np.asarray(y_b, dtype=np.float64)

    length_a = min(x_a.size, y_a.size)
    length_b = min(x_b.size, y_b.size)
    if length_a < 1 or length_b < 1:
        return None
    x_a, y_a = x_a[:length_a], y_a[:length_a]
    x_b, y_b = x_b[:length_b], y_b[:length_b]

    good_b = np.isfinite(x_b) & np.isfinite(y_b)
    if good_b.sum() < 1:
        return None
    x_b, y_b = x_b[good_b], y_b[good_b]
    if not np.all(np.diff(x_b) >= 0.0):
        order = np.argsort(x_b, kind="stable")
        x_b, y_b = x_b[order], y_b[order]

    good_a = np.isfinite(x_a)
    if good_a.sum() < 1:
        return None

    low = max(float(np.min(x_a[good_a])), float(x_b[0]))
    high = min(float(np.max(x_a[good_a])), float(x_b[-1]))
    if not (high >= low):
        return None

    inside = good_a & (x_a >= low) & (x_a <= high)
    if inside.sum() < 1:
        return None

    x = x_a[inside]
    return x, y_a[inside], np.interp(x, x_b, y_b)


def _regions(x: np.ndarray, owner: Sequence[Optional[str]],
             group: str) -> List[Tuple[float, float]]:
    """Where `group` owns the shading, as merged (low, high) in x.

    Two rules about the far end, and the difference between them is visible on
    every plot:

    Where the next sample belongs to *another* group, the region runs up to
    that sample, so the two bands abut and no sliver of plot shows through
    between them. That is the specification's "RED on [2, 3), BLACK on
    [3, 5]": RED ends exactly where BLACK takes over.

    Where the next sample is violating nothing, the region ends at the last
    sample that *was* violating -- so BLACK on [3, 5] is [3, 5] and not
    [3, 6]. Running on to the next sample would shade an interval in which
    the difference was within tolerance.

    A lone sample gives a zero-width region rather than none: a violation at
    one instant is still one, and the drawing gives it a minimum width.
    """
    count = len(owner)
    if count == 0:
        return []
    found: List[Tuple[float, float]] = []
    start = None
    for index in range(count):
        mine = owner[index] == group
        if mine and start is None:
            start = index
        elif not mine and start is not None:
            # Up to this sample when another group owns it, back to the last
            # of ours when nobody does.
            end = index if owner[index] is not None else index - 1
            found.append((float(x[start]), float(x[end])))
            start = None
    if start is not None:
        found.append((float(x[start]), float(x[count - 1])))
    return found


def _worst_group(order: Sequence[Tuple[float, str, np.ndarray]],
                 count: int) -> List[Optional[str]]:
    """Per sample, the violated group with the highest threshold, or None.

    `order` is (threshold, group, mask), and it is walked from the highest
    threshold down: the first group that is violated at a sample owns that
    sample, and nothing lower is drawn there. That is the whole of the
    priority rule, and doing it here means no drawing code can get it wrong.
    """
    owner: List[Optional[str]] = [None] * count
    for _threshold, group, mask in order:
        for index in range(count):
            if owner[index] is None and mask[index]:
                owner[index] = group
    return owner


def compare_run(reader: Reader, file_a: int, file_b: int, run_a: int,
                run_b: int, param: str, rules: Sequence[Rule],
                x_param: str = DEFAULT_X) -> Tuple[Optional[Flag], List[str]]:
    """Compare one parameter of one matched run. (flag or None, warnings)."""
    warnings: List[str] = []
    try:
        x_one = reader(file_a, run_a, x_param)
        x_two = reader(file_b, run_b, x_param)
        y_one = reader(file_a, run_a, param)
        y_two = reader(file_b, run_b, param)
    except Exception as exc:                    # a read that cannot be done
        return None, ["{0}: {1}".format(param, exc)]

    for name, data in ((x_param, x_one), (x_param, x_two),
                       (param, y_one), (param, y_two)):
        if data is None or getattr(data, "size", 0) == 0:
            return None, ["{0} holds no samples for {1}".format(name, param)]

    pair = aligned(x_one, y_one, x_two, y_two)
    if pair is None:
        return None, [
            "{0}: the two files share no {1} range, so nothing was compared."
            .format(param, x_param)]

    x, a, b = pair
    difference = np.abs(a - b)
    # A sample only one file has a number for is not a difference anyone
    # measured.
    comparable = np.isfinite(difference)

    masks = []
    for rule in rules:
        mask = comparable & (difference > rule.threshold)
        if bool(mask.any()):
            masks.append((rule.threshold, rule.group, mask))
    if not masks:
        return None, warnings

    masks.sort(key=lambda entry: -entry[0])
    owner = _worst_group(masks, x.size)

    violations = []
    for threshold, group, _mask in masks:
        violations.append(Violation(group, threshold,
                                    _regions(x, owner, group)))

    return Flag(str(run_a), param, x_param, violations), warnings


def compare_files(reader: Reader, file_a: int, file_b: int,
                  runs_a: Sequence[Tuple[int, str]],
                  runs_b: Sequence[Tuple[int, str]],
                  rules: Sequence[Rule],
                  name_a: str = "", name_b: str = "") -> Report:
    """Compare two whole files against a rule set.

    `runs_a` and `runs_b` are (run index, identifier) in each file. The
    identifiers are what the runs are paired by; the indices are how the
    reader is addressed.

    Every parameter named by a rule is compared, and only those: a rule set is
    a statement about which channels matter, and comparing the other four
    hundred would turn the report into a list of everything that moved.
    """
    ids_a = [identifier for _index, identifier in runs_a]
    ids_b = [identifier for _index, identifier in runs_b]
    params = []
    for rule in rules:
        if rule.param not in params:
            params.append(rule.param)
    check_compatible(ids_a, ids_b, params, params)

    where_b = {identifier: index for index, identifier in runs_b}
    by_param: Dict[str, List[Rule]] = {}
    x_of: Dict[str, str] = {}
    for rule in rules:
        by_param.setdefault(rule.param, []).append(rule)
        # One x per parameter: the last rule to name one wins, which is the
        # same "last definition wins" the configuration itself uses.
        x_of[rule.param] = rule.x_param

    flags: List[Flag] = []
    warnings: List[str] = []
    seen_warnings = set()

    def note(message: str) -> None:
        if message not in seen_warnings:
            seen_warnings.add(message)
            warnings.append(message)

    for index_a, identifier in runs_a:
        index_b = where_b.get(identifier)
        if index_b is None:                     # check_compatible ruled it out
            continue
        for param in params:
            flag, complaints = compare_run(
                reader, file_a, file_b, index_a, index_b, param,
                by_param[param], x_of.get(param, DEFAULT_X))
            for complaint in complaints:
                note("{0}: {1}".format(identifier, complaint))
            if flag is not None:
                # The identifier, not the index: it is what the report shows
                # and what the pages are named after.
                flag.run_id = identifier
                flags.append(flag)

    return Report(file_a, file_b, name_a, name_b, flags, warnings)
