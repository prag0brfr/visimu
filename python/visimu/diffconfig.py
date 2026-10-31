# Author: Paulo R. A. Gilz
"""The threshold configuration for a file-to-file comparison.

The file is YAML, and this reads it without PyYAML. That is a deliberate
choice rather than a shortcut: the machines this runs on reach the package
index through a proxy that has to be negotiated one wheel at a time, and the
shape being read here is a narrow one --

    - group_name: "BLACK"
      algoconfs:
        - variables: "ALFA"
          threshold: 25
        - variables: "Q1;TETA;BETA"
          x: "TIME"
          threshold: 10
    - group_name: "AMBER"
      ...

a list of groups, each with a list of entries, each entry three scalars. Keys
nobody here knows are skipped, at any level and to any depth, which is what
makes the real files -- full of fields this program has no opinion about --
readable at all.

What this is not: a YAML parser. Anchors, flow collections, multi-line
scalars, nested lists of lists have no meaning in this shape and are refused
rather than half-understood. A file that needs them is not this file.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# The x axis an entry is compared along when it does not say.
DEFAULT_X = "TIME"


class ConfigError(ValueError):
    """The file is not a threshold configuration.

    Carries the line number wherever there is one: "threshold must be a
    number" is a different message from "threshold must be a number, line 24",
    and only one of them can be acted on.
    """


class Rule:
    """One threshold: a group, a parameter, the x axis, and the value.

    One per (group, parameter) pair -- the file writes several parameters on
    one line, and each gets its own rule, because that is how they are used.
    """

    __slots__ = ("group", "param", "x_param", "threshold")

    def __init__(self, group: str, param: str, x_param: str, threshold: float):
        self.group = group
        self.param = param
        self.x_param = x_param or DEFAULT_X
        self.threshold = float(threshold)

    @property
    def key(self) -> Tuple[str, str]:
        return (self.group, self.param)

    def to_dict(self) -> dict:
        return {"group": self.group, "param": self.param,
                "x_param": self.x_param, "threshold": self.threshold}

    @staticmethod
    def from_dict(data: dict) -> "Rule":
        return Rule(str(data.get("group", "")), str(data.get("param", "")),
                    str(data.get("x_param", "") or DEFAULT_X),
                    float(data.get("threshold", 0.0)))

    def __repr__(self) -> str:        # pragma: no cover - debugging only
        return "<Rule {0}/{1} > {2}>".format(self.group, self.param,
                                             self.threshold)


# --------------------------------------------------------------- scanning


def _strip_comment(line: str) -> str:
    """Drop a trailing comment, respecting quotes.

    A '#' inside a quoted scalar is part of the name: these files carry
    channel names written by other programs, and one of them containing a
    hash is not a reason to lose the rest of the line.
    """
    out = []
    quote = ""
    for character in line:
        if quote:
            out.append(character)
            if character == quote:
                quote = ""
            continue
        if character in "\"'":
            quote = character
            out.append(character)
            continue
        if character == "#":
            break
        out.append(character)
    return "".join(out)


def _unquote(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def _number(text: str, line_number: int, what: str) -> float:
    try:
        return float(_unquote(text))
    except (TypeError, ValueError):
        raise ConfigError(
            "{0} must be a number, line {1}: {2!r}".format(
                what, line_number, text.strip()))


class _Line:
    """A line that matters: its indent, its key, its value, its number."""

    __slots__ = ("indent", "key", "value", "is_item", "number")

    def __init__(self, indent: int, key: str, value: str, is_item: bool,
                 number: int):
        self.indent = indent
        self.key = key
        self.value = value
        self.is_item = is_item
        self.number = number


def _scan(text: str) -> List[_Line]:
    """Every meaningful line, as (indent, key, value, list-item?).

    Blank lines, comments and the bare "..." the real files are littered with
    are dropped here, so nothing downstream has to know about them.
    """
    scanned = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw.replace("\t", "    "))
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped in ("...", "---"):
            # A document marker, or the elision the specification writes for
            # "fields this program ignores". Neither is data.
            continue

        indent = len(line) - len(line.lstrip(" "))
        is_item = stripped.startswith("- ")
        if is_item or stripped == "-":
            stripped = stripped[2:].strip() if is_item else ""
            # A list item's own indent is where its content starts, so that
            # "- key: value" sits at the same level as a following "key:".
            indent += 2
            is_item = True

        if not stripped:
            scanned.append(_Line(indent, "", "", True, number))
            continue

        if ":" not in stripped:
            raise ConfigError(
                "expected 'key: value', line {0}: {1!r}".format(
                    number, raw.strip()))
        key, _sep, value = stripped.partition(":")
        scanned.append(_Line(indent, _unquote(key.strip()), value.strip(),
                             is_item, number))
    return scanned


# ---------------------------------------------------------------- parsing


def parse_rules(text: str) -> List[Rule]:
    """The rules a configuration defines, in file order.

    Later definitions of the same (group, parameter) replace earlier ones, as
    the specification asks: the file is edited by hand and a duplicate is a
    correction, not an error.
    """
    lines = _scan(text)
    if not lines:
        raise ConfigError("the configuration is empty")

    groups = [line for line in lines if line.key == "group_name"]
    if not groups:
        raise ConfigError(
            "no group_name anywhere: this is not a threshold configuration")

    rules: Dict[Tuple[str, str], Rule] = {}
    order: List[Tuple[str, str]] = []

    group = ""
    group_indent = -1
    in_algoconfs = False
    algoconfs_indent = -1
    entry: Optional[dict] = None
    entry_indent = -1

    def close_entry():
        """Turn the entry just read into one rule per variable."""
        if entry is None:
            return
        names = [name.strip() for name in str(entry.get("variables", "")).split(";")]
        names = [name for name in names if name]
        if not names:
            raise ConfigError(
                "an algoconf entry names no variables, line {0}".format(
                    entry["line"]))
        if entry.get("threshold") is None:
            raise ConfigError(
                "{0} has no threshold, line {1}".format(
                    ", ".join(names), entry["line"]))
        for name in names:
            rule = Rule(group, name, entry.get("x") or DEFAULT_X,
                        entry["threshold"])
            if rule.key not in rules:
                order.append(rule.key)
            rules[rule.key] = rule

    for line in lines:
        if line.key == "group_name":
            close_entry()
            entry = None
            in_algoconfs = False
            group = _unquote(line.value)
            group_indent = line.indent
            if not group:
                raise ConfigError(
                    "a group has no name, line {0}".format(line.number))
            continue

        if line.indent < group_indent and line.key != "group_name":
            # Back out above the group's own level: whatever this is, it is
            # not part of the group. At the group's own level it is one of the
            # group's fields -- a list item's keys share an indent with the
            # item's first key, so "algoconfs" sits level with "group_name"
            # rather than under it, and only something further left has left
            # the group.
            close_entry()
            entry = None
            in_algoconfs = False
            continue

        if line.key == "algoconfs":
            close_entry()
            entry = None
            in_algoconfs = True
            algoconfs_indent = line.indent
            continue

        if not in_algoconfs:
            continue            # a field of the group this program ignores

        if line.indent <= algoconfs_indent:
            close_entry()
            entry = None
            in_algoconfs = False
            continue

        if line.is_item:
            close_entry()
            entry = {"line": line.number}
            entry_indent = line.indent
        elif entry is None:
            continue            # inside algoconfs but before the first entry
        elif line.indent < entry_indent:
            close_entry()
            entry = None
            continue

        if not line.key:
            continue            # a bare "-", whose fields follow

        if line.key == "variables":
            entry["variables"] = _unquote(line.value)
        elif line.key == "x":
            entry["x"] = _unquote(line.value)
        elif line.key == "threshold":
            entry["threshold"] = _number(line.value, line.number, "threshold")
        # Everything else inside an entry is ignored, by design.

    close_entry()

    if not rules:
        raise ConfigError("no thresholds found: every group is empty")
    return [rules[key] for key in order]


def read_rules(path: str) -> List[Rule]:
    """Parse a configuration file. The extension is not looked at."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return parse_rules(handle.read())


# ------------------------------------------------------------- priorities


def group_names(rules: List[Rule]) -> List[str]:
    """The groups a rule set mentions, in the order they first appear."""
    seen = []
    for rule in rules:
        if rule.group not in seen:
            seen.append(rule.group)
    return seen


def ties(rules: List[Rule]) -> List[Tuple[str, List[str]]]:
    """(parameter, groups) wherever two groups share a threshold.

    Priority is the numeric threshold and nothing else, so two groups equal on
    one parameter have no order between them. Reported rather than resolved:
    picking one silently would put a colour on the plot that the file does not
    justify.
    """
    by_param: Dict[str, Dict[float, List[str]]] = {}
    for rule in rules:
        by_param.setdefault(rule.param, {}).setdefault(
            rule.threshold, []).append(rule.group)
    clashes = []
    for param in sorted(by_param):
        for threshold in sorted(by_param[param]):
            groups = by_param[param][threshold]
            if len(groups) > 1:
                clashes.append((param, sorted(groups)))
    return clashes
