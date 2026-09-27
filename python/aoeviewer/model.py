# Author: Paulo R. A. Gilz
"""Application state: pages, plot tiles, selection and styling.

Deliberately free of Qt imports. Keeping the state model separate from the
widgets is what lets the workspace be saved, restored and tested without a
running UI. State that lives next to the draw calls cannot be exercised on its
own, which puts the parts most worth testing out of reach of every test.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Set, Tuple

from . import isogroups

MAX_TILES = 25  # a 5x5 grid

# Half-extent of the default 3D view. The normalised cube's half-diagonal is
# sqrt(3)/2 = 0.866, so 0.95 shows the whole box at any rotation with a small
# margin -- which is what "fit" has to return to.
DEFAULT_3D_HALF = 0.95

# How far the projection may be scaled, as a half-extent. The first ceiling was
# 20 -- barely 21x out from the default, reached in a couple of seconds of
# scrolling. There is no reason to stop someone pulling right back from a long
# trajectory; the floor is what stops a fast scroll turning the plot into the
# inside of a single line.
MIN_3D_HALF = 1e-3
MAX_3D_HALF = 5.0e3


def clamp_3d_half(value: float) -> float:
    return max(MIN_3D_HALF, min(MAX_3D_HALF, float(value)))


class ColorMode(IntEnum):
    """What decides a curve's colour."""

    SIMULATION = 0
    PARAMETER = 1
    ISO = 2


class StyleMode(IntEnum):
    """What decides a curve's dash pattern and marker.

    Independent of Colour by. Colour names one identity (usually the
    quantity); style names another (usually the file or the ISO value).
    NONE keeps the previous behaviour: dash and marker follow the colour key.
    """

    NONE = 0
    FILE = 1
    SIMULATION = 2
    ISO = 3


STYLE_MODE_LABELS = {
    StyleMode.NONE: "None",
    StyleMode.FILE: "File",
    StyleMode.SIMULATION: "Simulation",
    StyleMode.ISO: "ISO",
}


class MouseMode(IntEnum):
    """What the mouse does over a plot.

    Exactly one is active at a time and the others are genuinely off, not just
    unbound: in SELECT and CROSSHAIR the view box has mouse interaction disabled
    entirely, so a stray drag cannot pan the plot out from under you.
    """

    SELECT = 0      # click to make a tile active; no view manipulation
    CROSSHAIR = 1   # click a curve to inspect that sample
    PAN = 2         # drag to move; wheel does nothing
    ZOOM = 3        # wheel zooms, drag selects a rectangle to zoom into


MOUSE_MODE_LABELS = {
    MouseMode.SELECT: "Select",
    MouseMode.CROSSHAIR: "Inspect",
    MouseMode.PAN: "Pan",
    MouseMode.ZOOM: "Zoom",
}

MOUSE_MODE_HINTS = {
    MouseMode.SELECT: "Click a tile to make it active. The view does not move.",
    MouseMode.CROSSHAIR: "Click a curve to read every channel of that run at that point.",
    MouseMode.PAN: "Drag to move the view; the wheel is disabled. On a 3D tile: "
                   "drag turns it, middle-drag or shift-drag slides it, "
                   "right-drag or the wheel zooms.",
    MouseMode.ZOOM: "Wheel zooms; drag a rectangle to zoom into it.",
}


# A qualitative palette that stays distinguishable on both dark and light
# backgrounds.
#
# The first three entries are fixed by convention rather than chosen for
# separation: red, blue and dark green, in that order, whatever the colouring
# mode. Most plots carry three or four curves, and a report that pairs the
# same three colours with the same three meanings every time is easier to read
# across a set of figures than one whose first colour depends on how many
# curves happened to be on the tile. Dark green rather than the mid green that
# used to sit third, and no cyan anywhere near the front, because cyan and the
# lighter green read as the same colour once a line is one pixel wide.
#
# Everything after the third entry is ordered for separation as before.
PALETTE: Tuple[str, ...] = (
    "#e8232a",  # red
    "#1f6fe0",  # blue
    "#1a7a3c",  # dark green
    "#ffb300",  # amber
    "#e040fb",  # magenta
    "#8c6cff",  # violet
    "#ff8a3d",  # orange
    "#5aa9e6",  # light blue
    "#9ccc65",  # light green
    "#ff80ab",  # pink
    "#00a39a",  # teal
    "#a1631a",  # brown
)

DASH_STYLES: Tuple[str, ...] = ("solid", "dash", "dot", "dashdot")

# Point markers. The empty string is "no marker", which is the common case for
# a continuous trace.
SYMBOLS: Tuple[str, ...] = ("", "o", "s", "t", "t1", "d", "+", "x", "star", "p", "h")

SYMBOL_LABELS = {
    "": "none", "o": "circle", "s": "square", "t": "triangle down",
    "t1": "triangle up", "d": "diamond", "+": "plus", "x": "cross",
    "star": "star", "p": "pentagon", "h": "hexagon",
}


@dataclass(frozen=True)
class CurveKey:
    """Identifies one drawn curve: a parameter of a simulation of a file."""

    file_index: int
    sim_index: int
    param: str

    def as_tuple(self) -> Tuple[int, int, str]:
        return (self.file_index, self.sim_index, self.param)


@dataclass
class Marker:
    """A point in a run, shown differently depending on the tile's x axis.

    A marker is defined against one parameter -- "TIME = 5.0". On a tile plotted
    against that same parameter it is a vertical line. On any *other* tile it is
    the sample where that condition holds, drawn as a dot: set TIME = 5 and you
    see where 5 seconds falls on ALPHA = f(NZ), which is exactly the question
    people ask when they put a cursor on a time history.
    """

    x_param: str = "TIME"
    position: float = 0.0
    visible: bool = True
    color: str = "#ff3333"
    # Size of the dot drawn on tiles whose x axis is a different parameter.
    dot_size: float = 9.0

    # No caption field. Markers and bands carried one, drawn on the plot, and it
    # was the wrong place for it twice over: pyqtgraph anchors an InfiniteLine's
    # label to the line, so it sat on top of the data with nothing behind it,
    # and every keystroke while editing it rebuilt the tile and deleted the item
    # being edited. Text notes already do captions properly -- placed where you
    # want them, on an opaque ground, dragged rather than typed into a box -- so
    # a band that needs naming gets a note beside it. Workspaces written when
    # the field existed still load: the reader keeps only the fields this class
    # declares, so the old key is ignored rather than refused.

    def describe(self) -> str:
        return "{0} = {1:g}".format(self.x_param or "?", self.position)


@dataclass
class AreaBand:
    """A shaded x band drawn on every tile sharing the same x parameter."""

    x_param: str = "TIME"
    x_min: float = 0.0
    x_max: float = 1.0
    color: str = "#3380ff"
    alpha: float = 0.20
    visible: bool = True

    # No caption field -- see Marker above for why. Name a band with a text note
    # placed over it.

    def describe(self) -> str:
        return "{0} in [{1:g}, {2:g}]".format(self.x_param or "?",
                                              self.x_min, self.x_max)


# Where a text note sits, as a fraction of the tile. View coordinates rather
# than data coordinates on purpose: a caption is furniture, and furniture that
# slides off screen the moment you zoom is not useful. Data-anchored notes are
# what markers are for.
@dataclass
class TextNote:
    """A caption placed on a tile."""

    text: str = "Note"
    x: float = 0.05          # 0 = left edge, 1 = right edge
    y: float = 0.05          # 0 = top edge, 1 = bottom edge
    size: float = 10.0       # points
    color: str = "#e6e9ec"
    bold: bool = False
    visible: bool = True

    def to_dict(self) -> dict:
        return {
            "text": self.text, "x": self.x, "y": self.y, "size": self.size,
            "color": self.color, "bold": self.bold, "visible": self.visible,
        }

    @staticmethod
    def from_dict(data: dict) -> "TextNote":
        note = TextNote()
        for key, value in data.items():
            if hasattr(note, key):
                setattr(note, key, value)
        return note

    def describe(self) -> str:
        first = (self.text or "").splitlines()[0] if self.text else ""
        return first[:40] or "(empty)"


@dataclass
class TileState:
    """One plot in the grid."""

    # Everything that decides what this tile's legend looks like: whether it is
    # shown at all, where it sits, and which parts make up a row. Named in one
    # place so that "apply this legend to every tile" cannot quietly fall behind
    # a field added later -- TestLegendFields guards exactly that, by requiring
    # every legend-ish field of this class to appear here.
    #
    # No annotation, deliberately: @dataclass collects annotated names only, so
    # this stays a class attribute rather than becoming a field with a default.
    LEGEND_FIELDS = (
        "show_legend",
        "legend_position",
        "legend_x",
        "legend_y",
        "legend_show_param",
        "legend_show_iso",
        "legend_show_file",
        "legend_show_sim",
        "legend_show_style",
    )

    x_param: str = ""
    y_params: List[str] = field(default_factory=list)

    show_legend: bool = True
    # One of the four corners, or "custom" -- set automatically when the legend
    # is dragged somewhere that is not a corner. The custom position is a
    # fraction of the tile so it survives a resize.
    legend_position: str = "top-left"
    legend_x: float = 0.02
    legend_y: float = 0.02

    # Which parts make up a legend entry.
    #
    # The parameter name alone by default. Every other part is off because with
    # them on, a study of fifty runs produced fifty legend rows that differed
    # only in a run id, filling the tile and hiding the plot -- and the run is
    # already identifiable by colour. Turning file and simulation off is also
    # what collapses an ISO plot down to one row per ISO value, since identical
    # labels are then deduplicated. Tick what you need back on in STYLE.
    legend_show_param: bool = True
    legend_show_iso: bool = False
    legend_show_file: bool = False
    legend_show_sim: bool = False

    # Retained so workspaces written before it was withdrawn still load. It was
    # offered as "Line style marker" and never read by anything that draws: the
    # legend shows a sample of the curve itself, dash pattern and marker
    # included, so spelling those out in the text said nothing the row was not
    # already showing.
    legend_show_style: bool = False

    # Fixed axis ranges. When locked the tile stops auto-fitting, which is what
    # you want while comparing several plots on identical scales.
    lock_x: bool = False
    lock_y: bool = False
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0

    # False until the tile has been framed once. The view is the user's, not the
    # program's: a rebuild happens for all sorts of reasons that have nothing to
    # do with what is on screen -- restyling a curve, turning the grid on,
    # dragging a marker, coming back to a page -- and re-fitting on each of them
    # threw away whatever the user had zoomed in on. So the tile is fitted when
    # it first acquires data, and after that only when asked: Fit, double-click,
    # or a change to the SCALE limits.
    view_initialised: bool = False

    log_y: bool = False
    show_grid: bool = True
    show_points: bool = False

    # Parameter, not simulation. A tile is nearly always asked "which
    # quantity is this line", and colouring by run answers a different
    # question -- one that a study of twenty-five runs answers with
    # twenty-five colours that name nothing the user chose. Colour by run is
    # one click away for the case where the runs are the subject.
    color_mode: ColorMode = ColorMode.PARAMETER
    color_iso_param: str = ""
    style_mode: StyleMode = StyleMode.NONE
    style_iso_param: str = ""

    # --- 3D -------------------------------------------------------------
    # A 3D tile plots z against x and y as a projected wireframe path. It is a
    # projection drawn with the ordinary 2D machinery rather than an OpenGL
    # scene: pyqtgraph's GL view needs PyOpenGL, which is one more wheel to get
    # past a blocked package index, and for trajectories -- which is what these
    # files hold -- a rotatable projection reads just as well.
    plot3d: bool = False
    z_param: str = ""
    azimuth: float = 45.0     # degrees, rotation about the vertical axis
    elevation: float = 25.0   # degrees, above the horizontal plane
    show_3d_axes: bool = True   # the corner orientation triad

    # Grid lines and tick values on the three back walls of the data box. On by
    # default because an orthographic projection with no reference surface
    # cannot be read: there is no perspective cue, so depth is guesswork and
    # the tile carries no numeric scale at all. It is a checkbox rather than a
    # rule because a dense trajectory is sometimes clearer bare.
    show_3d_grid: bool = True

    # The visible window onto the projection: centre and half-extent, in the
    # normalised cube's own coordinates. Kept separately from x_min/x_max
    # because those describe *data* ranges of a 2D tile and mean nothing here --
    # and because a 3D tile is rebuilt on every rotation step, so without
    # somewhere to keep it, any zoom the user applied would be thrown away the
    # moment they turned the object.
    view3d_cx: float = 0.0
    view3d_cy: float = 0.0
    view3d_half: float = DEFAULT_3D_HALF

    # Captions belong to the tile, not to a parameter: unlike a marker, a note
    # is about this particular plot, and repeating it across a 5x5 grid because
    # it happened to be "global" would be noise.
    texts: List["TextNote"] = field(default_factory=list)

    def reset_3d_view(self) -> None:
        """Frame the whole cube again. This is what auto-fit does in 3D."""
        self.view3d_cx = 0.0
        self.view3d_cy = 0.0
        self.view3d_half = DEFAULT_3D_HALF

    def is_empty(self) -> bool:
        if self.plot3d:
            return not (self.x_param and self.y_params and self.z_param)
        return not self.x_param or not self.y_params

    def to_dict(self) -> dict:
        data = {
            "x_param": self.x_param,
            "y_params": list(self.y_params),
            "show_legend": self.show_legend,
            "legend_position": self.legend_position,
            "legend_x": self.legend_x,
            "legend_y": self.legend_y,
            "legend_show_style": self.legend_show_style,
            "legend_show_param": self.legend_show_param,
            "legend_show_iso": self.legend_show_iso,
            "legend_show_file": self.legend_show_file,
            "legend_show_sim": self.legend_show_sim,
            "lock_x": self.lock_x,
            "lock_y": self.lock_y,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "view_initialised": self.view_initialised,
            "log_y": self.log_y,
            "show_grid": self.show_grid,
            "show_points": self.show_points,
            "color_mode": int(self.color_mode),
            "color_iso_param": self.color_iso_param,
            "style_mode": int(self.style_mode),
            "style_iso_param": self.style_iso_param,
            "plot3d": self.plot3d,
            "z_param": self.z_param,
            "azimuth": self.azimuth,
            "elevation": self.elevation,
            "show_3d_axes": self.show_3d_axes,
            "show_3d_grid": self.show_3d_grid,
            "view3d_cx": self.view3d_cx,
            "view3d_cy": self.view3d_cy,
            "view3d_half": self.view3d_half,
            "texts": [note.to_dict() for note in self.texts],
        }
        return data

    @staticmethod
    def from_dict(data: dict) -> "TileState":
        tile = TileState()
        for key, value in data.items():
            # Workspaces written before the wireframe cube was replaced by the
            # corner orientation triad carry the old key. Same intent -- "show
            # the 3D reference" -- so it is honoured rather than dropped.
            if key == "show_3d_box":
                tile.show_3d_axes = bool(value)
                continue
            if key == "color_mode":
                tile.color_mode = ColorMode(int(value))
            elif key == "style_mode":
                try:
                    tile.style_mode = StyleMode(int(value))
                except (TypeError, ValueError):
                    tile.style_mode = StyleMode.NONE
            elif key == "y_params":
                tile.y_params = list(value)
            elif key == "texts":
                tile.texts = [TextNote.from_dict(entry) for entry in value]
            elif hasattr(tile, key):
                setattr(tile, key, value)
        return tile


def tile_title(tile: "TileState", max_names: int = 4, wrap: int = 36) -> str:
    """What a tile is plotting, as "ALPHA, NZ = f(TIME)".

    Every selected Y is named. Long lists wrap onto two lines instead of
    being truncated: axis labels no longer carry those names.
    """
    del max_names
    if tile.plot3d:
        if not (tile.x_param and tile.y_params and tile.z_param):
            return "3D tile — choose X, Y and Z"
        return "{0} = f({1}, {2})".format(tile.z_param, tile.x_param,
                                          tile.y_params[0])

    if not tile.y_params and not tile.x_param:
        return "empty tile"
    if not tile.y_params:
        return "no parameter selected  (x: {0})".format(tile.x_param)
    if not tile.x_param:
        return _wrap_names(tile.y_params, suffix="  (no x axis selected)", wrap=wrap)

    suffix = " = f({0})".format(tile.x_param)
    return _wrap_names(tile.y_params, suffix=suffix, wrap=wrap)


def _wrap_names(names, suffix: str, wrap: int) -> str:
    shown = ", ".join(names)
    single = shown + suffix
    if len(names) < 2:
        return single
    if len(single) <= wrap and len(names) < 5:
        return single
    mid = (len(names) + 1) // 2
    return "{0},\n{1}{2}".format(
        ", ".join(names[:mid]),
        ", ".join(names[mid:]),
        suffix,
    )


def order_parameters(names, selected):
    """Selected parameters first (in selection order), then the rest as given."""
    seen = set()
    out = []
    pool = set(names)
    for name in selected:
        if name in pool and name not in seen:
            out.append(name)
            seen.add(name)
    for name in names:
        if name not in seen:
            out.append(name)
    return out


def curve_color_key(tile: "TileState", param: str, sim_key: str,
                    iso_label: str) -> str:
    if tile.color_mode == ColorMode.PARAMETER:
        return "param::" + param
    if tile.color_mode == ColorMode.ISO and tile.color_iso_param and iso_label:
        return "iso::{0}::{1}".format(tile.color_iso_param, iso_label)
    return "sim::" + sim_key


def curve_line_key(tile: "TileState", param: str, sim_key: str,
                   file_id: str, color_iso_label: str,
                   style_iso_label: str) -> str:
    if tile.style_mode == StyleMode.FILE:
        return "file::" + (file_id or sim_key)
    if tile.style_mode == StyleMode.SIMULATION:
        return "sim::" + sim_key
    if tile.style_mode == StyleMode.ISO and tile.style_iso_param and style_iso_label:
        return "iso::{0}::{1}".format(tile.style_iso_param, style_iso_label)
    return curve_color_key(tile, param, sim_key, color_iso_label)


def y_axis_label(tile: "TileState", max_names: int = 3) -> str:
    """The y axis caption from names alone.

    The plot widget has a richer version that appends the unit when every
    parameter on the axis shares one; that needs the session, which this module
    deliberately does not have. This is the Qt-free, core-free form, used for
    exports and tests.
    """
    if not tile.y_params:
        return ""
    shown = ", ".join(tile.y_params[:max_names])
    return shown + ("..." if len(tile.y_params) > max_names else "")


@dataclass
class PageState:
    """A grid of tiles, shown one page at a time."""

    name: str = "Page 1"
    rows: int = 1
    cols: int = 1
    tiles: List[TileState] = field(default_factory=lambda: [TileState() for _ in range(MAX_TILES)])

    @property
    def tile_count(self) -> int:
        return max(1, min(self.rows * self.cols, MAX_TILES))

    def visible_tiles(self) -> List[TileState]:
        return self.tiles[: self.tile_count]

    def to_dict(self) -> dict:
        # Only the tiles the grid actually shows are worth persisting.
        return {
            "name": self.name,
            "rows": self.rows,
            "cols": self.cols,
            "tiles": [t.to_dict() for t in self.tiles[: self.tile_count]],
        }

    @staticmethod
    def from_dict(data: dict) -> "PageState":
        page = PageState(
            name=data.get("name", "Page 1"),
            rows=int(data.get("rows", 1)),
            cols=int(data.get("cols", 1)),
        )
        tiles = [TileState.from_dict(t) for t in data.get("tiles", [])]
        while len(tiles) < MAX_TILES:
            tiles.append(TileState())
        page.tiles = tiles[:MAX_TILES]
        return page


@dataclass
class IsoState:
    """Filtering and grouping configuration for one ISO parameter.

    Several of these can be active at once and they combine with AND: a run is
    shown only if it passes every active filter. Colouring is separate and keys
    on exactly one parameter, chosen per tile.
    """

    param: str = ""
    active: bool = False
    grouped: bool = False
    group_count: int = 3
    # value -> shown. Ordered ascending by the UI that fills it.
    value_enabled: Dict[float, bool] = field(default_factory=dict)
    # Parallel to Session.iso_auto_group() output; stored as plain tuples so the
    # state model stays free of core types.
    groups: List[Tuple[str, float, float, bool]] = field(default_factory=list)

    def enabled_values(self) -> List[float]:
        return [v for v, on in self.value_enabled.items() if on]


def iso_state_to_dict(state: IsoState) -> dict:
    """One ISO filter, as JSON.

    ``value_enabled`` is keyed by float, and JSON object keys are strings, so
    it is written as a list of pairs instead of an object: round-tripping
    ``{0.75: False}`` through an object would hand back the string ``"0.75"``
    and silently stop matching the value it was about. Non-finite keys are
    dropped, because a NaN value is never admitted by any filter anyway and
    ``NaN`` is not portable JSON.
    """
    values = [
        [float(value), bool(shown)]
        for value, shown in state.value_enabled.items()
        if math.isfinite(value)
    ]
    return {
        "param": state.param,
        "active": bool(state.active),
        "grouped": bool(state.grouped),
        "group_count": int(state.group_count),
        "value_enabled": values,
        "groups": [
            [str(name), float(low), float(high), bool(active)]
            for name, low, high, active in state.groups
        ],
    }


def iso_state_from_dict(data: object) -> Optional[IsoState]:
    """Rebuild one ISO filter, or return None if the entry is unusable.

    Everything here is defensive on purpose. A workspace is a file a user can
    edit, keep for a year, or write with an older version, and an ISO filter
    that cannot be read is not worth failing a load over -- the rest of the
    layout is still wanted. A filter naming a parameter that no longer exists
    is *kept*: it costs nothing, it shows in the ISO tab as "not in the loaded
    files", and it starts working again the moment that file is reopened.
    """
    if not isinstance(data, dict):
        return None
    param = data.get("param")
    if not isinstance(param, str) or not param.strip():
        return None

    state = IsoState(param=param.strip())
    state.active = bool(data.get("active", False))
    state.grouped = bool(data.get("grouped", False))
    try:
        state.group_count = max(1, int(data.get("group_count", 3)))
    except (TypeError, ValueError):
        state.group_count = 3

    # Accept the list-of-pairs form, and also an object keyed by a numeric
    # string, so a hand-edited workspace does the obvious thing.
    raw_values = data.get("value_enabled", [])
    pairs: List[Tuple[object, object]] = []
    if isinstance(raw_values, dict):
        pairs = list(raw_values.items())
    elif isinstance(raw_values, (list, tuple)):
        pairs = [tuple(p) for p in raw_values if isinstance(p, (list, tuple)) and len(p) == 2]
    for value, shown in pairs:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            state.value_enabled[number] = bool(shown)

    raw_groups = data.get("groups", [])
    if isinstance(raw_groups, (list, tuple)):
        for entry in raw_groups:
            if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                continue
            try:
                low = float(entry[1])
                high = float(entry[2])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(low) and math.isfinite(high)):
                continue
            if high < low:
                low, high = high, low
            name = entry[0] if isinstance(entry[0], str) else ""
            active = bool(entry[3]) if len(entry) > 3 else True
            state.groups.append((name or "{0:g} - {1:g}".format(low, high),
                                 low, high, active))

    # A grouped filter whose intervals did not survive is demoted rather than
    # left in a state the ISO tab would have to invent intervals for.
    if state.grouped and not state.groups:
        state.grouped = False
    return state


class StyleRegistry:
    """Colour, dash style, marker and width per styling key.

    Keys are namespaced by what they describe, so the three colour modes coexist
    without collisions:

        sim::<simulation unique key>
        param::<parameter name>
        iso::<iso parameter>::<value or group label>

    Colours are sticky: once a key has been given a colour it keeps it for the
    whole session, so a curve does not change colour when an unrelated file is
    loaded. Deriving palette indices from the current list order instead would
    mean adding a file could recolour everything already on screen.
    """

    def __init__(self) -> None:
        self._colors: Dict[str, str] = {}
        self._dashes: Dict[str, str] = {}
        self._symbols: Dict[str, str] = {}
        self._widths: Dict[str, float] = {}
        self._next_color = 0

        # Applied to any curve without its own explicit width.
        self.line_width: float = 1.6
        self.symbol_size: float = 6.0

    # --------------------------------------------------------------- colour

    def color_for(self, key: str) -> str:
        if key not in self._colors:
            self._colors[key] = PALETTE[self._next_color % len(PALETTE)]
            self._next_color += 1
        return self._colors[key]

    def has_color(self, key: str) -> bool:
        return key in self._colors

    def set_color(self, key: str, color: str) -> None:
        self._colors[key] = color

    # ---------------------------------------------------------- line style

    def dash_for(self, key: str) -> str:
        return self._dashes.get(key, "solid")

    def has_dash(self, key: str) -> bool:
        return key in self._dashes

    def assign_line_style(self, key: str, index: int) -> None:
        if key not in self._dashes:
            self._dashes[key] = DASH_STYLES[index % len(DASH_STYLES)]
        if index >= len(DASH_STYLES) and key not in self._symbols:
            extras = [code for code in SYMBOLS if code]
            if extras:
                self._symbols[key] = extras[(index - len(DASH_STYLES)) % len(extras)]

    def set_dash(self, key: str, style: str) -> None:
        if style in DASH_STYLES:
            self._dashes[key] = style

    def symbol_for(self, key: str) -> str:
        return self._symbols.get(key, "")

    def set_symbol(self, key: str, symbol: str) -> None:
        if symbol in SYMBOLS:
            self._symbols[key] = symbol

    def width_for(self, key: str) -> float:
        return self._widths.get(key, self.line_width)

    def set_width(self, key: str, width: float) -> None:
        self._widths[key] = max(0.1, min(float(width), 10.0))

    def clear_width_overrides(self) -> None:
        """Drop per-curve widths so every curve follows the global setting."""
        self._widths.clear()

    def reset(self, key: str) -> None:
        for table in (self._colors, self._dashes, self._symbols, self._widths):
            table.pop(key, None)

    # ------------------------------------------------------------ persist

    def to_dict(self) -> dict:
        return {
            "colors": dict(self._colors),
            "dashes": dict(self._dashes),
            "symbols": dict(self._symbols),
            "widths": dict(self._widths),
            "line_width": self.line_width,
            "symbol_size": self.symbol_size,
        }

    def load_dict(self, data: dict) -> None:
        self._colors = dict(data.get("colors", {}))
        self._dashes = dict(data.get("dashes", {}))
        self._symbols = dict(data.get("symbols", {}))
        self._widths = {k: float(v) for k, v in data.get("widths", {}).items()}
        self.line_width = float(data.get("line_width", 1.6))
        self.symbol_size = float(data.get("symbol_size", 6.0))
        self._next_color = len(self._colors)


class AppState:
    """Everything the UI needs that is not owned by the C++ core."""

    def __init__(self) -> None:
        self.pages: List[PageState] = [PageState()]
        self.active_page: int = 0
        self.active_tile: int = 0

        # (file_index, sim_index) pairs ticked in the DATA tab. One of the two
        # constraints on what is drawn; the ISO filters are the other, and
        # plotted_sims() is where they meet.
        self.selected_sims: Set[Tuple[int, int]] = set()

        self.iso_states: Dict[str, IsoState] = {}

        # Which runs the ISO filters admit, or None while none is active.
        # Derived from iso_states and the files, so it is not saved -- the
        # filters are, and resolve_iso_filters() rebuilds this from them
        # whenever the loaded files change.
        self._iso_admitted: Optional[Set[Tuple[int, int]]] = None
        # Diagnostics from the last resolve: filters that cannot apply because
        # no open file carries the parameter, and filters that apply but admit
        # nothing. Both exist so the interface can explain an empty plot.
        self._iso_inapplicable: Set[str] = set()
        self._iso_empty: Set[str] = set()
        self._plotted: Optional[Tuple[Tuple[int, int], ...]] = None
        self._plotted_from: int = -1
        self.styles = StyleRegistry()
        self.markers: List[Marker] = []
        self.areas: List[AreaBand] = []

        # Scale constraints declared for a *variable* rather than for a tile.
        # "Apply X range to every tile with the same X axis" used to walk the
        # tiles that existed at that moment and stop there, so a tile filled in
        # afterwards -- the common case, since you set up the scale first and
        # then populate the grid -- silently ignored the constraint. Recording
        # it against the parameter means it applies whenever that parameter
        # lands on a tile, now or later.
        self.global_x_limits: Dict[str, Tuple[float, float]] = {}
        self.global_y_limits: Dict[str, Tuple[float, float]] = {}

        # Tile copy/cut/paste.
        self.tile_clipboard: Optional[TileState] = None
        self.clipboard_is_cut: bool = False

        self.dark_mode: bool = True
        # How large the interface is drawn, as a multiple of the base type
        # scale. A view preference like the theme, and saved with it: someone
        # who has set the window to suit their screen should not have to set
        # it again every time they open a workspace.
        self.ui_scale: float = 1.0
        self.workspace_path: Optional[str] = None
        self.dirty: bool = False
        self.mouse_mode: MouseMode = MouseMode.SELECT

    # ----------------------------------------------------------------- iso

    def active_iso_filters(self) -> List["IsoState"]:
        """Every ISO parameter currently filtering, in name order."""
        return [
            self.iso_states[name]
            for name in sorted(self.iso_states)
            if self.iso_states[name].active
        ]

    def iso_state(self, param: str) -> "IsoState":
        if param not in self.iso_states:
            self.iso_states[param] = IsoState(param=param)
        return self.iso_states[param]

    def iso_admits(self, iso: "IsoState", value: float) -> bool:
        """Does one ISO filter let a run holding this value through?"""
        if value != value:      # NaN: the run has no value for this parameter
            return False
        if iso.grouped and iso.groups:
            index = isogroups.group_of(iso.groups, value)
            return index is not None and iso.groups[index][3]
        return iso.value_enabled.get(float(value), True)

    def resolve_iso_filters(self, files, values_for) -> None:
        """Work out which runs the ISO filters admit, and remember it.

        The ISO tab and the DATA tab are independent constraints, and this
        computes the ISO half on its own. It used to be written *into* the DATA
        selection -- a pass over every run calling set_selected() -- so
        applying a filter silently reticked the DATA tab: ask for a mass
        between 50000 and 100000 and every run in that band joined the plot
        whether or not you had chosen it, and every run outside it was
        unticked whether or not you had. Two controls, one piece of state, and
        each one overwriting the other's answer.

        Recorded separately instead, so neither tab can move the other's
        checkboxes, and plotted_sims() intersects the two to decide what is
        drawn. Filters still combine with AND -- adding a second one should
        narrow the set, not widen it.

        `files` is an iterable of (file_index, simulation_count) and
        `values_for(file_index, param)` returns that file's ISO column, both
        passed in because this module knows nothing about sessions. One column
        read per parameter per file, and each reads only the first record of
        every block, so this stays cheap on large files.
        """
        self._plotted = None
        filters = self.active_iso_filters()
        if not filters:
            # None rather than a set of every run. "No filter" then costs
            # nothing to represent and nothing to keep up to date when a file
            # arrives, which is the common case by a wide margin.
            self._iso_admitted = None
            self._iso_inapplicable = set()
            self._iso_empty = set()
            return

        admitted: Set[Tuple[int, int]] = set()
        self._iso_inapplicable = set()
        self._iso_empty = set()
        matched = {iso.param: 0 for iso in filters}

        for file_index, count in files:
            columns = []
            for iso in filters:
                column = values_for(file_index, iso.param)
                if len(column) == 0:
                    # This file does not carry the parameter at all. A filter
                    # cannot speak about a channel that is not there, so it
                    # does not constrain this file's runs.
                    #
                    # The previous code broke out of the loop here, which
                    # skipped the for/else and therefore excluded the run. That
                    # is defensible in isolation and wrong in practice: it made
                    # a filter on a parameter only *some* files carry hide every
                    # run of the others, and -- now that filters are restored
                    # from a workspace -- it made a workspace whose ISO
                    # parameter has since disappeared open to an empty plot with
                    # no explanation. A filter with nothing to say says nothing.
                    self._iso_inapplicable.add(iso.param)
                    continue
                columns.append((iso, column))

            for sim_index in range(count):
                for iso, column in columns:
                    if sim_index >= len(column):
                        # Short column: no value for this run, same reasoning.
                        continue
                    if not self.iso_admits(iso, float(column[sim_index])):
                        break
                else:
                    admitted.add((file_index, sim_index))
                    for iso, column in columns:
                        if sim_index < len(column):
                            matched[iso.param] += 1

        # A filter that is applicable but admits nothing is a different case
        # again, and a legitimate one -- you asked for a mass band no run is in.
        # It is recorded so the interface can say so rather than leaving an
        # empty plot to be puzzled over.
        self._iso_empty = {
            param for param, hits in matched.items()
            if hits == 0 and param not in self._iso_inapplicable
        }
        self._iso_admitted = admitted

    def iso_filters_resolved(self) -> bool:
        """True when an ISO filter is narrowing the selection."""
        return self._iso_admitted is not None

    def iso_inapplicable_params(self) -> Set[str]:
        """Active filters naming a parameter none of the open files carries.

        These constrain nothing. Reported so the ISO tab can label them and a
        workspace restored against different data can explain itself.
        """
        return set(getattr(self, "_iso_inapplicable", set()))

    def iso_empty_params(self) -> Set[str]:
        """Active, applicable filters that admit no run at all."""
        return set(getattr(self, "_iso_empty", set()))

    def iso_admitted(self, file_index: int, sim_index: int) -> bool:
        """Do the ISO filters admit this run, ignoring the DATA tab?

        The question the DATA tab asks to explain itself: a row that is ticked
        but answers False here is the reason a run the user chose is not on
        screen.
        """
        return (self._iso_admitted is None
                or (file_index, sim_index) in self._iso_admitted)

    # ------------------------------------------------------ global limits

    def set_global_x_limit(self, param: str, low: float, high: float) -> None:
        if param:
            self.global_x_limits[param] = (min(low, high), max(low, high))
            self.dirty = True

    def set_global_y_limit(self, param: str, low: float, high: float) -> None:
        if param:
            self.global_y_limits[param] = (min(low, high), max(low, high))
            self.dirty = True

    def clear_global_limit(self, param: str) -> None:
        self.global_x_limits.pop(param, None)
        self.global_y_limits.pop(param, None)
        self.dirty = True

    def y_limit_conflict(self, tile: "TileState") -> List[Tuple[str, float, float]]:
        """The tile's y parameters whose declared limits disagree.

        Empty when the tile has fewer than two constrained y parameters, or when
        every one of them asks for the same window. Otherwise every constrained
        parameter and its range, in name order, so the interface can name them.

        A tile has one y axis and cannot satisfy two different demands on it.
        See apply_global_limits() for what is done about that and why.
        """
        bounds = {p: self.global_y_limits[p] for p in tile.y_params
                  if p in self.global_y_limits}
        if len(set(bounds.values())) < 2:
            return []
        return sorted((p, low, high) for p, (low, high) in bounds.items())

    def apply_global_limits(self, tile: "TileState") -> bool:
        """Impose any declared variable limits on one tile. True if it changed.

        Called whenever a tile's parameters change, which is what makes the
        constraint apply to tiles filled in after it was declared rather than
        only to the ones that happened to exist at the time.

        A 3D tile is left alone: its frame is view3d_cx/cy/half, and x_min..y_max
        describe data ranges that mean nothing to a projection. Locking them did
        no visible harm but it marked the workspace dirty and left the tile
        mysteriously locked if it was ever switched back to two dimensions.
        """
        if tile.plot3d:
            return False

        changed = False

        # x cannot conflict: a tile has exactly one x parameter, so this lookup
        # returns at most one range however many constraints are declared.
        limit = self.global_x_limits.get(tile.x_param)
        if limit is not None:
            if not tile.lock_x or (tile.x_min, tile.x_max) != limit:
                tile.lock_x = True
                tile.x_min, tile.x_max = limit
                changed = True

        # y can conflict, because a tile may draw several parameters on its one
        # y axis and each of them may carry its own declared window.
        #
        # This used to impose the union of them. The union is a superset of both
        # and therefore honours neither, which quietly defeats the whole purpose
        # of the constraint -- the point is that two tiles showing the same
        # parameter can be compared directly, and under a union they cannot.
        # Worse, when the ranges differ in magnitude the union annihilates the
        # smaller channel: ALTITUDE at [0, 13000] unioned with MACH at [0, 1]
        # leaves Mach occupying 0.008% of the axis, a flat line on the bottom
        # edge, with nothing on screen to say why.
        #
        # There is no arithmetic that satisfies both demands, so none is
        # invented. Parameters that agree are applied; parameters that disagree
        # leave the tile exactly as the user had it -- free, and therefore
        # auto-fitting to show all of its data -- and the conflict is reported
        # through y_limit_conflict() so the SCALE tab can name it and offer the
        # union as a deliberate choice.
        if not self.y_limit_conflict(tile):
            bounds = [self.global_y_limits[p] for p in tile.y_params
                      if p in self.global_y_limits]
            if bounds:
                low, high = bounds[0]
                if not tile.lock_y or (tile.y_min, tile.y_max) != (low, high):
                    tile.lock_y = True
                    tile.y_min, tile.y_max = low, high
                    changed = True

        if changed:
            self.dirty = True
        return changed

    def apply_global_limits_everywhere(self) -> None:
        for page in self.pages:
            for tile in page.visible_tiles():
                self.apply_global_limits(tile)

    def apply_legend_everywhere(self, source: "TileState") -> int:
        """Give every tile, on every page, the legend of `source`.

        Comparing a set of tiles is the ordinary reason to have several, and a
        legend configured on one of them was configured for all of them --
        setting the same four tick boxes and the same corner eight or twenty
        times over is not a decision, it is transcription.

        Every page, not just the visible one, because the same is true across
        pages and because the alternative is a button whose effect depends on
        which page happens to be on screen. This matches "Apply colour mode to
        all tiles" above it, and the hidden tiles of a shrunken grid are
        included for the same reason those are: a tile that comes back when the
        grid grows should come back consistent with the rest.

        Returns how many tiles it changed, so the caller can say so rather than
        leaving the user to go and look. Tiles that already match are not
        counted and are not touched, which is what keeps a second click from
        reporting work it did not do.
        """
        changed = 0
        for page in self.pages:
            for tile in page.tiles:
                if tile is source:
                    continue
                values = [(name, getattr(source, name))
                          for name in TileState.LEGEND_FIELDS]
                if all(getattr(tile, name) == value for name, value in values):
                    continue
                for name, value in values:
                    setattr(tile, name, value)
                changed += 1
        if changed:
            self.dirty = True
        return changed

    def default_x_param(self, available: Sequence[str]) -> str:
        """The x axis to reach for when a tile has not been given one.

        A time-like channel when the files have one, which is the x axis of
        very nearly every one of these; otherwise the first parameter, so the
        answer is never empty when anything is loaded.
        """
        if not available:
            return ""
        return next((p for p in available if p.upper().startswith("TIME")),
                    available[0])

    def default_iso_param(self, available: Sequence[str]) -> str:
        """The parameter to colour by when a tile has chosen ISO but not which.

        One owner for the rule, because it is read in two places that must not
        disagree: here, to commit the choice, and the STYLE panel, to show it
        in the dropdown before it is committed. A panel that displayed one
        parameter while the tile adopted another is exactly the confusion this
        prevents.

        A parameter the user has already set up in the ISO tab comes first --
        having configured it is the clearest statement of interest available --
        and otherwise the same default the x axis gets, so the answer is never
        empty while anything is loaded.
        """
        if not available:
            return ""
        names = set(available)
        configured = [p for p in self.iso_states if p in names]
        return configured[0] if configured else self.default_x_param(available)

    def apply_parameter_defaults(self, available: Sequence[str]) -> bool:
        """Fill in the choices a tile needs but the user has not made.

        A tile starts with no x axis at all. The PARAMS panel shows a sensible
        default in its combo -- TIME, usually -- but showing is not choosing,
        and a panel refresh must not edit the tile behind the user's back: that
        was AOE-14, where the same click left different state depending on
        which tab happened to be open. So the default has to be committed
        somewhere deliberate, and this is it: once per redraw, silently,
        wherever a tile is actually plotting something and still has no x axis.

        Without it a tile showed "no x axis selected" while the panel beside it
        displayed TIME as the selection -- the display and the state disagreed,
        and the display was the one telling the truth about what the user
        thought they had picked.

        Blank tiles are left alone. Giving an x axis to all twenty-five cells
        of an empty grid would relabel every one of them for no benefit.
        """
        if not available:
            return False

        default_x = self.default_x_param(available)
        names = set(available)
        changed = False

        for page in self.pages:
            for tile in page.tiles:
                # Only tiles with something to draw. A tile with neither is
                # empty, not misconfigured.
                if not tile.x_param and (tile.y_params or tile.z_param):
                    tile.x_param = default_x
                    changed = True

                # The same reasoning for colour-by-ISO: the mode is selected,
                # the parameter it needs never was, and the curves come out one
                # colour with nothing to say why.
                if tile.color_mode == ColorMode.ISO and not tile.color_iso_param:
                    chosen = self.default_iso_param(available)
                    if chosen:
                        tile.color_iso_param = chosen
                        changed = True

                if tile.style_mode == StyleMode.ISO and not tile.style_iso_param:
                    chosen = tile.color_iso_param or self.default_iso_param(available)
                    if chosen:
                        tile.style_iso_param = chosen
                        changed = True

        if changed:
            self.dirty = True
        return changed

    def resolve_missing_parameters(self, available: Sequence[str]) -> List[str]:
        """Re-point tiles at channels that exist, once, when the files change.

        A tile can name a channel none of the open files carry: a workspace
        saved against a different study, or a file removed mid-session. That has
        to be settled somewhere, and the wrong somewhere is a panel's refresh --
        those run only when their tab is shown, so the same click left different
        state behind depending on which tab happened to be open, and nothing
        marked the workspace dirty.

        Doing it here makes it one deliberate step at a known moment, and
        returns what changed so the window can say so rather than leaving the
        user to discover their x axis has moved.

        Only the axis channels are repaired. A y channel that is missing is left
        named: it plots nothing today and comes back by itself when the file it
        belongs to is reopened, which is much the friendlier behaviour when a
        study is loaded a file at a time.
        """
        if not available:
            # No files open. Nothing is missing; everything is simply absent,
            # and clearing the tiles here would lose the user's layout.
            return []

        names = set(available)
        default_x = self.default_x_param(available)
        notes: List[str] = []

        for page_index, page in enumerate(self.pages):
            for tile_index, tile in enumerate(page.tiles):
                where = "page {0}, tile {1}".format(page_index + 1, tile_index + 1)

                if tile.x_param and tile.x_param not in names:
                    notes.append("{0}: x axis {1} -> {2}".format(
                        where, tile.x_param, default_x))
                    tile.x_param = default_x

                if tile.z_param and tile.z_param not in names:
                    notes.append("{0}: z axis {1} cleared".format(where, tile.z_param))
                    tile.z_param = ""

                # Both ISO encodings, not just the colour's. Style-by-ISO
                # arrived in 2.1.0 and this repair did not follow it, so a tile
                # dashed by a parameter the new files do not carry kept naming
                # it: curve_line_key() then fell through to the colour key and
                # the dashes quietly stopped distinguishing anything, with the
                # STYLE tab still showing the old parameter as the answer.
                for field, label in (("color_iso_param", "colour-by"),
                                     ("style_iso_param", "style-by")):
                    current = getattr(tile, field)
                    if not current or current in names:
                        continue
                    configured = [p for p in self.iso_states if p in names]
                    replacement = configured[0] if configured else ""
                    notes.append("{0}: {1} {2} -> {3}".format(
                        where, label, current, replacement or "none"))
                    setattr(tile, field, replacement)

        if notes:
            self.dirty = True
        return notes

    # -------------------------------------------------------------- pages

    @property
    def page(self) -> PageState:
        self.active_page = max(0, min(self.active_page, len(self.pages) - 1))
        return self.pages[self.active_page]

    @property
    def tile(self) -> TileState:
        page = self.page
        index = max(0, min(self.active_tile, page.tile_count - 1))
        return page.tiles[index]

    def add_page(self) -> PageState:
        page = PageState(name="Page {0}".format(len(self.pages) + 1))
        self.pages.append(page)
        self.active_page = len(self.pages) - 1
        self.dirty = True
        return page

    def remove_page(self, index: int) -> None:
        if len(self.pages) <= 1:
            return
        if 0 <= index < len(self.pages):
            self.pages.pop(index)
            self.active_page = min(self.active_page, len(self.pages) - 1)
            self.dirty = True

    def rename_page(self, index: int, name: str) -> None:
        name = name.strip()
        if name and 0 <= index < len(self.pages):
            self.pages[index].name = name
            self.dirty = True

    def move_page(self, index: int, delta: int) -> int:
        """Reorder a page. Returns its new index."""
        target = index + delta
        if not (0 <= index < len(self.pages)) or not (0 <= target < len(self.pages)):
            return index
        self.pages[index], self.pages[target] = self.pages[target], self.pages[index]
        self.dirty = True
        return target

    # --------------------------------------------------------------- tiles

    def move_tile(self, direction: str) -> bool:
        """Swap the active tile with its neighbour. Returns True if it moved.

        Swapping rather than inserting keeps the grid full and means the move is
        its own inverse, which is what people expect
        from arrow buttons.
        """
        page = self.page
        cols = max(1, page.cols)
        index = self.active_tile
        row, col = divmod(index, cols)

        if direction == "left" and col > 0:
            target = index - 1
        elif direction == "right" and col < cols - 1:
            target = index + 1
        elif direction == "up" and row > 0:
            target = index - cols
        elif direction == "down" and row < page.rows - 1:
            target = index + cols
        else:
            return False

        if not (0 <= target < page.tile_count):
            return False

        page.tiles[index], page.tiles[target] = page.tiles[target], page.tiles[index]
        self.active_tile = target
        self.dirty = True
        return True

    def copy_tile(self) -> None:
        self.tile_clipboard = TileState.from_dict(self.tile.to_dict())
        self.clipboard_is_cut = False

    def cut_tile(self) -> None:
        self.copy_tile()
        self.clear_tile()
        self.clipboard_is_cut = True

    def paste_tile(self) -> bool:
        if self.tile_clipboard is None:
            return False
        page = self.page
        index = max(0, min(self.active_tile, page.tile_count - 1))
        page.tiles[index] = TileState.from_dict(self.tile_clipboard.to_dict())
        self.dirty = True
        return True

    def clear_tile(self) -> None:
        page = self.page
        index = max(0, min(self.active_tile, page.tile_count - 1))
        page.tiles[index] = TileState()
        self.dirty = True

    # ---------------------------------------------------------- selection

    def is_selected(self, file_index: int, sim_index: int) -> bool:
        return (file_index, sim_index) in self.selected_sims

    def set_selected(self, file_index: int, sim_index: int, on: bool) -> None:
        pair = (file_index, sim_index)
        if on:
            self.selected_sims.add(pair)
        else:
            self.selected_sims.discard(pair)
        self._plotted = None
        self.dirty = True

    def clear_selection(self) -> None:
        self.selected_sims.clear()
        self._plotted = None
        self.dirty = True

    def is_plotted(self, file_index: int, sim_index: int) -> bool:
        """Will this run be drawn?

        Ticked in the DATA tab *and* admitted by every active ISO filter. Both
        halves, which is the whole point: a run whose ISO value is filtered out
        stays off every tile however firmly it is ticked, and a run the ISO
        filters admit stays off until it is ticked.
        """
        pair = (file_index, sim_index)
        if pair not in self.selected_sims:
            return False
        return self._iso_admitted is None or pair in self._iso_admitted

    def plotted_sims(self) -> Tuple[Tuple[int, int], ...]:
        """Every run that will be drawn, in order.

        Cached, because this is the hot question: every tile asks it on every
        rebuild, and a page of twenty-five tiles asks twenty-five times for an
        answer that cannot have changed in between.

        Invalidated by set_selected() and by resolve_iso_filters(), which are
        the two doors this can change through -- and also whenever the number
        of ticked runs no longer matches what the cache was built from, which
        catches selected_sims being written directly. A same-sized swap done
        that way would slip past; nothing in the application does one, and
        going through set_selected() is why.
        """
        if self._plotted is None or self._plotted_from != len(self.selected_sims):
            if self._iso_admitted is None:
                pairs = self.selected_sims
            else:
                pairs = self.selected_sims & self._iso_admitted
            self._plotted = tuple(sorted(pairs))
            self._plotted_from = len(self.selected_sims)
        return self._plotted

    def excluded_by_iso(self) -> int:
        """How many ticked runs the ISO filters are holding back.

        The number that explains a plot which came back shorter than the DATA
        tab suggests it should.
        """
        if self._iso_admitted is None:
            return 0
        return len(self.selected_sims) - len(self.plotted_sims())

    def selection_for_file(self, file_index: int) -> List[int]:
        return sorted(s for f, s in self.selected_sims if f == file_index)

    def drop_file(self, file_index: int) -> None:
        """Forget selections for a removed file and shift higher indices down."""
        def remap(pairs):
            moved = set()
            for f, s in pairs:
                if f == file_index:
                    continue
                moved.add((f - 1, s) if f > file_index else (f, s))
            return moved

        self.selected_sims = remap(self.selected_sims)
        # The ISO answer is indexed by file too, so it has to move with it or
        # it would start admitting the wrong runs of the wrong files.
        if self._iso_admitted is not None:
            self._iso_admitted = remap(self._iso_admitted)
        self._plotted = None

    # ---------------------------------------------------------- workspace

    def to_dict(self) -> dict:
        return {
            "version": 3,
            "app_version": "2.1.2",
            "pages": [p.to_dict() for p in self.pages],
            "active_page": self.active_page,
            "selected_sims": sorted(list(self.selected_sims)),
            "styles": self.styles.to_dict(),
            "dark_mode": self.dark_mode,
            "ui_scale": self.ui_scale,
            "mouse_mode": int(self.mouse_mode),
            "markers": [
                {"x_param": m.x_param, "position": m.position,
                 "visible": m.visible, "color": m.color,
                 "dot_size": m.dot_size}
                for m in self.markers
            ],
            "areas": [
                {"x_param": a.x_param, "x_min": a.x_min, "x_max": a.x_max,
                 "color": a.color, "alpha": a.alpha,
                 "visible": a.visible}
                for a in self.areas
            ],
            "global_x_limits": {k: list(v) for k, v in self.global_x_limits.items()},
            "global_y_limits": {k: list(v) for k, v in self.global_y_limits.items()},
            "iso_states": [iso_state_to_dict(s) for s in self.iso_states.values()],
        }

    def load_dict(self, data: dict) -> None:
        pages = [PageState.from_dict(p) for p in data.get("pages", [])]
        self.pages = pages or [PageState()]
        self.active_page = int(data.get("active_page", 0))
        self.selected_sims = {tuple(pair) for pair in data.get("selected_sims", [])}
        # The filter *definitions* are saved -- which parameters are filtered
        # on, their intervals and which values are ticked. What is not saved is
        # the *answer*: which runs those filters admit describes the files the
        # last workspace had open, and MainWindow.refresh_all() rebuilds it
        # once the new ones are known.
        self.iso_states = {}
        for entry in data.get("iso_states", []):
            state = iso_state_from_dict(entry)
            if state is not None:
                self.iso_states[state.param] = state
        self._iso_admitted = None
        self._plotted = None
        self.styles.load_dict(data.get("styles", {}))
        self.dark_mode = bool(data.get("dark_mode", True))
        # Absent from every workspace written before 2.1.2, and from one
        # written by hand; 1.0 is the size those were laid out at.
        try:
            self.ui_scale = float(data.get("ui_scale", 1.0))
        except (TypeError, ValueError):
            self.ui_scale = 1.0
        self.mouse_mode = MouseMode(int(data.get("mouse_mode", int(MouseMode.SELECT))))
        self.markers = [
            Marker(**{k: v for k, v in m.items() if k in Marker.__annotations__})
            for m in data.get("markers", [])
        ]
        self.areas = [
            AreaBand(**{k: v for k, v in a.items() if k in AreaBand.__annotations__})
            for a in data.get("areas", [])
        ]
        self.global_x_limits = {
            k: (float(v[0]), float(v[1]))
            for k, v in data.get("global_x_limits", {}).items() if len(v) == 2
        }
        self.global_y_limits = {
            k: (float(v[0]), float(v[1]))
            for k, v in data.get("global_y_limits", {}).items() if len(v) == 2
        }
        self.dirty = False


def save_workspace(path: str, state: AppState, files: Sequence[str],
                   computed: Dict[str, str], aliases: Dict[str, str],
                   display_names: Optional[Dict[str, dict]] = None) -> None:
    """Write a workspace as JSON.

    JSON rather than an ad-hoc delimited text format: such a format breaks
    the moment a file path or parameter name contained the delimiter, and could
    not be extended without a version bump nobody implemented.

    `display_names` carries the labels the user gave files and runs in the DATA
    tab. It is optional and written only when it holds something, so a session
    in which nothing was renamed writes the same keys 2.1.0 wrote, and a reader
    that does not know these two ignores them. That is why they are extra keys
    rather than a schema bump: nothing older needs to understand them to keep
    working, and nothing newer is confused by their absence.
    """
    payload = state.to_dict()
    payload["files"] = list(files)
    payload["computed"] = dict(computed)
    payload["aliases"] = dict(aliases)
    if display_names:
        file_names = {
            key: value
            for key, value in (display_names.get("files") or {}).items()
            if isinstance(key, str) and isinstance(value, str) and value
        }
        sim_names = {}
        for file_key, per_sim in (display_names.get("simulations") or {}).items():
            if not isinstance(file_key, str) or not isinstance(per_sim, dict):
                continue
            kept = {
                sim_key: name
                for sim_key, name in per_sim.items()
                if isinstance(sim_key, str) and isinstance(name, str) and name
            }
            if kept:
                sim_names[file_key] = kept
        if file_names:
            payload["file_names"] = file_names
        if sim_names:
            payload["sim_names"] = sim_names
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_workspace(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
