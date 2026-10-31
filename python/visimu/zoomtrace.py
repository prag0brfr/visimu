# Author: Paulo R. A. Gilz
"""A trace of the zoom gesture, for diagnosing a zoom that stops working.

Temporary instrumentation. It exists to answer one question -- why does a
rubber-band zoom sometimes fail to start -- from a log file produced on the
machine where it happens, rather than from a description of it.

What it records, and why each piece is here:

* every mouse press, drag step and release reaching a tile, with the button,
  the modifiers and the position;
* which branch of the tile's own handlers took the event, and whether the
  event was handed on to pyqtgraph or consumed;
* the view box's state at each step: its mouse mode (RectMode is the one that
  rubber-bands), whether the mouse is enabled on each axis, and whether the
  band item is visible;
* **the scene's drag bookkeeping**, which is where a stuck gesture would live.
  pyqtgraph does not drive the band from the widget's events: the scene
  collects a press, decides on the first move that a drag is in progress, and
  remembers the button and the item it belongs to until the matching release.
  If a release never reaches it, `dragButtons` keeps that button for ever and
  every later press is read as the continuation of a drag that ended minutes
  ago -- which looks exactly like "the band stopped appearing".

Every line is one record, `key=value` throughout, so the log can be read with
the eye and filtered with grep:

    ZOOM press tile=3 mode=ZOOM button=Left ... scene.dragButtons=[]

Moves are throttled: a drag is hundreds of them, and the first few plus a
periodic sample say everything the rest would. The count is reported on the
release, so nothing is hidden by the throttle.

It is on by default -- the point is a log from a machine nobody can attach a
debugger to -- and `VISIMU_ZOOM_TRACE=0` turns it off.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

LOGGER_NAME = "visimu.zoom"

# How many moves of one drag are logged in full before the sample rate drops,
# and one in how many after that.
MOVES_IN_FULL = 3
MOVE_SAMPLE = 25

_log = logging.getLogger(LOGGER_NAME)
_enabled: Optional[bool] = None


def enabled() -> bool:
    """Is the trace on? Read once; set VISIMU_ZOOM_TRACE=0 to turn it off."""
    global _enabled
    if _enabled is None:
        flag = os.environ.get("VISIMU_ZOOM_TRACE", "1").strip().lower()
        _enabled = flag not in ("0", "false", "no", "off")
    return _enabled


def banner() -> None:
    """Said once at startup, so a log without the trace is obvious."""
    if enabled():
        _log.info("ZOOM trace on (VISIMU_ZOOM_TRACE=0 to turn it off)")


def log(what: str, **fields) -> None:
    """One record. Never raises: a broken trace must not break the program."""
    if not enabled():
        return
    try:
        parts = ["ZOOM", what]
        for key in sorted(fields):
            parts.append("{0}={1}".format(key, fields[key]))
        _log.info(" ".join(parts))
    except Exception:                                    # pragma: no cover
        pass


# ------------------------------------------------------------- descriptions


def button_name(event) -> str:
    """The button of an event, by name, without needing Qt to be importable."""
    try:
        value = int(event.button())
    except (AttributeError, TypeError, ValueError):
        return "?"
    return {0: "None", 1: "Left", 2: "Right", 4: "Middle"}.get(value, str(value))


def buttons_held(event) -> str:
    try:
        return str(int(event.buttons()))
    except (AttributeError, TypeError, ValueError):
        return "?"


def modifier_names(event) -> str:
    try:
        bits = int(event.modifiers())
    except (AttributeError, TypeError, ValueError):
        return "?"
    names = []
    for bit, name in ((0x02000000, "Shift"), (0x04000000, "Ctrl"),
                      (0x08000000, "Alt"), (0x10000000, "Meta")):
        if bits & bit:
            names.append(name)
    return "+".join(names) if names else "none"


def position(event) -> str:
    """Where the event happened, in widget pixels."""
    for getter in ("position", "pos", "scenePos"):
        try:
            point = getattr(event, getter)()
            return "{0:.1f},{1:.1f}".format(float(point.x()), float(point.y()))
        except (AttributeError, TypeError, ValueError):
            continue
    return "?"


def view_state(view) -> dict:
    """The view box's state, as the band's behaviour depends on it."""
    out = {}
    try:
        mode = view.state.get("mouseMode")
        # pyqtgraph's own numbering, which is the opposite way round from the
        # one everybody assumes: RectMode is 1 and PanMode is 3.
        out["vb.mouseMode"] = {1: "Rect", 3: "Pan"}.get(mode, mode)
    except Exception:
        out["vb.mouseMode"] = "?"
    try:
        out["vb.mouseEnabled"] = "{0},{1}".format(
            *[bool(flag) for flag in view.state.get("mouseEnabled", [None, None])])
    except Exception:
        out["vb.mouseEnabled"] = "?"
    try:
        out["vb.bandVisible"] = bool(view.rbScaleBox.isVisible())
    except Exception:
        out["vb.bandVisible"] = "?"
    try:
        (x_low, x_high), (y_low, y_high) = view.viewRange()
        out["vb.x"] = "{0:.6g}..{1:.6g}".format(x_low, x_high)
        out["vb.y"] = "{0:.6g}..{1:.6g}".format(y_low, y_high)
    except Exception:
        pass
    try:
        limits = view.state.get("limits", {})
        x_limit = limits.get("xLimits")
        y_limit = limits.get("yLimits")
        # Only when there is one: a hard bound outliving the lock that set it
        # is a fault this trace has already been used to find once.
        if x_limit and x_limit != [None, None]:
            out["vb.xLimits"] = x_limit
        if y_limit and y_limit != [None, None]:
            out["vb.yLimits"] = y_limit
    except Exception:
        pass
    return out


def scene_state(widget) -> dict:
    """pyqtgraph's own drag bookkeeping, read off the scene.

    The most important half of this trace. These four attributes are what the
    scene uses to decide whether a press begins a drag, which item that drag
    belongs to, and when it is over; a release that never arrives leaves them
    describing a gesture that has finished, and nothing afterwards starts a
    new one.
    """
    out = {}
    try:
        scene = widget.scene()
    except Exception:
        return {"scene": "?"}
    if scene is None:
        return {"scene": "None"}
    try:
        out["scene.clickEvents"] = len(scene.clickEvents)
    except Exception:
        out["scene.clickEvents"] = "?"
    try:
        out["scene.dragButtons"] = list(scene.dragButtons)
    except Exception:
        out["scene.dragButtons"] = "?"
    try:
        item = scene.dragItem
        out["scene.dragItem"] = type(item).__name__ if item is not None else "None"
    except Exception:
        out["scene.dragItem"] = "?"
    try:
        out["scene.lastDrag"] = "yes" if scene.lastDrag is not None else "no"
    except Exception:
        out["scene.lastDrag"] = "?"
    return out


def should_log_move(count: int) -> bool:
    """Whether this move of a drag is one of the ones recorded."""
    if count <= MOVES_IN_FULL:
        return True
    return count % MOVE_SAMPLE == 0
