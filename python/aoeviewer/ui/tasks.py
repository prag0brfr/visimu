# Author: Paulo R. A. Gilz
"""Running core work off the UI thread.

Opening three 500 MB files, or the first plot after doing so, takes seconds. Done
on the GUI thread that is a frozen window: no repaint, no cancel, and on Windows
the "not responding" ghosting that makes people kill the process.

This works because `ctypes.CDLL` releases the GIL for the duration of every call,
so a plain Python thread sitting inside `aoe_preload` genuinely runs in parallel
with the event loop -- the core is meanwhile using all its worker threads inside
that one call.

The important constraint, and it is no longer the obvious one. The core session
*is* internally synchronised: every C ABI entry point takes a reader-writer lock,
so two threads calling it cannot corrupt it. What a lock cannot make safe is the
thing the viewer actually does with the session, which is to hold pointers into
its sample buffers and draw from them. Adding a file, reloading one or writing a
sidecar moves those buffers, and a plot midway through reading one then reads
freed memory -- a lock around each call does not help, because the pointer was
handed out before the call that invalidated it.

So the rule stands: while a task is running the UI does not call the session at
all -- panels are disabled and the tiles show a placeholder. It is now a rule
about the lifetime of what the session hands out rather than about the session
itself, and it is still simpler than the alternative, which would be to copy
every column out before drawing it.
"""

from __future__ import annotations

import inspect
import traceback
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, QThread, Signal

from ..logging_setup import get_logger

_log = get_logger("tasks")


def _wants_report(function: Callable) -> bool:
    """Whether this work function asked to report progress.

    By its signature rather than a flag at the call site, so the two cannot
    disagree: a function written as ``def work(report):`` gets one, and the
    many that are not are called as they always were. A callable whose
    signature cannot be read -- a C function, an odd partial -- is assumed not
    to want one, which is the safe assumption: being called with no argument is
    how every task in this program worked before.
    """
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    return any(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.default is p.empty
        for p in parameters.values()
    )


class Task(QThread):
    """Runs one callable on a worker thread and reports the outcome.

    A task that knows how many steps it has can say so: define the work
    function as ``def work(report):`` and call ``report(done)`` as it goes.
    Anything that does not want to is left alone, and shows an indeterminate
    bar -- which is the honest display for a single parse the core reports
    nothing from the inside of.
    """

    succeeded = Signal(object)
    failed = Signal(str)
    progressed = Signal(int)

    def __init__(self, function: Callable[[], Any], parent: Optional[QObject] = None):
        super().__init__(parent)
        self._function = function

    def run(self) -> None:
        try:
            if _wants_report(self._function):
                # Emitting a signal is how a worker thread is allowed to reach
                # the GUI thread: Qt queues it onto the receiver's event loop
                # rather than touching a widget from here.
                result = self._function(self.progressed.emit)
            else:
                result = self._function()
        except Exception:
            # The traceback is worth keeping: an exception here comes from the
            # core boundary and the line number is the only clue to which call.
            self.failed.emit(traceback.format_exc(limit=6))
            return
        self.succeeded.emit(result)


class TaskRunner(QObject):
    """Serialises background tasks and reports busy state.

    Tasks queue rather than overlap, because they all touch the same
    unsynchronised session. A second request arriving while one is in flight
    replaces any still-pending request -- for redraw work the newest is the only
    one worth doing.
    """

    started = Signal(str, int)  # description, number of steps (0 = unknown)
    finished = Signal()
    failed = Signal(str)
    progressed = Signal(int)    # steps completed

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._task: Optional[Task] = None
        self._pending: Optional[tuple] = None
        self._description = ""
        self._in_callback = False

    @property
    def busy(self) -> bool:
        """True while a task is running *or* its callback is still running.

        The callback counts, and that is the whole point. Completion callbacks
        routinely submit follow-up work -- loading files finishes by asking for
        a redraw -- and if `busy` were false at that moment, submit() would
        start the second task immediately instead of queueing it. Two threads
        would then be inside the unsynchronised core session at once, and
        `finished` would already have been emitted, re-enabling the UI to touch
        it as well. Serialising is the one guarantee this class exists to make.
        """
        return self._in_callback or (self._task is not None
                                     and self._task.isRunning())

    def submit(self, description: str, function: Callable[[], Any],
               on_done: Optional[Callable[[Any], None]] = None,
               steps: int = 0) -> None:
        """Queue some work. `steps` is how many, when that is known.

        A known step count is what turns the status bar's bar from "something
        is happening" into "3 of 7", which for three large files is the
        difference between a wait that looks stuck and one that is visibly
        moving.
        """
        if self.busy:
            # Coalesce: only the most recent queued request matters.
            self._pending = (description, function, on_done, steps)
            return
        self._start(description, function, on_done, steps)

    def _start(self, description: str, function: Callable[[], Any],
               on_done: Optional[Callable[[Any], None]], steps: int = 0) -> None:
        self._description = description
        self._task = Task(function, self)
        self._task.succeeded.connect(lambda result: self._complete(result, on_done))
        self._task.failed.connect(self._on_failed)
        self._task.progressed.connect(self.progressed)
        self.started.emit(description, int(max(0, steps)))
        self._task.start()

    def _complete(self, result: Any, on_done: Optional[Callable[[Any], None]]) -> None:
        # Stay busy across the callback -- see the note on `busy`. Clearing
        # `_task` first is what let a callback's own submit() run a second
        # worker alongside the first.
        self._in_callback = True
        try:
            if on_done is not None:
                on_done(result)
        except Exception:
            # A callback that raises must not stop _drain() running: the UI is
            # disabled while a task is in flight, so skipping it would leave the
            # whole window permanently dead with no clue why.
            message = traceback.format_exc(limit=8)
            _log.exception("task callback failed")
            self.failed.emit(message)
        finally:
            self._in_callback = False
            self._task = None
            self._drain()

    def _on_failed(self, message: str) -> None:
        self._in_callback = True
        _log.error("background task failed:\n%s", message)
        try:
            self.failed.emit(message)
        finally:
            self._in_callback = False
            self._task = None
            self._drain()

    def _drain(self) -> None:
        if self._pending is not None:
            description, function, on_done, steps = self._pending
            self._pending = None
            self._start(description, function, on_done, steps)
        else:
            self.finished.emit()

    def wait_for_idle(self, milliseconds: int = 30000) -> None:
        """Block until nothing is running. Only for shutdown.

        Waits out the whole chain, not just the task in flight: a completion
        callback can queue another one, and destroying the session while any of
        them is inside the core writes into freed memory.
        """
        import time

        deadline = time.monotonic() + milliseconds / 1000.0
        while self.busy or self._pending is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _log.warning("gave up waiting for background work to finish")
                return
            task = self._task
            if task is not None:
                task.wait(int(remaining * 1000))
            else:
                break
