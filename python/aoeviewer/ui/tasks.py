# Author: Paulo R. A. Gilz
"""Running core work off the UI thread.

Opening three 500 MB files, or the first plot after doing so, takes seconds. Done
on the GUI thread that is a frozen window: no repaint, no cancel, and on Windows
the "not responding" ghosting that makes people kill the process.

This works because `ctypes.CDLL` releases the GIL for the duration of every call,
so a plain Python thread sitting inside `aoe_preload` genuinely runs in parallel
with the event loop -- the core is meanwhile using all its worker threads inside
that one call.

The important constraint: **the core session is not internally synchronised.**
Only one thread may touch it at a time. So the rule here is that while a task is
running, the UI does not call the session at all -- panels are disabled and the
tiles show a placeholder. That is simpler and safer than a lock, which would just
move the freeze from the paint loop to the click handler.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, QThread, Signal

from ..logging_setup import get_logger

_log = get_logger("tasks")


class Task(QThread):
    """Runs one callable on a worker thread and reports the outcome."""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, function: Callable[[], Any], parent: Optional[QObject] = None):
        super().__init__(parent)
        self._function = function

    def run(self) -> None:
        try:
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

    started = Signal(str)      # human-readable description
    finished = Signal()
    failed = Signal(str)

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
               on_done: Optional[Callable[[Any], None]] = None) -> None:
        if self.busy:
            # Coalesce: only the most recent queued request matters.
            self._pending = (description, function, on_done)
            return
        self._start(description, function, on_done)

    def _start(self, description: str, function: Callable[[], Any],
               on_done: Optional[Callable[[Any], None]]) -> None:
        self._description = description
        self._task = Task(function, self)
        self._task.succeeded.connect(lambda result: self._complete(result, on_done))
        self._task.failed.connect(self._on_failed)
        self.started.emit(description)
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
            description, function, on_done = self._pending
            self._pending = None
            self._start(description, function, on_done)
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
