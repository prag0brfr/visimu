# Author: Paulo R. A. Gilz
"""Crash and error logging.

Anything that goes wrong should leave a file behind. A GUI has nowhere to print
a traceback -- a frozen build has no console at all -- so without this a failure
is a window that closes, or a button that silently does nothing, and there is
nothing to send back to whoever has to fix it.

Three sources are captured:
  * `sys.excepthook`, for anything unhandled on the main thread;
  * `threading.excepthook`, for the background task threads;
  * a Qt message handler, for the warnings Qt itself emits (which is where
    layout and painting problems show up first).

The log lives beside the application, or in the user's home if that is not
writable -- a frozen executable often sits in a read-only directory.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import platform
import sys
import threading
import traceback
from typing import Optional

LOGGER_NAME = "visimu"
_log_path: Optional[str] = None


def log_path() -> Optional[str]:
    """Where the log is being written, or None if logging is not set up."""
    return _log_path


def _choose_directory() -> str:
    candidates = []

    override = os.environ.get("VISIMU_LOG_DIR")
    if override:
        candidates.append(override)

    if getattr(sys, "frozen", False):
        candidates.append(os.path.dirname(sys.executable))
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.abspath(os.path.join(here, "..", "..")))

    candidates.append(os.path.join(os.path.expanduser("~"), ".visimu"))

    for directory in candidates:
        try:
            os.makedirs(directory, exist_ok=True)
            probe = os.path.join(directory, ".write-test")
            with open(probe, "w") as handle:
                handle.write("")
            os.remove(probe)
            return directory
        except OSError:
            continue

    # Last resort: the working directory. If even that fails, setup() gives up
    # rather than preventing the application from starting.
    return os.getcwd()


def setup(level: int = logging.INFO) -> Optional[str]:
    """Install file logging and the exception hooks. Returns the log path."""
    global _log_path

    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return _log_path

    logger.setLevel(logging.DEBUG)

    directory = _choose_directory()
    path = os.path.join(directory, "visimu.log")

    try:
        # Rotating, so a long-running session or a repeating error cannot fill
        # the disk -- which on a locked-down machine is a support call.
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
    except OSError:
        return None

    handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s  %(message)s")
    )
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    # A windowed frozen build has no stderr at all (sys.stderr is None), and a
    # StreamHandler wrapped around it fails on every single record. The file
    # handler above is the one that matters; this is only for a console run.
    if sys.stderr is not None:
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(level)
        console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(console)

    _log_path = path
    _install_hooks(logger)
    _log_environment(logger)
    return path


def _log_environment(logger: logging.Logger) -> None:
    """Record what this is running on. Half of any bug report is the platform."""
    from . import __version__

    logger.info("=" * 68)
    logger.info("visimu %s starting", __version__)
    logger.info("python   %s", sys.version.replace("\n", " "))
    logger.info("platform %s %s", platform.platform(), platform.machine())

    try:
        from . import _core
        logger.info("core     %s from %s", _core.VERSION, _core.library_path())
        logger.info("workers  %s", _core.WORKER_COUNT)
    except Exception:
        logger.exception("core library could not be described")

    try:
        import PySide6
        import pyqtgraph
        import numpy
        logger.info("PySide6 %s, pyqtgraph %s, numpy %s",
                    PySide6.__version__, pyqtgraph.__version__, numpy.__version__)
    except Exception:
        logger.warning("GUI libraries could not be described", exc_info=True)


def _install_hooks(logger: logging.Logger) -> None:
    previous_hook = sys.excepthook

    def excepthook(kind, value, tb):
        # KeyboardInterrupt is a deliberate quit, not a fault.
        if issubclass(kind, KeyboardInterrupt):
            previous_hook(kind, value, tb)
            return
        logger.critical(
            "unhandled exception\n%s", "".join(traceback.format_exception(kind, value, tb))
        )
        previous_hook(kind, value, tb)

    sys.excepthook = excepthook

    if hasattr(threading, "excepthook"):
        def thread_hook(args):
            if issubclass(args.exc_type, SystemExit):
                return
            logger.critical(
                "unhandled exception in thread %s\n%s",
                getattr(args.thread, "name", "?"),
                "".join(
                    traceback.format_exception(
                        args.exc_type, args.exc_value, args.exc_traceback
                    )
                ),
            )

        threading.excepthook = thread_hook


def install_qt_handler() -> None:
    """Route Qt's own warnings into the log. Call after QApplication exists."""
    try:
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler
    except ImportError:
        return

    logger = logging.getLogger(LOGGER_NAME + ".qt")
    levels = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def handler(mode, context, message):
        logger.log(levels.get(mode, logging.INFO), "%s", message)

    qInstallMessageHandler(handler)


def get_logger(suffix: str = "") -> logging.Logger:
    return logging.getLogger(LOGGER_NAME + ("." + suffix if suffix else ""))
