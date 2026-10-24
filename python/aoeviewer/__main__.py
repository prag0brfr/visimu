# Author: Paulo R. A. Gilz
"""Entry point: ``python -m aoeviewer`` and the frozen executable."""

from __future__ import annotations

import argparse
import os
import sys
import traceback


def _bootstrap_path() -> None:
    """Make the package importable when run from a source checkout.

    Running ``python python/aoeviewer/__main__.py`` directly puts the *module's*
    directory on sys.path rather than the package's parent, so the package
    itself is not importable. Frozen builds do not need this.
    """
    if getattr(sys, "frozen", False):
        return
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    if parent not in sys.path:
        sys.path.insert(0, parent)


def _emit(text: str) -> None:
    """Write a line of diagnostics wherever it can actually be seen.

    A windowed PyInstaller build has ``sys.stdout is None``, so a bare print()
    raises -- which is exactly how a diagnostic tool ends up producing no
    diagnostics. Try the normal stream, fall back to the raw file descriptor
    (present when the parent handed us a pipe), and give up quietly if neither
    works; the log file has the same content regardless.
    """
    for stream in (sys.stdout, sys.__stdout__, sys.stderr, sys.__stderr__):
        try:
            if stream is not None:
                stream.write(text + "\n")
                stream.flush()
                return
        except Exception:
            continue
    try:
        os.write(1, (text + "\n").encode("utf-8", "replace"))
    except Exception:
        pass


def selftest() -> int:
    """Import everything the GUI needs and report on each piece.

    The point is to turn "Importing the numpy C-extensions failed" -- a dialog
    that truncates the one line naming the missing DLL -- into a full report in
    the log and on stdout. Each import is attempted separately so one failure
    does not hide the state of the rest.
    """
    from aoeviewer.logging_setup import get_logger, log_path

    log = get_logger("selftest")
    failures = 0

    _emit("aoeViewer self-test")
    _emit("  frozen      : {0}".format(bool(getattr(sys, "frozen", False))))
    _emit("  executable  : {0}".format(sys.executable))
    _emit("  bundle dir  : {0}".format(getattr(sys, "_MEIPASS", "(not frozen)")))
    _emit("  python      : {0}".format(sys.version.replace("\n", " ")))
    _emit("  log file    : {0}".format(log_path() or "(none)"))

    checks = [
        ("numpy", lambda: __import__("numpy").__version__),
        ("PySide6", lambda: __import__("PySide6").__version__),
        ("PySide6.QtWidgets", lambda: "ok"),
        ("pyqtgraph", lambda: __import__("pyqtgraph").__version__),
    ]

    for name, describe in checks:
        try:
            __import__(name)
            detail = describe()
            _emit("  OK    {0:<20} {1}".format(name, detail))
            log.info("selftest ok: %s %s", name, detail)
        except Exception:
            failures += 1
            report = traceback.format_exc()
            _emit("  FAIL  {0}".format(name))
            for line in report.rstrip().splitlines():
                _emit("        " + line)
            log.error("selftest failed: %s\n%s", name, report)

    # The core library is separate: it is loaded by ctypes, not by import, so
    # it fails in an entirely different way (a missing dependent DLL rather
    # than a missing module).
    try:
        from aoeviewer import _core

        _emit("  OK    {0:<20} {1} from {2}".format(
            "core", _core.VERSION, _core.library_path()))
        log.info("selftest ok: core %s", _core.VERSION)
    except Exception:
        failures += 1
        report = traceback.format_exc()
        _emit("  FAIL  core")
        for line in report.rstrip().splitlines():
            _emit("        " + line)
        log.error("selftest failed: core\n%s", report)

    # Constructing a QApplication is the last thing that can fail without any
    # of the above being wrong -- a missing Qt platform plugin, typically.
    if failures == 0:
        try:
            from PySide6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is None:
                try:
                    app = QApplication([sys.argv[0]])
                except Exception:
                    # No display (a headless build machine, or SSH without X).
                    # That is not a packaging fault, so fall back rather than
                    # reporting a failure the user cannot act on.
                    app = QApplication([sys.argv[0], "-platform", "offscreen"])
            _emit("  OK    {0:<20} {1}".format("QApplication", "created"))
            del app
        except Exception:
            failures += 1
            report = traceback.format_exc()
            _emit("  FAIL  QApplication")
            for line in report.rstrip().splitlines():
                _emit("        " + line)
            log.error("selftest failed: QApplication\n%s", report)

    _emit("")
    _emit("{0} check(s) failed.".format(failures) if failures else "All checks passed.")
    return 1 if failures else 0


def main(argv=None) -> int:
    _bootstrap_path()

    parser = argparse.ArgumentParser(
        prog="aoeviewer", description="Viewer for AOE simulation result files."
    )
    parser.add_argument("files", nargs="*", help="result files to open at startup")
    parser.add_argument("--version", action="store_true", help="print versions and exit")
    parser.add_argument(
        "--selftest", action="store_true",
        help="check that every dependency loads, report, and exit. Run this "
             "first when the application will not start.",
    )
    parser.add_argument(
        "--build-cache", action="store_true",
        help="write a .aoec sidecar for each named file and exit, without "
             "opening a window. This is how the application builds sidecars "
             "for large files in the background; it is also useful on its own, "
             "to warm a directory of results before anyone opens them.",
    )
    parser.add_argument(
        "--memory-limit", type=float, default=2048.0, metavar="MB",
        help="cap on cached sample data (default 2048). Columns beyond this are "
             "dropped least-recently-used first and re-read on demand.",
    )
    args = parser.parse_args(argv)

    # Set up logging before anything else can fail, so a failure during import
    # or startup still leaves a file behind.
    from aoeviewer.logging_setup import log_path, setup as setup_logging

    setup_logging()

    if args.selftest:
        return selftest()

    # Before --version and before the window: this path must not import Qt,
    # both because it has no use for it and because a machine that cannot
    # start a GUI can still perfectly well write a sidecar.
    if args.build_cache:
        from aoeviewer.cachebuild import main as build_caches

        return build_caches([os.path.abspath(p) for p in args.files])

    if args.version:
        from aoeviewer import __version__
        from aoeviewer import _core

        _emit("aoeviewer   {0}".format(__version__))
        _emit("core        {0}".format(_core.VERSION))
        _emit("core lib    {0}".format(_core.library_path()))
        _emit("workers     {0}".format(_core.WORKER_COUNT))
        _emit("log         {0}".format(log_path() or "(none)"))
        return 0

    try:
        return _run_gui(args)
    except Exception:
        return _report_startup_failure(log_path())


def _run_gui(args) -> int:
    # Imported here, not at module scope: --version, --help and --selftest
    # should work even if Qt is missing or the display is unavailable.
    from PySide6.QtWidgets import QApplication

    from aoeviewer.ui.main_window import MainWindow

    QApplication.setApplicationName("aoeViewer")
    QApplication.setOrganizationName("aoeViewer")

    app = QApplication(sys.argv)
    _apply_window_icon(app)

    from aoeviewer.logging_setup import install_qt_handler

    install_qt_handler()
    window = MainWindow(initial_files=[os.path.abspath(p) for p in args.files])
    window.memory_limit_bytes = int(max(64.0, args.memory_limit) * 1024 * 1024)
    window.show()
    return app.exec()


def _apply_window_icon(app) -> None:
    """Give the running window the application icon.

    The icon compiled into the .exe is what Explorer and a pinned shortcut
    show; it does not reach the window itself. Without this the running
    application has Qt's default icon in its title bar and on the taskbar
    button, which is how a branded executable still ends up looking unbranded
    the moment you launch it.

    Best-effort by design: a missing icon is a cosmetic problem and must never
    stop the application starting.
    """
    try:
        from PySide6.QtGui import QIcon

        for path in _icon_candidates():
            if path and os.path.isfile(path):
                icon = QIcon(path)
                if not icon.isNull():
                    app.setWindowIcon(icon)
                    return
    except Exception:
        pass


def _icon_candidates():
    """Where the .ico might be, frozen or from a checkout."""
    name = "aoeviewer.ico"
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        yield os.path.join(bundle, name)
        yield os.path.join(os.path.dirname(sys.executable), name)

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    yield os.path.join(root, "packaging", name)


def _report_startup_failure(path) -> int:
    """Record a failed start, and show it if there is anything to show it with.

    Without this a frozen build dies to a dialog that truncates the traceback
    at the point where it stops being useful. The log always gets the whole
    thing; the message box is best-effort, because the reason we are here may
    well be that Qt itself is what failed.
    """
    report = traceback.format_exc()

    try:
        from aoeviewer.logging_setup import get_logger

        get_logger("startup").critical("startup failed\n%s", report)
    except Exception:
        pass

    _emit("aoeViewer could not start.\n")
    _emit(report)
    if path:
        _emit("Full details written to: {0}".format(path))
    _emit("\nRun 'aoeviewer --selftest' for a component-by-component report.")

    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance() or QApplication(sys.argv)
        box = QMessageBox()
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("aoeViewer could not start")
        box.setText("aoeViewer could not start.")
        box.setInformativeText(
            "Details were written to:\n{0}".format(path or "(no log file)")
        )
        box.setDetailedText(report)
        box.exec()
        del app
    except Exception:
        pass

    return 1


if __name__ == "__main__":
    sys.exit(main())
