#!/usr/bin/env python3
# Author: Paulo R. A. Gilz
"""Build the standalone aoeViewer executable.

    python packaging/build_exe.py             one folder  (recommended)
    python packaging/build_exe.py --onefile   single file
    python packaging/build_exe.py --clean     discard previous build state
    python packaging/build_exe.py --debug     console build, prints tracebacks

Run this on the platform you are targeting: PyInstaller does not cross-compile,
so a Windows .exe must be built on Windows and a Linux binary on Linux.

--onefile vs the default, honestly:

  one folder   starts in well under a second, and is a directory you copy or zip.
  --onefile    is a single file, but every launch unpacks ~60 MB into a temp
               directory first, so it starts in several seconds, and corporate
               antivirus scans that unpack every time.

Unless you specifically need one file, take the folder.

After building, the executable is launched once with --selftest and the result
printed. That catches a bundle that builds cleanly but cannot import what it
needs, which is otherwise only discoverable by double-clicking and reading a
truncated dialog.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SPEC = os.path.join(PROJECT, "packaging", "aoeviewer.spec")


def core_library_name() -> str:
    if sys.platform == "win32":
        return "aoecore.dll"
    if sys.platform == "darwin":
        return "libaoecore.dylib"
    return "libaoecore.so"


def ensure_icon() -> None:
    """Generate packaging/aoeviewer.ico if it is not there.

    It is drawn by a script rather than committed as a binary, so on a fresh
    checkout it does not exist yet. Building an unbranded executable and saying
    nothing would be the wrong way to handle that, and making the user run a
    second command by hand is unnecessary when this one can do it.
    """
    icon = os.path.join(PROJECT, "packaging", "aoeviewer.ico")
    if os.path.isfile(icon):
        return

    script = os.path.join(PROJECT, "tools", "make_icon.py")
    print("Icon missing; generating it with tools/make_icon.py ...")
    result = subprocess.run([sys.executable, script], cwd=PROJECT)
    if result.returncode != 0 or not os.path.isfile(icon):
        raise SystemExit(
            "Could not generate the icon ({0}).\n"
            "Run 'python tools/make_icon.py' and check the error.".format(icon)
        )


def refresh_shell_icon_cache() -> None:
    """Ask Explorer to re-read icons for rebuilt executables (Windows only).

    The icon *is* embedded -- the .exe has RT_ICON and RT_GROUP_ICON resources
    -- but Explorer caches an icon per file path and does not always notice
    that the file changed. The cache is populated the first time a path is
    seen, so an executable first built before the icon existed keeps showing
    the generic one at that path for good, however many times it is rebuilt.

    `ie4uinit.exe -show` is the supported way to tell the shell to rebuild that
    cache. It is a user-level call, it touches nothing else, and if it is not
    present (a stripped install, or a future Windows that renamed it) the build
    carries on -- a stale icon is a cosmetic problem, not a build failure.
    """
    if sys.platform != "win32":
        return
    for arguments in (["ie4uinit.exe", "-show"], ["ie4uinit.exe", "-ClearIconCache"]):
        try:
            subprocess.run(arguments, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=20)
            return
        except (OSError, subprocess.SubprocessError):
            continue


def check_prerequisites() -> None:
    problems = []

    core = os.path.join(PROJECT, "build", core_library_name())
    if not os.path.isfile(core):
        problems.append(
            "The core library is missing ({0}).\n"
            "    Build it first:  {1}".format(
                core, "build.bat" if sys.platform == "win32" else "make"
            )
        )

    for module, hint in (
        ("PyInstaller", "pip install --user PyInstaller"),
        ("PySide6", "pip install --user PySide6-Essentials shiboken6"),
        ("pyqtgraph", "pip install --user pyqtgraph"),
        ("numpy", "pip install --user numpy"),
    ):
        try:
            __import__(module)
        except ImportError:
            problems.append("{0} is not installed.\n    {1}".format(module, hint))

    if problems:
        print("Cannot build yet:\n")
        for problem in problems:
            print("  * " + problem + "\n")
        print(
            "If pip cannot reach the internet, download the wheels by hand and\n"
            "install them offline:\n"
            "    pip install --user --no-index --find-links C:\\wheels --no-deps <name>"
        )
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onefile", action="store_true",
                        help="produce a single executable instead of a folder")
    parser.add_argument("--clean", action="store_true",
                        help="remove previous build artefacts first")
    parser.add_argument("--debug", action="store_true",
                        help="keep a console window so startup errors are readable")
    parser.add_argument(
        "--runtime-tmpdir", metavar="DIR", default=None,
        help="--onefile only: where the bundle unpacks itself at run time. "
             "Use this when the machine forbids running code from %%TEMP%% "
             "(AppLocker), which shows up as import errors from numpy or Qt. "
             "Pass '.' to unpack beside the executable.",
    )
    parser.add_argument("--no-selftest", action="store_true",
                        help="skip launching the result to check it starts")
    args = parser.parse_args()

    check_prerequisites()
    ensure_icon()

    dist = os.path.join(PROJECT, "dist")
    work = os.path.join(PROJECT, "build", "pyinstaller")

    if args.clean:
        for path in (dist, work):
            shutil.rmtree(path, ignore_errors=True)

    env = dict(os.environ)
    env["AOEVIEWER_ONEFILE"] = "1" if args.onefile else "0"
    env["AOEVIEWER_CONSOLE"] = "1" if args.debug else "0"
    env.pop("AOEVIEWER_RUNTIME_TMPDIR", None)
    if args.runtime_tmpdir:
        if args.onefile:
            # Passed through verbatim: PyInstaller expands it on the target
            # machine, so "%LOCALAPPDATA%\\aoeviewer" is a valid value here.
            env["AOEVIEWER_RUNTIME_TMPDIR"] = args.runtime_tmpdir
        else:
            print("Note: --runtime-tmpdir only affects --onefile builds; ignoring it.")

    # --specpath is a makespec option and PyInstaller rejects it outright when a
    # .spec file is given ("makespec options not valid when a .spec file is
    # given"). --distpath and --workpath are fine.
    command = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--distpath", dist,
        "--workpath", work,
        SPEC,
    ]

    print("Building {0} ...".format("single file" if args.onefile else "folder"))
    result = subprocess.run(command, cwd=PROJECT, env=env)
    if result.returncode != 0:
        return result.returncode

    suffix = ".exe" if sys.platform == "win32" else ""
    if args.onefile:
        target = os.path.join(dist, "aoeviewer" + suffix)
    else:
        target = os.path.join(dist, "aoeviewer", "aoeviewer" + suffix)

    print("\nBuilt.")
    print("  Output: {0}".format(target))
    if not args.onefile:
        print("  Ship the whole 'aoeviewer' folder; the executable is inside it.")

    if os.path.isdir(dist):
        total = 0
        for root, _dirs, files in os.walk(dist):
            for name in files:
                total += os.path.getsize(os.path.join(root, name))
        print("  Size:   {0:.0f} MB".format(total / (1024 * 1024)))

    # Explorer caches icons per path; a rebuilt executable at a path that once
    # had no icon keeps showing the generic one until the cache is refreshed.
    refresh_shell_icon_cache()

    if args.no_selftest:
        return 0
    return selftest(target)


def selftest(target: str) -> int:
    """Launch the built executable once and report whether it can start.

    This is the step that would otherwise be a double-click and a dialog box
    with the traceback cut off. --selftest imports everything the GUI needs and
    exits, so a bundle with a missing extension module fails here, loudly, with
    the real error, instead of at the user's desk.
    """
    print("\nSelf-test: {0} --selftest".format(os.path.basename(target)))
    if not os.path.exists(target):
        print("  FAILED: the executable was not produced.")
        return 1

    try:
        result = subprocess.run(
            [target, "--selftest"],
            cwd=os.path.dirname(target) or ".",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("  FAILED: the executable did not exit within 5 minutes.")
        return 1
    except OSError as exc:
        print("  FAILED: could not launch it: {0}".format(exc))
        return 1

    output = result.stdout.decode("utf-8", "replace").rstrip()
    for line in output.splitlines():
        print("  | " + line)

    if result.returncode == 0:
        print("\n  OK - the bundle starts and every dependency imports.")
        return 0

    print("\n  FAILED (exit code {0}).".format(result.returncode))
    # A windowed build has no stdout at all, so the log is the only record.
    print_log_tail(os.path.dirname(target) or ".")
    return 1


def print_log_tail(directory: str, lines: int = 60) -> None:
    path = os.path.join(directory, "aoeviewer.log")
    if not os.path.isfile(path):
        path = os.path.join(os.path.expanduser("~"), ".aoeviewer", "aoeviewer.log")
    if not os.path.isfile(path):
        print("  No aoeviewer.log was written either.")
        return
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            tail = handle.read().splitlines()[-lines:]
    except OSError as exc:
        print("  Could not read {0}: {1}".format(path, exc))
        return
    print("\n  Last {0} lines of {1}:".format(len(tail), path))
    for line in tail:
        print("  > " + line)


if __name__ == "__main__":
    sys.exit(main())
