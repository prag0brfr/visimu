# Author: Paulo R. A. Gilz
"""Catch attributes used before they are assigned, without running Qt.

The GUI cannot be exercised in every environment this is developed in -- PySide6
is not always installable -- so a whole family of mistakes has no test to catch
it. This one is worth catching statically because it is silent until the moment
the window is built, and then it is fatal:

    self.text_panel.changed.connect(self.grid.refresh_texts)   # line 285
    ...
    self.grid = PlotGrid(...)                                  # line 287

`self.grid.refresh_texts` is looked up *now*, not when the signal fires, so the
window dies during construction with "'MainWindow' object has no attribute
'grid'". It reads perfectly well; nothing but running it, or this, finds it.

The rule: reading `self.X` before assigning it, in the only method of the class
that ever assigns it. The "only method" part is what makes the check usable --
`self._curves` is read and then reassigned all over the place, and that is
correct, because `__init__` set it first. An attribute assigned nowhere but
below the line that reads it has genuinely not been created yet.

Reads inside a nested function or lambda are deferred, so they are exempt, as is
anything guarded by hasattr/getattr, and so are class-level defaults.

Parsing rather than importing is the point: this must work with no Qt present.
"""

from __future__ import annotations

import ast
import io
import os
import re
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PACKAGE = os.path.join(_ROOT, "python", "visimu")


def _python_files():
    for directory, _dirs, names in os.walk(_PACKAGE):
        if "__pycache__" in directory:
            continue
        for name in sorted(names):
            if name.endswith(".py"):
                yield os.path.join(directory, name)


class _MethodScan(ast.NodeVisitor):
    """Records, for one method body, where each `self.X` is assigned and read.

    Nested functions and lambdas are not descended into: a body that runs later
    may legitimately refer to anything the constructor sets up afterwards.
    """

    def __init__(self, self_name: str):
        self._self = self_name
        self.assigned = {}   # name -> first line assigned
        self.read = []       # (name, line)
        self._guarded = set()

    # Do not descend: these run later.
    def visit_FunctionDef(self, node):
        return

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        return

    def visit_Call(self, node):
        # hasattr(self, "grid") / getattr(self, "grid", None) are explicit
        # guards, so whatever they protect is intentional.
        if (isinstance(node.func, ast.Name)
                and node.func.id in ("hasattr", "getattr")
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == self._self
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)):
            self._guarded.add(node.args[1].value)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == self._self:
            if isinstance(node.ctx, ast.Store):
                self.assigned.setdefault(node.attr, node.lineno)
            elif isinstance(node.ctx, ast.Load):
                self.read.append((node.attr, node.lineno))
        self.generic_visit(node)


def _class_attribute_names(class_node: ast.ClassDef):
    """Names bound by class-level statements: `x: int = 3`, `x = 3`, slots."""
    names = set()
    for statement in class_node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            names.add(statement.target.id)
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _findings(path: str):
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)

    problems = []
    for class_node in ast.walk(tree):
        if not isinstance(class_node, ast.ClassDef):
            continue

        methods = [m for m in class_node.body
                   if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and m.args.args]

        # Every self.X assignment anywhere in the class, by method. Includes
        # assignments inside nested bodies, since a callback that sets an
        # attribute still creates it.
        assigned_by = {}
        for method in methods:
            self_name = method.args.args[0].arg
            for node in ast.walk(method):
                if (isinstance(node, ast.Attribute)
                        and isinstance(node.ctx, ast.Store)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == self_name):
                    assigned_by.setdefault(node.attr, set()).add(method.name)

        class_level = _class_attribute_names(class_node)

        for method in methods:
            scan = _MethodScan(method.args.args[0].arg)
            for statement in method.body:
                scan.visit(statement)

            for name, line in scan.read:
                first_assigned = scan.assigned.get(name)
                if first_assigned is None or name in scan._guarded:
                    continue
                if line >= first_assigned:
                    continue
                # Assigned elsewhere as well -- __init__, a class-level
                # default, another method -- so by the time this runs it
                # exists, and reading before reassigning is deliberate.
                if name in class_level:
                    continue
                if assigned_by.get(name, set()) - {method.name}:
                    continue
                problems.append(
                    "{0}:{1}  {2}.{3}() reads self.{4} before assigning it "
                    "on line {5}".format(
                        os.path.relpath(path, _ROOT), line,
                        class_node.name, method.name, name, first_assigned)
                )
    return problems


class TestConstructionOrder(unittest.TestCase):
    def test_no_attribute_is_used_before_it_is_assigned(self):
        problems = []
        for path in _python_files():
            problems.extend(_findings(path))
        self.assertEqual(
            problems, [],
            "Attributes read before they are assigned:\n  " + "\n  ".join(problems)
        )


class TestRedrawWiring(unittest.TestCase):
    """The defaults have to be committed somewhere, and only one place does it.

    Panels display a default x axis but no longer write it to the tile -- a
    refresh is not a user action. That makes AppState.apply_parameter_defaults
    the single place a tile actually acquires one, and it is only reached if
    the redraw path calls it. Nothing else would notice its absence until a
    tile said "no x axis selected" beside a panel showing TIME.
    """

    def _method_source(self, path: str, class_name: str, method_name: str) -> str:
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        tree = ast.parse(source, filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for method in node.body:
                    if (isinstance(method, ast.FunctionDef)
                            and method.name == method_name):
                        return ast.get_source_segment(source, method) or ""
        self.fail("{0}.{1}() not found in {2}".format(class_name, method_name, path))

    def test_the_redraw_path_commits_tile_defaults(self):
        body = self._method_source(
            os.path.join(_PACKAGE, "ui", "main_window.py"),
            "MainWindow", "refresh_plots")
        self.assertIn("apply_parameter_defaults", body)

    def test_the_file_change_path_resolves_missing_parameters(self):
        body = self._method_source(
            os.path.join(_PACKAGE, "ui", "main_window.py"),
            "MainWindow", "refresh_all")
        self.assertIn("resolve_missing_parameters", body)

    def test_panel_refreshes_do_not_assign_the_axis_parameters(self):
        """AOE-14: a refresh displays, it does not decide.

        Panels refresh only when their tab is shown, so anything they assign
        makes the same click leave different state behind depending on where
        the user last clicked.
        """
        path = os.path.join(_PACKAGE, "ui", "panels.py")
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)

        offenders = []
        watched = {"x_param", "z_param", "color_iso_param", "y_params"}
        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            for method in cls.body:
                if not isinstance(method, ast.FunctionDef):
                    continue
                if method.name not in ("refresh", "_refresh_editor"):
                    continue
                for node in ast.walk(method):
                    if (isinstance(node, ast.Attribute)
                            and isinstance(node.ctx, ast.Store)
                            and node.attr in watched):
                        offenders.append("{0}.{1}() line {2} assigns .{3}".format(
                            cls.name, method.name, node.lineno, node.attr))
        self.assertEqual(offenders, [],
                         "Panel refreshes must not edit the tile:\n  "
                         + "\n  ".join(offenders))


class TestInspectWiring(unittest.TestCase):
    """Clicking a curve in Inspect mode must bring its panel forward.

    The panel is shown when the mode is chosen, but the user is free to go and
    look at something else afterwards -- and then a click read a whole run,
    filled in a table, and left it on a tab nobody was watching.
    """

    def _source(self, class_name: str, method_name: str) -> str:
        path = os.path.join(_PACKAGE, "ui", "main_window.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        tree = ast.parse(source, filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for method in node.body:
                    if (isinstance(method, ast.FunctionDef)
                            and method.name == method_name):
                        return ast.get_source_segment(source, method) or ""
        self.fail("{0}.{1}() not found".format(class_name, method_name))

    def test_a_clicked_point_shows_the_panel(self):
        self.assertIn("_show_panel",
                      self._source("MainWindow", "_on_point_inspected"))

    def test_a_measured_slope_shows_the_panel(self):
        self.assertIn("_show_panel",
                      self._source("MainWindow", "_on_slope_measured"))

    def test_choosing_the_mode_shows_the_panel(self):
        self.assertIn("_show_panel", self._source("MainWindow", "set_mouse_mode"))


class TestTheOldNameIsGone(unittest.TestCase):
    """The application is called visimu, and nothing is called the old name.

    Spelled in halves throughout, so that this file -- which has to name the
    thing it is forbidding -- is not itself the one occurrence that makes the
    test fail.

    What is deliberately *not* renamed: the AOE result format and its
    extensions (.aoe, .aoec, .aoews), the core library (aoecore), and the
    AOE_CXX / AOE_PYTHON build variables. Those name the data format this
    reads, which belongs to the files and not to the program.
    """

    OLD = "aoe" + "viewer"

    SKIP_DIRS = {".git", "build", "dist", "__pycache__", ".pytest_cache"}
    SKIP_EXT = {".pdf", ".aux", ".toc", ".out", ".log", ".ico", ".png",
                ".pyc", ".so", ".dll", ".dylib", ".exe",
                ".aoe", ".aoec", ".aoews"}

    def files(self, readable_only=False):
        """Every file in the tree, build output aside.

        readable_only drops the ones there is no point reading -- a PDF, an
        icon, a compiled library. Their *names* are still checked: a document
        called after the old application is exactly the kind of leftover this
        is for.
        """
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for folder, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS]
            for name in sorted(names):
                if (readable_only
                        and os.path.splitext(name)[1].lower() in self.SKIP_EXT):
                    continue
                yield os.path.join(root, folder, name)

    def test_no_file_is_named_after_it(self):
        for path in self.files():
            self.assertNotIn(self.OLD, os.path.basename(path).lower(),
                             "%s is still named after the old application"
                             % os.path.basename(path))

    def test_no_file_mentions_it(self):
        here = os.path.abspath(__file__)
        offenders = []
        for path in self.files(readable_only=True):
            if os.path.abspath(path) == here:
                continue            # this file names it in order to forbid it
            try:
                body = io.open(path, encoding="utf-8", newline="").read()
            except (UnicodeDecodeError, OSError):
                continue
            if self.OLD in body.lower():
                offenders.append(os.path.relpath(path, os.path.dirname(here)))
        self.assertEqual(offenders, [],
                         "the old name is still written in: %s"
                         % ", ".join(sorted(offenders)))

    def test_the_package_is_importable_under_the_new_name(self):
        import visimu

        self.assertTrue(visimu.__version__)
        self.assertNotIn(self.OLD, visimu.__name__.lower())


class TestEveryQtNameIsImported(unittest.TestCase):
    """A name used but never imported is a NameError on the user's machine.

    The suite cannot see it: the Qt stubs hand back a class for any attribute
    asked of them, so an un-imported QToolButton resolves to *something* in
    every test and raises only in a real build. That is exactly what happened
    -- the PARAMS panel's move buttons shipped with no import and the
    executable died on the panel's first line.

    So the modules are read rather than run: every Qt-looking name they use
    has to be bound at module level, by an import or a definition.
    """

    def modules(self):
        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "python", "visimu")
        for folder, _dirs, files in os.walk(root):
            if "__pycache__" in folder:
                continue
            for name in sorted(files):
                if name.endswith(".py"):
                    yield os.path.join(folder, name)

    @staticmethod
    def bound_names(tree):
        """Everything the module binds at its top level."""
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    names.add(alias.asname or alias.name)
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    names.add(node.target.id)
        return names

    def test_no_qt_name_is_used_without_being_imported(self):
        pattern = re.compile(r"^Q[A-Z]\w+$")
        for path in self.modules():
            source = io.open(path, encoding="utf-8").read()
            tree = ast.parse(source)
            bound = self.bound_names(tree)
            used = {node.id for node in ast.walk(tree)
                    if isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and pattern.match(node.id)}
            missing = sorted(used - bound)
            self.assertEqual(missing, [],
                             "%s uses %s without importing it"
                             % (os.path.basename(path), ", ".join(missing)))

    def test_the_other_names_the_ui_leans_on_are_imported_too(self):
        """Qt, Signal, pg and np: the same trap, without the Q prefix."""
        for path in self.modules():
            source = io.open(path, encoding="utf-8").read()
            tree = ast.parse(source)
            bound = self.bound_names(tree)
            used = {node.id for node in ast.walk(tree)
                    if isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id in ("Qt", "Signal", "pg", "np")}
            missing = sorted(used - bound)
            self.assertEqual(missing, [],
                             "%s uses %s without importing it"
                             % (os.path.basename(path), ", ".join(missing)))


class TestTheToolbarGroups(unittest.TestCase):
    """One bar, four groups, left to right: the workspace, Export, Mouse,
    Interface -- and Interface against the right edge.

    The order is read off the source because a toolbar is built once, in one
    method, and the order of the calls *is* the order on screen.
    """

    def bar(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "python", "visimu", "ui", "main_window.py")
        body = io.open(path, encoding="utf-8").read()
        start = body.index("def _build_toolbar")
        return body[start:body.index("\n    def ", start + 10)]

    def test_the_groups_are_in_order(self):
        bar = self.bar()
        places = [
            ('add("New workspace"', "New"),
            ('add("Load workspace"', "Load"),
            ("self.save_action", "Save"),
            ("self.save_as_action", "Save as"),
            ('QLabel(" Export ")', "the Export label"),
            ('add("Image"', "Image"),
            ('add("CSV"', "CSV"),
            ('QLabel(" Mouse ")', "the Mouse label"),
            ("self.mode_group = QActionGroup", "the mouse modes"),
            ('add(\n            "Help"', "Help"),
            ('QLabel(" Interface ")', "the Interface label"),
            ('add("Theme"', "Theme"),
            ('add(\n            "Smaller"', "Smaller"),
            ('add(\n            "Larger"', "Larger"),
        ]
        last = -1
        for needle, name in places:
            where = bar.find(needle)
            self.assertGreater(where, -1, "%s is not on the bar" % name)
            self.assertGreater(where, last, "%s is out of order" % name)
            last = where

    def test_the_workspace_group_carries_no_label(self):
        """It is the first thing on the bar; a caption on it would be the one
        label with nothing to tell apart from."""
        bar = self.bar()
        before = bar[:bar.index('add("New workspace"')]
        self.assertNotIn("QLabel", before)

    def test_the_interface_group_is_flush_right(self):
        """The stretch goes in after the mouse modes, so everything from the
        Interface label on is against the right edge."""
        bar = self.bar()
        self.assertLess(bar.index("self.mode_actions[MouseMode.SELECT]"),
                        bar.index("Policy.Expanding"))
        self.assertLess(bar.index("Policy.Expanding"),
                        bar.index('QLabel(" Interface ")'))

    def test_help_sits_just_left_of_the_interface_group(self):
        """Behind its own rule, and not under the Interface caption: it
        belongs to no group."""
        bar = self.bar()
        self.assertLess(bar.index('"Help"'), bar.index('QLabel(" Interface ")'))
        self.assertGreater(bar.index('"Help"'), bar.index("Policy.Expanding"))
        between = bar[bar.index('"Help"'):bar.index('QLabel(" Interface ")')]
        self.assertIn("addSeparator()", between)

    def test_the_old_interface_label_is_not_still_in_the_middle(self):
        bar = self.bar()
        self.assertEqual(bar.count('QLabel(" Interface ")'), 1)
        self.assertEqual(bar.count("Policy.Expanding"), 1)


class TestNewWorkspace(unittest.TestCase):
    """New, Load and Close all offer to save first, through one prompt.

    One prompt because the question is the same one, and because Cancel has to
    mean Cancel in all three: the thing the user was doing does not happen.
    """

    def source(self, method):
        """One method's text. Handles the last method in the file, which has
        no next `def` to stop at."""
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "python", "visimu", "ui", "main_window.py")
        body = io.open(path, encoding="utf-8").read()
        start = body.index("def " + method)
        end = body.find("\n    def ", start + 10)
        return body[start:end if end > 0 else len(body)]

    @staticmethod
    def code(body):
        """The method without its docstring, so a sentence about the code is
        not mistaken for the code."""
        parts = body.split('"""')
        return parts[0] + "".join(parts[2:]) if len(parts) > 2 else body

    def test_all_three_ask_before_discarding(self):
        for method in ("new_workspace", "load_workspace_dialog", "closeEvent"):
            self.assertIn("_may_discard_workspace", self.source(method),
                          "%s does not offer to save" % method)

    def test_cancel_stops_the_thing_that_asked(self):
        after = self.code(self.source("new_workspace")).split(
            "_may_discard_workspace")[1]
        self.assertIn("return", after[:80])
        self.assertIn("event.ignore()", self.code(self.source("closeEvent")))

    def test_a_cancelled_save_as_counts_as_cancel(self):
        """Answering Yes and then cancelling the file dialog leaves the work
        unsaved; carrying on would throw away what the user just tried to
        keep."""
        body = self.code(self.source("_may_discard_workspace"))
        self.assertIn("if self.state.dirty:",
                      body.split("self.save_workspace()")[1])

    def test_a_clean_workspace_is_not_asked_about(self):
        body = self.source("_may_discard_workspace")
        self.assertIn("if not self.state.dirty:", body)
        self.assertIn("return True", body)

    def test_new_starts_from_a_default_state_and_rebinds_it(self):
        """The panels and tiles each hold a reference to the state object, so
        a new one has to reach all of them in the same turn."""
        body = self.source("new_workspace")
        self.assertIn("self.state = AppState()", body)
        self.assertIn("self._rebind_state()", body)
        self.assertIn("self.refresh_all()", body)
        self.assertIn("self._update_window_title()", body)

    def test_new_closes_the_files_too(self):
        """A workspace names the files it was built over, so a blank one names
        none."""
        body = self.source("new_workspace")
        self.assertIn("self.session.clear()", body)

    def test_new_waits_for_a_preload_before_clearing(self):
        """session.clear() drops the datasets a running preload is reading."""
        body = self.code(self.source("new_workspace"))
        self.assertLess(body.index("self.tasks.wait_for_idle()"),
                        body.index("self.session.clear()"))

    def test_new_has_a_shortcut_and_an_icon(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "python", "visimu", "ui", "main_window.py")
        body = io.open(path, encoding="utf-8").read()
        self.assertIn('add("New workspace", self.new_workspace, "new", "Ctrl+N"',
                      body)
        icons = io.open(os.path.join(here, "python", "visimu", "ui",
                                     "icons.py"), encoding="utf-8").read()
        self.assertIn('"new": _draw_new,', icons)


class TestTheBuildScripts(unittest.TestCase):
    """One command builds the core and packages the executable.

    The two halves have to happen in that order and in one run: the
    executable embeds the Python front end but loads the core library at run
    time, so packaging without building first ships yesterday's core inside
    today's bundle.
    """

    def read(self, name):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with io.open(os.path.join(root, name), encoding="utf-8",
                     newline="") as handle:
            return handle.read()

    def test_the_shell_script_has_unix_line_endings(self):
        """A CR at the end of every line makes /bin/sh fail on the first
        compound statement, reporting a syntax error in code that is correct.
        It is the one file in this tree that cannot be CRLF, which is exactly
        why it is easy to convert by accident.
        """
        body = self.read("build.sh")
        self.assertNotIn("\r", body, "build.sh must keep LF line endings")

    def test_the_batch_file_has_windows_line_endings(self):
        self.assertIn("\r\n", self.read("build.bat"))

    def test_both_build_everything_by_default(self):
        """No argument means the core and the single-file executable: that is
        the job, and it was two commands."""
        shell = self.read("build.sh")
        self.assertIn("BUILD_EXE=1", shell)
        self.assertIn('EXE_ARGS="--onefile"', shell)
        self.assertIn("build_exe.py", shell)

        batch = self.read("build.bat")
        self.assertIn("set BUILD_EXE=1", batch)
        self.assertIn("set EXE_MODE=--onefile", batch)
        self.assertIn("build_exe.py", batch)

    def test_both_can_still_build_the_core_alone(self):
        """Packaging takes a minute against a couple of seconds for the core,
        so building it to run the C++ tests against must stay cheap."""
        shell = self.read("build.sh")
        self.assertIn("core|lib|dll|so)", shell)
        self.assertIn("BUILD_EXE=0", shell)

        batch = self.read("build.bat")
        self.assertIn('if /I "%1"=="core" set BUILD_EXE=0', batch)

    def test_the_python_tests_do_not_package_an_executable(self):
        """They need the library, not a bundle."""
        shell = self.read("build.sh")
        start = shell.index("    pytest)")
        block = shell[start:shell.index(";;", start)]
        self.assertIn('"$0" core', block)

    def test_packaging_comes_after_the_core_in_both(self):
        """The order that matters, read as the order of the two commands in
        the file: the compiler invocation, then build_exe.py."""
        for name, compiler in (("build.sh", "$CXX\" $CXXFLAGS -DAOE_BUILD_SHARED"),
                               ("build.bat", "-DAOE_BUILD_SHARED")):
            body = self.read(name)
            self.assertLess(body.index(compiler), body.index("build_exe.py --onefile")
                            if "build_exe.py --onefile" in body
                            else body.rindex("build_exe.py"), name)

    def test_packaging_is_a_branch_and_not_the_fall_through(self):
        """So that `core` really does skip it rather than running it with no
        arguments."""
        self.assertIn('if [ "$BUILD_EXE" -eq 1 ]; then', self.read("build.sh"))
        self.assertIn("if !BUILD_EXE! EQU 1 goto :package",
                      self.read("build.bat"))

    def test_the_folder_variant_passes_no_onefile(self):
        shell = self.read("build.sh")
        start = shell.index("    exe-folder|exedir|folder)")
        self.assertIn('EXE_ARGS=""', shell[start:shell.index(";;", start)])

        batch = self.read("build.bat")
        self.assertIn('if /I "%1"=="exe-folder" set EXE_MODE=', batch)

    def test_the_old_two_step_advice_is_gone_from_the_scripts(self):
        """They used to end by telling the user to run build_exe.py himself,
        which is the second command this change exists to remove."""
        for name in ("build.sh", "build.bat"):
            body = self.read(name)
            self.assertNotIn("build the standalone executable (one folder)",
                             body, name)


class TestDrawPathWiring(unittest.TestCase):
    """Static checks on the parts of the tile that cannot be run here."""

    def _read(self, module: str) -> str:
        path = os.path.join(_PACKAGE, *module.split("/"))
        with open(path, encoding="utf-8") as handle:
            return handle.read()

    def _source(self, module: str, class_name: str, method_name: str) -> str:
        path = os.path.join(_PACKAGE, *module.split("/"))
        source = self._read(module)
        tree = ast.parse(source, filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for method in node.body:
                    if (isinstance(method, ast.FunctionDef)
                            and method.name == method_name):
                        return ast.get_source_segment(source, method) or ""
        self.fail("{0}.{1}() not found".format(class_name, method_name))

    def test_a_fit_reports_that_the_view_moved(self):
        """Statistics are of the visible window, so they must follow a fit.

        The fit sets the ranges with range events suppressed -- otherwise every
        intermediate setXRange/setYRange would trigger a redraw -- which also
        silenced the one signal the Inspect panel listens to. Figures followed a
        pan and a zoom, then sat still after Fit, describing a window the plot
        was no longer showing.
        """
        body = self._source("ui/plot_tile.py", "PlotTile", "autoscale")
        self.assertIn("view_changed.emit", body)

    def test_curves_are_grouped_rather_than_drawn_one_by_one(self):
        """One Qt item per curve does not survive a thousand-run study.

        Curves are collected into a group per pen and painted as one polyline;
        the walk fills in a _GroupPlan, and _adopt_groups() then decides which
        of the items already on the plot each plan can keep.
        """
        body = self._source("ui/plot_tile.py", "PlotTile", "rebuild")
        self.assertIn("_GroupPlan", body)
        self.assertIn("plan.entries.append", body)
        self.assertIn("_adopt_groups", body)

    def test_a_rebuild_does_not_force_a_resample(self):
        """Forcing here is what made every rebuild redraw curves it had just
        proven identical -- a restyle, a grid toggle or a marker drag paid for
        re-decimating every curve in the tile from full resolution.

        A newly built group has no sampled_for and is drawn anyway; an adopted
        one carries the key its points were computed for. So the skip in
        _resample_visible() is the whole mechanism, and forcing past it defeats
        the reuse that _adopt_groups() exists to find.

        Read off _frame_after_rebuild, which is where the framing and the
        redraw at the end of a rebuild live now that the diff path shares
        them with the ordinary one.
        """
        body = self._source("ui/plot_tile.py", "PlotTile",
                            "_frame_after_rebuild")
        self.assertIn("self._resample_visible()", body)
        self.assertNotIn("_resample_visible(force=True)", body)

        rebuild = self._source("ui/plot_tile.py", "PlotTile", "rebuild")
        self.assertNotIn("_resample_visible(force=True)", rebuild)
        # And every path out of a rebuild that drew something goes through it.
        self.assertIn("self._frame_after_rebuild(tile)", rebuild)

    def test_the_legend_is_capped(self):
        body = self._source("ui/plot_tile.py", "PlotTile", "rebuild")
        self.assertIn("MAX_LEGEND_ROWS", body)

    def test_a_redraw_that_needs_no_read_skips_the_worker(self):
        """Most redraws load nothing, and a background task is not free.

        Starting one disables the panels, the rail and the whole plot grid, and
        re-enables them a moment later; every plot widget repaints on each of
        those. Restyling a curve was paying for two full-grid repaints, a
        thread and two trips round the event loop before a curve was redrawn.

        `busy` must still force the slow path: while a worker is inside the core
        session, nothing on the GUI thread may touch it.
        """
        body = self._source("ui/main_window.py", "MainWindow", "refresh_plots")
        self.assertIn("_loaded_signature", body)
        self.assertIn("not self.tasks.busy", body)

    def test_what_was_loaded_is_forgotten_when_it_may_be_gone(self):
        """The fast path above is only sound while the signature is honest.

        Anything that can move or drop the sample buffers has to clear it, or a
        later redraw takes the fast path and reads columns back in on the GUI
        thread -- which on a 500 MB file is the freeze the worker exists to
        prevent.
        """
        body = self._source("ui/main_window.py", "MainWindow", "refresh_all")
        self.assertIn("_loaded_signature = None", body,
                      "refresh_all() must forget the loaded set")

        # A reload replaces every sample buffer the plots were holding, so it
        # reaches refresh_all() rather than repainting on its own.
        reload_body = self._source("ui/main_window.py", "MainWindow",
                                   "reload_files")
        self.assertIn("refresh_all", reload_body)

    def test_a_colour_mode_radio_is_identified_by_its_mode(self):
        """The row may be reordered; the ids may not follow it.

        A button's id is what idClicked hands to _on_mode_changed(), which
        turns it straight into a ColorMode -- and a ColorMode is what a
        workspace stores. Registering the buttons by position instead, which
        is what enumerate() over the labels did, means reordering the row
        silently re-labels every saved tile: "Colour by Simulation" reads back
        as Parameter.
        """
        body = self._source("ui/panels.py", "StylePanel", "__init__")
        colour, _sep, _rest = body.partition('_title("Style by")')
        self.assertTrue(_sep, "the Style by section has moved")
        self.assertIn("self.mode_group.addButton(radio, int(mode))", colour)

        order = [name for name in ("PARAMETER", "SIMULATION", "ISO")
                 if "ColorMode." + name in colour]
        self.assertEqual(
            order, ["PARAMETER", "SIMULATION", "ISO"],
            "Parameter leads the Colour by row")
        self.assertLess(colour.index('"Parameter"'), colour.index('"Simulation"'))

    def test_every_panel_goes_in_through_a_scroll_area(self):
        """The window's minimum height must not be the tallest panel's.

        A plain widget's minimum height is the sum of its children's, and
        STYLE is tall: four switches, two encodings with their combos and
        hints, two tables and a legend group. Added to the toolbar, the page
        bar and the status bar that came out near 950 px -- fine on a
        1080-high screen and taller than the usable area of a 1050-high one,
        where the window could then neither be shrunk nor maximised, because a
        maximise that cannot satisfy the minimum is refused.
        """
        body = self._source("ui/main_window.py", "MainWindow", "_build_central")
        self.assertIn("QScrollArea", body)
        self.assertIn("setWidgetResizable(True)", body)
        # Horizontal scrolling would mean a panel could hide a control off to
        # the side, which is a worse answer than letting it compress.
        self.assertIn("ScrollBarAlwaysOff", body)

    def test_the_panel_on_screen_is_asked_for_by_name(self):
        """currentWidget() is the scroll area now, not the panel.

        Three places compared it against a panel and would quietly stop
        matching: the lazy-refresh test, the stale-panel sweep, and the
        Inspect tab's live statistics.
        """
        source = self._read("ui/main_window.py")
        self.assertIn("def _current_panel", source)
        self.assertNotIn("panel_stack.currentWidget() is", source)
        self.assertNotIn("is self.panel_stack.currentWidget()", source)

    def test_the_ui_scale_reaches_what_measures_itself(self):
        """The stylesheet carries the font sizes and the padding, but not
        everything: widgets that paint their own text measure it themselves,
        and the tab rail's *width* is the height of its rotated labels. A rail
        that did not re-measure would clip every one of them."""
        body = self._source("ui/main_window.py", "MainWindow", "apply_theme")
        self.assertIn("stylesheet(theme)", body)
        self.assertIn("rail.refresh_metrics()", body)
        self.assertIn("setIconSize", body)

        step = self._source("ui/main_window.py", "MainWindow", "step_ui_scale")
        self.assertIn("set_ui_scale", step)
        self.assertIn("apply_theme", step)
        # Saved with the workspace, like the theme is.
        self.assertIn("self.state.ui_scale", step)

    def test_the_scale_is_pushed_into_the_theme_before_anything_measures(self):
        """And before the toolbar, the panels or the first paint.

        The state's scale and the theme module's agree at startup because both
        take the same default, but agreeing by coincidence is how they come
        apart later. Setting it explicitly in __init__ -- ahead of the three
        build steps, every one of which measures something -- is what makes the
        state the single source of the size actually on screen.
        """
        body = self._source("ui/main_window.py", "MainWindow", "__init__")
        self.assertIn("set_ui_scale(self.state.ui_scale)", body)
        position = body.index("set_ui_scale(self.state.ui_scale)")
        for later in ("_build_toolbar()", "_build_central()", "apply_theme()"):
            self.assertIn(later, body)
            self.assertLess(position, body.index(later),
                            "the scale is set after " + later)

    def test_fit_lives_with_the_tile_controls(self):
        """It acts on the tiles, and you pick a tile by clicking it.

        The shortcut is the part that has to survive the move: an action on no
        widget has no shortcut at all, so it is added to the window rather
        than to a bar.
        """
        bar = self._source("ui/main_window.py", "MainWindow", "_build_toolbar")
        self.assertNotIn('add("Fit"', bar,
                         "Fit must not be in the main toolbar any more")
        self.assertIn("self.addAction(self.fit_action)", bar)
        self.assertIn("Ctrl+0", bar)

        central = self._source("ui/main_window.py", "MainWindow", "_build_central")
        self.assertIn("fit_requested", central)

        toolbar = self._read("ui/plot_toolbar.py")
        self.assertIn("fit_requested", toolbar)
        # It re-frames rather than rearranging, so it gets its own rule after
        # the copy/cut/paste group.
        _before, _sep, after = toolbar.partition('"clear", self._clear_tile')
        self.assertTrue(_sep, "the tile clipboard group has moved")
        self.assertIn("_separator()", after)
        self.assertIn("fit_button", after)
        # And Unlock after Fit, since it is Fit with the locks taken off.
        _f, _sep, after_fit = after.partition("fit_button")
        self.assertIn("unlock_button", after_fit)

    def test_the_theme_toggle_lives_in_the_toolbar(self):
        """With the rest of what changes how the window looks.

        It sat at the foot of the tab rail, where it took vertical space from
        the tabs on the screen least able to spare it -- and it is not a tab:
        it selects no panel.
        """
        bar = self._source("ui/main_window.py", "MainWindow", "_build_toolbar")
        self.assertIn("self.theme_action", bar)
        # In the group behind the separator, with the interface-size buttons:
        # everything before that rule acts on the workspace or the plots.
        _before, _sep, after = bar.partition('QLabel(" Interface ")')
        self.assertTrue(_sep, "the interface group has moved")
        self.assertIn("theme_action", after)

        central = self._source("ui/main_window.py", "MainWindow", "_build_central")
        self.assertNotIn("theme_toggled", central)
        self.assertNotIn("theme_toggled", self._read("ui/sidebar.py"))

    def test_the_theme_button_offers_the_theme_it_would_give(self):
        """A toggle labelled with its current state reads as a description;
        one labelled with its result reads as an offer, which is what a button
        is. So the icon and the wording flip with the state, and they are
        re-applied wherever the theme is."""
        body = self._source("ui/main_window.py", "MainWindow",
                            "_update_theme_action")
        self.assertIn("sun", body)
        self.assertIn("moon", body)
        self.assertIn("setText", body)

        icons_source = self._read("ui/icons.py")
        self.assertIn('"sun":', icons_source)
        self.assertIn('"moon":', icons_source)

        applied = self._source("ui/main_window.py", "MainWindow",
                               "_apply_toolbar_icons")
        self.assertIn("_update_theme_action", applied)

    def test_axes_are_chosen_from_what_can_actually_be_plotted(self):
        """Not from `parameters`, which outlives the files.

        Formulas and aliases stay defined when a study is closed, so
        `parameters` is non-empty with nothing open -- and both of these
        choose a channel on the user's behalf. Reading the plain list there is
        what re-pointed every tile's x axis at an alias while the session held
        no files at all.
        """
        for method in ("refresh_all", "refresh_plots"):
            body = self._source("ui/main_window.py", "MainWindow", method)
            for call in ("resolve_missing_parameters", "apply_parameter_defaults"):
                if call not in body:
                    continue
                self.assertIn("plottable_parameters", body,
                              "{0}() must not choose from session.parameters"
                              .format(method))

    def test_reloading_a_file_happens_on_the_worker(self):
        """Re-reading a 578 MB file is a full parse.

        It also moves every sample buffer the plots are holding pointers into,
        so it belongs in a submitted task like every other one -- and the panel
        that owns the button must not make the call itself.
        """
        body = self._source("ui/main_window.py", "MainWindow", "reload_files")
        self.assertIn("self.tasks.submit", body)
        # Either signature: a work function takes a progress reporter when it
        # has steps worth counting. What this test is about is which thread the
        # parse happens on, not how the bar is drawn.
        before_work = body
        for spelling in ("def work():", "def work(report):"):
            head, separator, _rest = body.partition(spelling)
            if separator:
                before_work = head
                break
        else:
            self.fail("reload_files() no longer submits a worker body")
        self.assertNotIn(".reload()", before_work,
                         "the core must not be re-read on the GUI thread")

        panel = self._source("ui/panels.py", "DataPanel", "_reload_selected")
        self.assertIn("reload_requested.emit", panel)
        self.assertNotIn(".reload()", panel,
                         "the panel asks for a reload, it does not perform one")

    def test_nothing_polls_the_disk(self):
        """The 3-second watch timer is gone, and must stay gone.

        It interrupted work that had nothing to do with the file it was asking
        about, and it could not be trusted anyway: a solver halfway through
        rewriting its output is a file that has changed and must not be read.
        Whether the data on disk is finished is a question only the person
        running the solver can answer, so they answer it -- with Reload.
        """
        source = self._read("ui/main_window.py")
        for gone in ("_watch_timer", "_check_for_changes", "_declined_reloads",
                     "_suppress_reload_prompt"):
            self.assertNotIn(gone, source,
                             "{0} belongs to the disk poller".format(gone))

    def test_enabling_the_ui_twice_over_does_nothing(self):
        """Disabling a widget changes how it is drawn, so every plot in the grid
        repaints on the way down and again on the way back up. A chain of queued
        tasks emits started/finished for each one, asking for a state the UI is
        already in."""
        body = self._source("ui/main_window.py", "MainWindow", "_set_ui_enabled")
        self.assertIn("self._ui_enabled", body)

    def test_the_active_border_is_not_reapplied(self):
        """setStyleSheet does no comparison of its own: handed the string it
        already has, it still repolishes the widget and repaints its whole
        scene. _refresh_active_border() calls this on every tile in the page on
        every mouse press, so without the guard one click redrew every plot."""
        body = self._source("ui/plot_tile.py", "PlotTile", "set_active_border")
        self.assertIn("_border_active", body)
        self.assertIn("return", body)

    def test_the_iso_tree_is_built_with_its_signals_off(self):
        """Every setCheckState during a build raises itemChanged exactly as a
        click does. Unguarded, building the tree writes every box it draws back
        into the state and switches on every filter it drew -- and the tree
        draws one item per value, so a sweep does it hundreds of times.

        setUpdatesEnabled is the other half: without it Qt relayouts and
        repaints the widget on every single item.
        """
        body = self._source("ui/panels.py", "IsoPanel", "_rebuild_tree")
        self.assertIn("blockSignals", body)
        self.assertIn("setUpdatesEnabled(False)", body)

    def test_the_iso_tree_skips_a_rebuild_that_would_change_nothing(self):
        """A refresh arrives for reasons that change none of the tree -- a page
        switched, the panel simply shown again -- and a rebuild throws away and
        recreates one widget item per value."""
        body = self._source("ui/panels.py", "IsoPanel", "_rebuild_tree")
        self.assertIn("_tree_signature", body)
        self.assertIn("_tree_state", body)

    def test_only_a_refresh_invalidates_the_iso_value_cache(self):
        """Reading a parameter's distinct values scans every block header of
        every open file, and formatting them crosses the ABI once per value. The
        cache is only sound because refresh() is the one thing that runs when
        the loaded files change -- so it must be the one thing that clears it,
        and it must actually do so."""
        body = self._source("ui/panels.py", "IsoPanel", "refresh")
        self.assertIn("_values_cache = {}", body)
        for method in ("_rebuild_tree", "_on_item_changed", "_set_all"):
            other = self._source("ui/panels.py", "IsoPanel", method)
            self.assertNotIn("_values_cache = {}", other,
                             "{0}() must not clear the cache".format(method))

    def test_the_tiles_do_not_clip_to_view(self):
        """pyqtgraph's clipToView says of itself that it "presumes that x-values
        have uniform spacing": it derives dx from the first and last x and
        slices the array by index arithmetic.

        Curves that share a pen are drawn here as one polyline -- several runs
        concatenated with a NaN between them -- so x runs 0..100, NaN, 0..100,
        and both the spacing and the ordering that slice depends on are false.
        It cut most of the runs out of the item, and the tile came back missing
        curves. ISO colouring is what made it visible: colouring by ISO value is
        what makes many runs share one pen.

        Nothing is lost by leaving it off -- the core decimates to the visible
        x range before the data crosses into Python.
        """
        body = self._source("ui/plot_tile.py", "PlotTile", "__init__")
        self.assertNotIn("setClipToView(True)", body)

    def test_a_failing_tile_does_not_stop_the_others(self):
        """A rebuild reaches a lot of drawing code, and when something in it
        raised the loop stopped at that tile: every tile after it kept whatever
        it was showing before, silently. That reads as "the plot did not
        update", which is far harder to diagnose than one empty tile and a line
        in the log."""
        body = self._source("ui/plot_grid.py", "PlotGrid", "refresh")
        self.assertIn("try:", body)
        self.assertIn("_log.exception", body)

    def test_the_legend_is_placed_after_it_is_filled(self):
        """A legend anchored while empty and then filled changes size under its
        own anchor, and pyqtgraph re-derives the anchor from whichever corner
        the item is nearest -- so a legend dragged to the middle crept further
        on every rebuild until it was back against an edge."""
        body = self._source("ui/plot_tile.py", "PlotTile", "rebuild")
        placed = body.index("_place_legend")
        adopted = body.index("_adopt_groups")
        self.assertGreater(placed, adopted,
                           "_place_legend must come after the groups are built")

    def test_the_resample_path_writes_one_item_per_group(self):
        body = self._source("ui/plot_tile.py", "PlotTile", "_resample_visible")
        self.assertIn("group.item.setData", body)
        # A NaN between runs is what stops the end of one run being joined to
        # the start of the next by a stroke across the plot.
        self.assertIn("connect=\"finite\"", body)


class TestTheCheckItselfWorks(unittest.TestCase):
    """A checker that silently passes on everything is worse than none."""

    def _scan_source(self, source: str):
        import tempfile

        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "sample.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source)
        return _findings(path)

    def test_it_catches_the_real_mistake(self):
        problems = self._scan_source(
            "class W:\n"
            "    def build(self):\n"
            "        self.panel.changed.connect(self.grid.refresh)\n"
            "        self.grid = Grid()\n"
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("self.grid", problems[0])

    def test_the_correct_order_is_accepted(self):
        self.assertEqual(self._scan_source(
            "class W:\n"
            "    def build(self):\n"
            "        self.grid = Grid()\n"
            "        self.panel.changed.connect(self.grid.refresh)\n"
        ), [])

    def test_a_deferred_body_is_exempt(self):
        # The lambda runs long after construction, so naming a later attribute
        # inside it is correct and common.
        self.assertEqual(self._scan_source(
            "class W:\n"
            "    def build(self):\n"
            "        self.panel.changed.connect(lambda: self.grid.refresh())\n"
            "        self.grid = Grid()\n"
        ), [])

    def test_a_hasattr_guard_is_exempt(self):
        self.assertEqual(self._scan_source(
            "class W:\n"
            "    def build(self):\n"
            "        if hasattr(self, 'grid'):\n"
            "            self.grid.refresh()\n"
            "        self.grid = Grid()\n"
        ), [])

    def test_an_attribute_from_another_method_is_not_flagged(self):
        # Assigned in __init__, used in another method: entirely normal.
        self.assertEqual(self._scan_source(
            "class W:\n"
            "    def __init__(self):\n"
            "        self.grid = Grid()\n"
            "    def later(self):\n"
            "        self.grid.refresh()\n"
        ), [])

    def test_read_then_reassign_is_not_flagged(self):
        # The overwhelmingly common shape: __init__ creates it, a later method
        # reads the old value and replaces it. Flagging this made the check
        # useless noise.
        self.assertEqual(self._scan_source(
            "class W:\n"
            "    def __init__(self):\n"
            "        self.items = []\n"
            "    def rebuild(self):\n"
            "        for item in self.items:\n"
            "            drop(item)\n"
            "        self.items = []\n"
        ), [])

    def test_a_class_level_default_is_not_flagged(self):
        self.assertEqual(self._scan_source(
            "class W:\n"
            "    timer = None\n"
            "    def build(self):\n"
            "        if self.timer is not None:\n"
            "            self.timer.stop()\n"
            "        self.timer = Timer()\n"
        ), [])


if __name__ == "__main__":
    unittest.main()
