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
import os
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PACKAGE = os.path.join(_ROOT, "python", "aoeviewer")


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


class TestDrawPathWiring(unittest.TestCase):
    """Static checks on the parts of the tile that cannot be run here."""

    def _source(self, module: str, class_name: str, method_name: str) -> str:
        path = os.path.join(_PACKAGE, *module.split("/"))
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
        """
        body = self._source("ui/plot_tile.py", "PlotTile", "rebuild")
        self.assertIn("self._resample_visible()", body)
        self.assertNotIn("_resample_visible(force=True)", body)

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
        for method in ("refresh_all", "_check_for_changes"):
            body = self._source("ui/main_window.py", "MainWindow", method)
            self.assertIn("_loaded_signature = None", body,
                          "{0}() must forget the loaded set".format(method))

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
