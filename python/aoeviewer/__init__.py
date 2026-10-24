# Author: Paulo R. A. Gilz
"""aoeViewer - fast viewer for AOE simulation result files.

The heavy lifting (parsing, formula evaluation, decimation) lives in a C++ core
reached through a plain C ABI; this package is the Python front end plus a
scripting API over the same core.

Typical scripted use::

    from aoeviewer import Session

    with Session() as session:
        f = session.add_file("results/run_042.aoe")
        session.preload(f.index, [], ["TIME", "ALPHA"])
        for sim in f.simulations:
            t = sim.channel("TIME")
            a = sim.channel("ALPHA")
            print(sim.sim_id, a.max())
"""

__version__ = "2.2.1"

# Imported lazily rather than at module import.
#
# Loading `aoeviewer` used to pull in the core library and NumPy immediately,
# which meant a failure in either happened *before* logging could be installed
# -- so the one thing that would explain it was never written down. It also made
# `--version` and `--help` fail on a machine where the core was missing, which
# is exactly when you want to run them.
_LAZY = {
    "Session": "session",
    "FileHandle": "session",
    "SimulationHandle": "session",
    "IsoGroup": "session",
    "CoreError": "session",
}


def __getattr__(name):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError("module 'aoeviewer' has no attribute {0!r}".format(name))
    import importlib

    module = importlib.import_module("." + module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value          # cache, so this happens once
    return value


def __dir__():
    return sorted(list(globals()) + list(_LAZY))

__all__ = [
    "Session",
    "FileHandle",
    "SimulationHandle",
    "IsoGroup",
    "CoreError",
    "__version__",
]
