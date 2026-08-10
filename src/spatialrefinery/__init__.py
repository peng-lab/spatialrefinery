from __future__ import annotations

from importlib.metadata import version

from spatialrefinery import _logging  # attaches the package's rich log handler

__version__ = version("spatialrefinery")

__all__ = ["__version__", "core", "io"]


def __getattr__(name: str):
    """Lazily expose `spatialrefinery.core`/`.io` so `import spatialrefinery` stays cheap."""
    if name in ("core", "io"):
        import importlib

        module = importlib.import_module(f"spatialrefinery.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
