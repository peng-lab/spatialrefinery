"""Technology-agnostic abstractions: registry, utilities, converters, downloaders.

`registry` and `utils` are imported eagerly here since both are light
(stdlib/numpy/pandas only at module scope). `converter` and `downloader`
pull in heavier dependencies (cv2, tifffile, openslide, ...) and are
exposed lazily via module `__getattr__` so that a bare `import
spatialrefinery.core` stays cheap.
"""

from __future__ import annotations

from spatialrefinery.core import registry, utils

__all__ = ["registry", "utils", "converter", "downloader"]


def __getattr__(name: str):
    if name in ("converter", "downloader"):
        import importlib

        module = importlib.import_module(f"spatialrefinery.core.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
