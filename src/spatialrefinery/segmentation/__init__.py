"""Nucleus segmentation on H&E whole-slide images, and its SpatialData export.

Both submodules pull heavy optional dependencies -- `instanseg` and `torch` for
`instanseg`, `spatialdata` and `geopandas` for `to_spatialdata` -- so they are
exposed lazily here, matching `spatialrefinery.core`. Install them with the
`segmentation` extra:

    uv pip install 'spatialrefinery[segmentation]'
"""

from __future__ import annotations

__all__ = ["instanseg", "to_spatialdata"]


def __getattr__(name: str):
    if name in __all__:
        import importlib

        module = importlib.import_module(f"spatialrefinery.segmentation.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
