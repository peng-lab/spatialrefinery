from __future__ import annotations

from importlib.metadata import version

from spatialrefinery import _logging  # attaches the package's rich log handler

__version__ = version("spatialrefinery")

#: Public functions re-exported at the top level, keyed by their owning module.
#: Kept in `__getattr__` (rather than imported eagerly) so `import spatialrefinery`
#: does not pull in `spatialdata`/`openslide` just to read `__version__`.
_LAZY_ATTRS = {
    "convert_to_ometiff": "spatialrefinery.core.converter",
    "download_xenium_study": "spatialrefinery.io.xenium",
    "xenium_to_spatialdata": "spatialrefinery.io.xenium",
    "xenium_to_spatialdata_zip": "spatialrefinery.io.xenium",
}

__all__ = ["__version__", "core", "io", "segmentation", *_LAZY_ATTRS]


def __getattr__(name: str):
    """Lazily expose `spatialrefinery.core`/`.io`/`.segmentation` and the documented top-level functions."""
    import importlib

    if name in ("core", "io", "segmentation"):
        module = importlib.import_module(f"spatialrefinery.{name}")
        globals()[name] = module
        return module
    if name in _LAZY_ATTRS:
        module = importlib.import_module(_LAZY_ATTRS[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily-exposed names in `dir(spatialrefinery)`/tab-completion."""
    return sorted(__all__)
