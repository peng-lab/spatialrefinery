"""Registry for spatial-omics technologies and image-format converters.

This module is the single extension point for adding a new technology
(Visium, MERFISH, CosMx, ...) or a new image-format reader without having
to modify existing code. It intentionally imports nothing beyond the
standard library and ``typing`` so that ``import spatialrefinery.core``
never drags in heavy dependencies (spatialdata, openslide, cv2, ...).

Two independent namespaces are registered here:

- **Technologies** (e.g. ``"xenium"``): bundles a downloader, a converter,
  and a file-finder for one spatial-omics platform.
- **Converters by file suffix** (e.g. ``".svs"``): image-format readers,
  which are not technology-scoped -- a ``.svs`` file is a ``.svs`` file
  regardless of which assay produced it.

Dataset catalogs are deliberately *not* registered here. The sample
catalog lives in ``example_data/``, which is gitignored and not shipped
in the wheel; a dataset registry would either duplicate that catalog in
Python or force it into the package. Downloaders derive their study list
on demand from the manifest instead (see :mod:`spatialrefinery.io.xenium`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from spatialrefinery.core.converter import BaseConverter
    from spatialrefinery.core.downloader import BaseDownloader

logger = logging.getLogger(__name__)


class RegistryError(KeyError):
    """Raised when a technology or converter cannot be found, or a registration conflicts."""


@dataclass(frozen=True, slots=True)
class TechnologySpec:
    """Everything the package knows about one spatial-omics technology."""

    name: str
    downloader: type[BaseDownloader] | None = None
    converter: type[BaseConverter] | None = None
    file_finder: Callable[[Path], dict[str, Path | None]] | None = None
    aliases: tuple[str, ...] = ()
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


_TECHNOLOGIES: dict[str, TechnologySpec] = {}
_ALIASES: dict[str, str] = {}
_CONVERTERS: dict[str, type] = {}
_BUILTINS_LOADED = False


def _normalise(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _normalise_suffix(suffix: str) -> str:
    suffix = suffix.strip().lower()
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return suffix


def _ensure_builtins() -> None:
    """Import built-in registrations exactly once, lazily.

    The flag is set *before* the import so that a partially-failed heavy
    import (e.g. a missing optional dependency) can never recurse back
    into this function.
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    try:
        import spatialrefinery.core.converter
        import spatialrefinery.io  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        logger.warning(
            "built-in registrations unavailable (%s); install the full dependency set "
            "with `pip install spatialrefinery` (and `spatialrefinery[czi]` for CZI support)",
            exc,
        )


# --------------------------------------------------------------------- #
# Technologies
# --------------------------------------------------------------------- #


def register_technology(spec: TechnologySpec, *, overwrite: bool = False) -> TechnologySpec:
    """Register a :class:`TechnologySpec`, returning it for convenience."""
    key = _normalise(spec.name)
    if key in _TECHNOLOGIES and not overwrite:
        raise RegistryError(f"Technology {spec.name!r} is already registered (pass overwrite=True to replace it).")
    _TECHNOLOGIES[key] = spec
    for alias in spec.aliases:
        _ALIASES[_normalise(alias)] = key
    return spec


def unregister_technology(name: str) -> None:
    """Remove a technology and any aliases pointing to it. No-op if unknown."""
    key = _normalise(name)
    _TECHNOLOGIES.pop(key, None)
    for alias, target in list(_ALIASES.items()):
        if target == key:
            del _ALIASES[alias]


def get_technology(name: str) -> TechnologySpec:
    """Look up a technology by name or alias, case- and whitespace-insensitively."""
    _ensure_builtins()
    key = _normalise(name)
    key = _ALIASES.get(key, key)
    try:
        return _TECHNOLOGIES[key]
    except KeyError:
        known = ", ".join(sorted(_TECHNOLOGIES)) or "<none registered>"
        raise RegistryError(f"Unknown technology {name!r}. Known technologies: {known}") from None


def list_technologies() -> list[str]:
    """Return every registered technology name, sorted."""
    _ensure_builtins()
    return sorted(_TECHNOLOGIES)


# --------------------------------------------------------------------- #
# Converters (by file suffix)
# --------------------------------------------------------------------- #


def register_converter(
    cls: type | None = None,
    *,
    suffixes: tuple[str, ...] | list[str] | None = None,
    overwrite: bool = False,
) -> Any:
    """Register a converter class against the file suffixes it handles.

    Usable bare (``@register_converter``, in which case ``cls.input_suffixes``
    supplies the suffixes) or parametrised (``@register_converter(suffixes=[".czi"])``).
    """

    def _register(target: type) -> type:
        resolved = suffixes if suffixes is not None else getattr(target, "input_suffixes", ())
        for suffix in resolved:
            key = _normalise_suffix(suffix)
            if key in _CONVERTERS and not overwrite:
                raise RegistryError(
                    f"Suffix {key!r} is already registered to {_CONVERTERS[key].__name__} "
                    f"(pass overwrite=True to replace it)."
                )
            _CONVERTERS[key] = target
        return target

    if cls is not None:
        return _register(cls)
    return _register


def get_converter_for(path: str | Path) -> type:
    """Return the converter class registered for ``path``'s suffix."""
    _ensure_builtins()
    from pathlib import Path as _Path

    p = _Path(path)
    if p.name.lower().endswith((".ome.tif", ".ome.tiff")):
        # `.ome.tif`/`.ome.tiff` is *our own* pyramidal output convention, not a
        # vendor input format -- even though its final suffix (`.tif`) matches a
        # registered image converter. Without this, re-running a conversion over
        # a directory that already contains outputs would re-ingest them as inputs.
        raise RegistryError(f"{p} is itself a converter output (OME-TIFF); refusing to treat it as an input.")

    suffix = _normalise_suffix(p.suffix)
    try:
        return _CONVERTERS[suffix]
    except KeyError:
        known = ", ".join(sorted(_CONVERTERS)) or "<none registered>"
        raise RegistryError(f"No converter registered for suffix {suffix!r}. Known suffixes: {known}") from None


def list_converters() -> dict[str, str]:
    """Return a mapping of registered suffix -> converter class name."""
    _ensure_builtins()
    return {suffix: cls.__name__ for suffix, cls in sorted(_CONVERTERS.items())}
