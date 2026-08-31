r"""Bridges over three InstanSeg defects that block its whole-slide path here.

Both are upstream bugs in `instanseg` 0.1.1 (and unchanged on `main`). They
are patched at call time rather than vendored, so removing this module is all
it takes once InstanSeg fixes them.

1. **zarr 3.** `eval_whole_slide_image` builds its label canvas with
   `zarr.DirectoryStore`, a zarr-2 name that zarr 3 renamed to
   `zarr.storage.LocalStore`. InstanSeg therefore pins `zarr>=2.0.0,<3` in its
   `io` extra -- a cap that collides head-on with `spatialdata` (`zarr>=3.0.0`)
   and `anndata` (`zarr>=3.1`), so installing that extra breaks this package
   outright. The pin's stated reason ("tiffslide
   doesn't support zarr v3 yet") is stale: tiffslide 4.0 added zarr-3 support
   in Bayer-Group/tiffslide#97. Only the name is missing.

2. **`TiffSlide` is never imported.** `InstanSeg.read_slide` calls
   `TiffSlide(image_str)` at `inference_class.py:236`, but every import of that
   name in the module is function-local (lines 63, 93, 183), so `read_slide`
   raises `NameError` on any whole-slide image -- the code path is simply
   untested upstream.

3. **Invalid GeoJSON.** The exporter writes a comma after every feature and
   then closes the array, so output ends `...}},\\n]` -- which `json.load`
   and QuPath both reject. `repair_geojson_trailing_comma` fixes the artefact
   rather than working around it at read time.

All three are idempotent and are exercised by `tests/test_compat.py`.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def patch_zarr_directory_store() -> bool:
    """Alias `zarr.DirectoryStore` to `zarr.storage.LocalStore` when it is missing.

    `LocalStore` is a drop-in for the four things InstanSeg does with the
    store: construct from a path, hand to `zarr.zeros(..., overwrite=True)`,
    slice-assign tiles, and reopen with `zarr.open(path, mode="r")`.

    Returns
    -------
    bool
        True if the alias was installed (zarr 3), False if `zarr.DirectoryStore`
        already exists (zarr 2) and nothing was changed.
    """
    import zarr

    if hasattr(zarr, "DirectoryStore"):
        return False

    zarr.DirectoryStore = zarr.storage.LocalStore  # type: ignore[attr-defined]
    logger.debug("Aliased zarr.DirectoryStore -> zarr.storage.LocalStore (zarr %s)", zarr.__version__)
    return True


def patch_instanseg_tiffslide() -> bool:
    """Inject `TiffSlide` into `instanseg.inference_class`'s module globals.

    `read_slide` references the name without importing it, so every
    whole-slide call raises `NameError` until it is bound.

    Returns
    -------
    bool
        True if the name was injected, False if it was already present.

    Raises
    ------
    ImportError
        If tiffslide is not installed; install `spatialrefinery[segmentation]`.
    """
    from instanseg import inference_class
    from tiffslide import TiffSlide

    if getattr(inference_class, "TiffSlide", None) is not None:
        return False

    inference_class.TiffSlide = TiffSlide
    logger.debug("Injected TiffSlide into instanseg.inference_class (upstream NameError in read_slide)")
    return True


def patch_instanseg() -> None:
    """Apply every InstanSeg compatibility patch. Safe to call repeatedly."""
    patch_zarr_directory_store()
    patch_instanseg_tiffslide()


def repair_geojson_trailing_comma(path) -> bool:
    r"""Drop the trailing comma InstanSeg leaves before the closing bracket.

    `_zarr_to_json_export` streams features out with a comma after each one and
    then writes `]`, so every GeoJSON it produces ends `...}},\\n]` and is
    invalid JSON. `json.load` rejects it, and so does any other consumer
    (QuPath included), which makes this worth fixing in the artefact rather
    than working around at read time.

    Only the tail is rewritten -- these files run to tens of megabytes.

    Parameters
    ----------
    path
        The GeoJSON to repair, modified in place.

    Returns
    -------
    bool
        True if a trailing comma was removed, False if the file was already
        well-formed.
    """
    import re
    from pathlib import Path

    path = Path(path)
    window = 4096
    # Matched on bytes, never decoded: a 4 KiB window can start mid-UTF-8
    # sequence, and decoding would both raise and desynchronise the offsets
    # used to seek back into the file.
    with path.open("r+b") as handle:
        size = handle.seek(0, 2)
        start = max(0, size - window)
        handle.seek(start)
        tail = handle.read()

        # `, ] <whitespace> EOF`, allowing whitespace either side of the comma.
        match = re.search(rb",(\s*\]\s*)$", tail)
        if match is None:
            return False

        handle.seek(start + match.start())
        handle.write(match.group(1))
        handle.truncate()

    logger.debug("Removed trailing comma from %s", path)
    return True
