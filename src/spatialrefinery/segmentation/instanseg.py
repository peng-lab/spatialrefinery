"""Nucleus segmentation on H&E whole-slide images via InstanSeg.

InstanSeg owns the whole-slide layer this needs -- tiling, cross-tile label
matching, an Otsu tissue prefilter and GeoJSON export -- and its runtime
requirements (`numpy>=1.24`, `torch>=2.0`, no upper bounds) sit inside the
environment this package already has, so it installs additively without
disturbing `numpy`, `pydantic`, `zarr` or `spatialdata`.

See `_compat` for the three upstream defects that had to be bridged to make
that whole-slide path usable here.

Results land in `<outdir>/<wsi filename incl. extension>/cells.geojson`.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: InstanSeg appends this to the input stem when naming its outputs.
PREDICTION_TAG = "_instanseg_prediction"

DEFAULT_MODEL = "brightfield_nuclei"
DEFAULT_TILE_SIZE = 512
DEFAULT_OVERLAP = 80
DEFAULT_DETECTION_SIZE = 20


def _resolve_device(gpu_id: int | None) -> str:
    """Return the torch device string, falling back to CPU when no GPU is visible."""
    import torch

    if gpu_id is None or not torch.cuda.is_available():
        if gpu_id is not None:
            logger.warning("No CUDA device visible; falling back to CPU (this will be slow).")
        return "cpu"
    return f"cuda:{gpu_id}"


def _find_prediction_geojson(directory: Path) -> Path | None:
    """Return the GeoJSON InstanSeg wrote into `directory`, if any.

    InstanSeg names it `<stem><PREDICTION_TAG>.geojson`. `stem` strips only the
    final suffix, so `slide.ome.tif` yields `slide.ome_instanseg_prediction.geojson`
    -- hence a glob rather than a constructed name.
    """
    matches = sorted(directory.glob(f"*{PREDICTION_TAG}.geojson"))
    return matches[0] if matches else None


DEFAULT_CLAHE_GRID = 8


def _apply_clahe(tile, clip_limit: float, grid: int = DEFAULT_CLAHE_GRID):
    """Locally equalise a tile's lightness, leaving hue alone.

    Applied to L in LAB rather than to RGB, so the haematoxylin/eosin hue
    balance the model was trained on is preserved and only local contrast
    changes.
    """
    import cv2
    import numpy as np

    if tile.ndim != 3 or tile.shape[-1] < 3:
        return tile
    rgb = np.ascontiguousarray(tile[..., :3], dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _enable_tile_clahe(model, clip_limit: float, grid: int = DEFAULT_CLAHE_GRID) -> None:
    """Make `model` run CLAHE over every tile before inference.

    `eval_whole_slide_image` reads each tile and hands it straight to
    `self._to_tensor`, so wrapping that one method is enough to reach every
    tile without forking InstanSeg. Per-tile is also the right granularity:
    CLAHE is adaptive by design, and the model's 80 px tile overlap plus
    cross-tile label matching absorb the small discontinuities at the seams.
    """
    original = model._to_tensor

    def _to_tensor_with_clahe(image):
        return original(_apply_clahe(image, clip_limit, grid))

    model._to_tensor = _to_tensor_with_clahe


def segment_wsi(
    wsi_path: str | Path,
    outdir: str | Path,
    *,
    pixel_size: float | None = None,
    gpu_id: int | None = 0,
    model_type: str = DEFAULT_MODEL,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    detection_size: int = DEFAULT_DETECTION_SIZE,
    use_otsu_threshold: bool = True,
    clahe_clip: float | None = None,
    seed_threshold: float | None = None,
    skip_existing: bool = True,
) -> Path:
    """Segment nuclei in one whole-slide image and return the GeoJSON path.

    Parameters
    ----------
    wsi_path
        The slide to segment. Read through tiffslide, so SVS/NDPI/OME-TIFF and
        other TIFF-backed formats work.
    outdir
        Parent directory. Results go to `<outdir>/<wsi_path.name>/cells.geojson`;
        the directory is named with the full filename (extension included) so
        that `a.svs` and `a.ndpi` cannot collide. The zarr stage names its store
        for the stem instead (`a.ome.tif` -> `a.zarr`); see
        `to_spatialdata.default_zarr_path`.
    pixel_size
        Microns per pixel. Read from the slide metadata when omitted. InstanSeg
        rejects a value outside [0.1, 1] micron, so pass this explicitly for
        slides with missing or nonsensical resolution tags.
    gpu_id
        CUDA device index, or None to force CPU. Under the SLURM worker each
        process sees a single GPU via `CUDA_VISIBLE_DEVICES`, so this stays 0.
    use_otsu_threshold
        Skip tiles outside the tissue mask, so background is not segmented.
    clahe_clip
        Run CLAHE over each tile at this clip limit before inference. Off by
        default. Worth setting (2.0 is a reasonable start) on weakly
        haematoxylin-stained slides, where InstanSeg's per-tile percentile
        normalisation leaves pale nuclei below the seed threshold: on a pale
        kidney H&E it recovered 22% more nuclei, and 27% together with
        `seed_threshold=0.4`.
    seed_threshold
        Override the model's seed threshold (default 0.7). Lower detects
        fainter nuclei at some risk of over-segmentation.
    skip_existing
        Return immediately if `cells.geojson` is already present, which makes
        an interrupted batch resumable.

    Returns
    -------
    Path
        The written `cells.geojson`.
    """
    wsi_path = Path(wsi_path)
    sample_dir = Path(outdir) / wsi_path.name
    cells_geojson = sample_dir / "cells.geojson"

    if skip_existing and cells_geojson.exists():
        logger.info("Skipping %s: %s already exists", wsi_path.name, cells_geojson)
        return cells_geojson

    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI not found: {wsi_path}")

    sample_dir.mkdir(parents=True, exist_ok=True)

    # InstanSeg writes its .zarr and .geojson next to the *input* file. The
    # slides live on a shared read-only dataset mount, so run it against a
    # symlink inside the output directory and let the outputs land there.
    linked_wsi = sample_dir / wsi_path.name
    if linked_wsi.is_symlink() or linked_wsi.exists():
        linked_wsi.unlink()
    linked_wsi.symlink_to(wsi_path.resolve())

    from instanseg import InstanSeg

    from spatialrefinery.segmentation._compat import (
        patch_instanseg,
        repair_geojson_trailing_comma,
    )

    # Bridges two upstream defects that break the whole-slide path; see _compat.
    patch_instanseg()

    device = _resolve_device(gpu_id)
    logger.info("Segmenting %s with InstanSeg(%s) on %s", wsi_path.name, model_type, device)

    model = InstanSeg(model_type, device=device, image_reader="tiffslide", verbosity=1)
    if clahe_clip is not None:
        logger.info("Applying CLAHE (clip=%.1f) to each tile before inference", clahe_clip)
        _enable_tile_clahe(model, clahe_clip)

    # Only forwarded when set, so the model's own defaults stay in charge.
    model_kwargs = {} if seed_threshold is None else {"seed_threshold": seed_threshold}

    try:
        model.eval_whole_slide_image(
            image=str(linked_wsi),
            pixel_size=pixel_size,
            tile_size=tile_size,
            overlap=overlap,
            detection_size=detection_size,
            use_otsu_threshold=use_otsu_threshold,
            save_geojson=True,
            **model_kwargs,
        )
    finally:
        linked_wsi.unlink(missing_ok=True)

    produced = _find_prediction_geojson(sample_dir)
    if produced is None:
        raise FileNotFoundError(
            f"InstanSeg reported success but wrote no *{PREDICTION_TAG}.geojson in {sample_dir}. "
            "GeoJSON export needs rasterio and geojson: install spatialrefinery[segmentation]."
        )

    # InstanSeg leaves a trailing comma before the closing bracket, so the file
    # it just wrote is not valid JSON until this runs.
    if repair_geojson_trailing_comma(produced):
        logger.debug("Repaired trailing comma in %s", produced.name)

    produced.replace(cells_geojson)
    logger.info("Wrote %s", cells_geojson)
    return cells_geojson
