"""Tests for `spatialrefinery.segmentation`.

The InstanSeg model itself is not exercised here -- it needs a GPU and a
downloaded checkpoint, and is covered by the smoke test in
`slurm/SEGMENTATION_PLAN.md`. What these tests pin is the glue around it:
where the GeoJSON is looked for, how its properties are handled, and that the
image element is built lazily from the slide's own pyramid.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import tifffile as tf

from spatialrefinery.segmentation.instanseg import (
    PREDICTION_TAG,
    _find_prediction_geojson,
    segment_wsi,
)
from spatialrefinery.segmentation.to_spatialdata import (
    explode_multipolygons,
    geojson_to_spatialdata,
    wsi_image_element,
)


def make_slide(path: Path, height: int = 256, width: int = 192, seed: int = 0) -> np.ndarray:
    """Write a random RGB tiled TIFF and return its pixels."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    with tf.TiffWriter(path) as writer:
        writer.write(image, tile=(64, 64), photometric="rgb")
    return image


def square(x0: float, y0: float, size: float = 4.0) -> list:
    """Return a closed square ring as GeoJSON coordinates."""
    return [[[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size], [x0, y0]]]


def write_geojson(path: Path, n: int = 3) -> None:
    """Write a GeoJSON with InstanSeg's real property schema.

    Taken verbatim from a whole-slide run: a bare list (not a FeatureCollection),
    and every feature carries `object_type`, `measurements`, and a *constant*
    `classification` of "Detection" -- which is a marker, not a cell type, and
    must not reach the table.
    """
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": square(10.0 * i, 10.0 * i)},
            "properties": {
                "object_type": "detection",
                "measurements": [{"name": "Label", "value": i + 1}],
                "classification": "Detection",
            },
        }
        for i in range(n)
    ]
    path.write_text(json.dumps(features))


def make_template(path: Path, n_vars: int = 5) -> None:
    """Write a minimal template AnnData carrying only `var`."""
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    ad.AnnData(X=np.zeros((1, n_vars), dtype=np.float32), var=var).write_h5ad(path)


# --------------------------------------------------------------------- #
# Output discovery
# --------------------------------------------------------------------- #


def test_find_prediction_geojson_handles_double_suffix(tmp_path):
    """`slide.ome.tif` has stem `slide.ome`, so the name cannot be reconstructed."""
    expected = tmp_path / f"slide.ome{PREDICTION_TAG}.geojson"
    expected.touch()
    assert _find_prediction_geojson(tmp_path) == expected


def test_find_prediction_geojson_returns_none_when_absent(tmp_path):
    """A missing GeoJSON is reported as None, not an IndexError."""
    (tmp_path / "unrelated.geojson").touch()
    assert _find_prediction_geojson(tmp_path) is None


# --------------------------------------------------------------------- #
# segment_wsi guard rails (no model involved)
# --------------------------------------------------------------------- #


def test_segment_wsi_skips_existing_without_loading_model(tmp_path, monkeypatch):
    """`--skip-existing` must return before importing InstanSeg, so resume is cheap."""
    wsi = tmp_path / "slide.tif"
    make_slide(wsi)
    sample_dir = tmp_path / "out" / "slide.tif"
    sample_dir.mkdir(parents=True)
    cells = sample_dir / "cells.geojson"
    cells.write_text("[]")

    # Any attempt to build the model would blow up here.
    monkeypatch.setattr(
        "spatialrefinery.segmentation._compat.patch_instanseg",
        lambda: pytest.fail("model path entered despite existing cells.geojson"),
    )

    assert segment_wsi(wsi, tmp_path / "out") == cells


def test_segment_wsi_missing_slide_raises(tmp_path):
    """A missing slide fails loudly rather than producing an empty result."""
    with pytest.raises(FileNotFoundError):
        segment_wsi(tmp_path / "nope.tif", tmp_path / "out")


# --------------------------------------------------------------------- #
# GeoJSON handling
# --------------------------------------------------------------------- #


def test_explode_multipolygons_splits_parts(tmp_path):
    """A MultiPolygon becomes one row per part, each keeping the parent's properties."""
    src = tmp_path / "cells.geojson"
    src.write_text(
        json.dumps(
            [
                {
                    "type": "Feature",
                    "geometry": {"type": "MultiPolygon", "coordinates": [square(0, 0), square(20, 20)]},
                    "properties": {"object_type": "annotation"},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": square(40, 40)},
                    "properties": {"object_type": "annotation"},
                },
            ]
        )
    )

    gdf = explode_multipolygons(src, tmp_path / "exploded.geojson")

    assert len(gdf) == 3
    assert gdf.geometry.geom_type.unique().tolist() == ["Polygon"]
    assert (tmp_path / "exploded.geojson").exists()


def test_instanseg_classification_is_a_constant_marker(tmp_path):
    """InstanSeg's `classification` is always "Detection", never a cell type.

    Carrying that constant into `table.obs` would add a column that looks like
    a cell-type label but distinguishes nothing, so it must stay out.
    """
    src = tmp_path / "cells.geojson"
    write_geojson(src, n=3)
    gdf = explode_multipolygons(src, tmp_path / "exploded.geojson")
    assert set(gdf["classification"]) == {"Detection"}


# --------------------------------------------------------------------- #
# Image element
# --------------------------------------------------------------------- #


def test_wsi_image_element_is_channel_first_multiscale(tmp_path):
    """The element must be (c, y, x) with an RGB channel axis, built from the file lazily."""
    wsi = tmp_path / "slide.tif"
    image = make_slide(wsi, height=256, width=192)

    element = wsi_image_element(wsi, scale_factors=(2,))

    # A multiscale element is a DataTree; its full-resolution level is "scale0".
    full = element["scale0"].image if hasattr(element, "__getitem__") else element
    assert full.dims == ("c", "y", "x")
    assert full.shape == (3, 256, 192)
    np.testing.assert_array_equal(np.asarray(full.data[:, :8, :8]), image[:8, :8, :].transpose(2, 0, 1))


def test_wsi_image_element_rejects_non_tiff(tmp_path):
    """A format tifffile cannot open should point at the OME-TIFF converter."""
    bogus = tmp_path / "slide.czi"
    bogus.write_bytes(b"not a tiff")
    with pytest.raises(ValueError, match="convert_to_ometiff"):
        wsi_image_element(bogus)


# --------------------------------------------------------------------- #
# End-to-end conversion
# --------------------------------------------------------------------- #


def test_geojson_to_spatialdata_writes_expected_elements(tmp_path):
    """The written zarr must carry the image, the shapes, and a table annotating them."""
    import spatialdata

    wsi = tmp_path / "slide.tif"
    make_slide(wsi)
    geojson = tmp_path / "cells.geojson"
    write_geojson(geojson, n=4)
    template = tmp_path / "template.h5ad"
    make_template(template, n_vars=5)

    zarr_path = tmp_path / "slide.tif.zarr"
    geojson_to_spatialdata(
        geojson_path=geojson,
        zarr_path=zarr_path,
        image_path=wsi,
        template_adata_path=template,
        write_zip=False,
    )

    assert zarr_path.exists()
    sdata = spatialdata.read_zarr(zarr_path)
    assert "he_image" in sdata.images
    assert "nucleus_boundaries" in sdata.shapes
    assert len(sdata["nucleus_boundaries"]) == 4

    table = sdata["table"]
    assert table.n_obs == 4
    assert table.var_names.tolist() == [f"gene_{i}" for i in range(5)]
    assert "classification" not in table.obs.columns
    assert set(table.obs["region"]) == {"nucleus_boundaries"}
    assert table.obsm["spatial"].shape == (4, 2)


def test_geojson_to_spatialdata_rejects_empty_geojson(tmp_path):
    """An empty segmentation should fail loudly, not write a degenerate zarr."""
    wsi = tmp_path / "slide.tif"
    make_slide(wsi)
    geojson = tmp_path / "cells.geojson"
    geojson.write_text("[]")
    template = tmp_path / "template.h5ad"
    make_template(template)

    with pytest.raises(ValueError, match="No polygons"):
        geojson_to_spatialdata(
            geojson_path=geojson,
            zarr_path=tmp_path / "out.zarr",
            image_path=wsi,
            template_adata_path=template,
            write_zip=False,
        )


def test_centroids_are_planar_pixel_coordinates(tmp_path):
    """Centroids must be plain pixel means, not reprojected through a geographic CRS.

    `gpd.read_file` tags GeoJSON as EPSG:4326 by default; if that CRS survives,
    `.centroid` is computed against a spherical datum and every coordinate
    drifts. The squares here are axis-aligned, so the answer is exact.
    """
    wsi = tmp_path / "slide.tif"
    make_slide(wsi)
    geojson = tmp_path / "cells.geojson"
    write_geojson(geojson, n=3)  # squares of side 4 at (0,0), (10,10), (20,20)
    template = tmp_path / "template.h5ad"
    make_template(template)

    import spatialdata

    zarr_path = tmp_path / "out.zarr"
    geojson_to_spatialdata(
        geojson_path=geojson,
        zarr_path=zarr_path,
        image_path=wsi,
        template_adata_path=template,
        write_zip=False,
    )

    spatial = spatialdata.read_zarr(zarr_path)["table"].obsm["spatial"]
    expected = np.array([[2.0, 2.0], [12.0, 12.0], [22.0, 22.0]])
    np.testing.assert_allclose(np.sort(spatial, axis=0), expected, atol=1e-9)


# --------------------------------------------------------------------- #
# Sensitivity controls
# --------------------------------------------------------------------- #


def test_clahe_widens_dynamic_range_without_shifting_shape_or_dtype():
    """CLAHE must return a drop-in tile whose lightness is more spread out.

    Spread is the property that matters and the one that holds regardless of
    texture: this cohort's slides are pale (median lightness 205/255), which
    compresses nuclei against cytoplasm until local equalisation pulls them
    apart. Measured on a real crop from the kidney slide, std goes 21.8 -> 34.0.
    """
    from spatialrefinery.segmentation.instanseg import _apply_clahe

    rng = np.random.default_rng(0)
    tile = np.full((128, 128, 3), 205, dtype=np.uint8)
    tile[40:60, 40:60] = 180  # a faint nucleus
    tile = np.clip(tile + rng.integers(-3, 4, tile.shape), 0, 255).astype(np.uint8)

    out = _apply_clahe(tile, clip_limit=2.0)

    assert out.shape == tile.shape
    assert out.dtype == np.uint8
    assert out.mean(axis=2).std() > tile.mean(axis=2).std()


def test_apply_clahe_passes_through_non_rgb():
    """A tile without three channels is returned untouched rather than crashing."""
    from spatialrefinery.segmentation.instanseg import _apply_clahe

    grey = np.full((16, 16), 128, dtype=np.uint8)
    assert _apply_clahe(grey, clip_limit=2.0) is grey


def test_enable_tile_clahe_wraps_every_tile():
    """The wrapper must intercept `_to_tensor`, which is the WSI loop's only tile hook."""
    from spatialrefinery.segmentation.instanseg import _enable_tile_clahe

    seen = []

    class FakeModel:
        def _to_tensor(self, image):
            seen.append(image)
            return image

    model = FakeModel()
    original = model._to_tensor
    _enable_tile_clahe(model, clip_limit=2.0)
    assert model._to_tensor is not original

    tile = np.full((64, 64, 3), 200, dtype=np.uint8)
    tile[20:30, 20:30] = 170
    model._to_tensor(tile)

    assert len(seen) == 1
    # The model saw the enhanced tile, not the raw one.
    assert not np.array_equal(seen[0], tile)
    assert seen[0].shape == tile.shape
