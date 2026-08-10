"""Tests for `spatialrefinery.core.utils`.

Only exercises the dependency-light surface (pure helpers, spot-grid
geometry, and the mask->polygon extraction used by `segment_tissue`) so this
file runs fast and offline; `segment_tissue` itself needs a real
openslide-readable image on disk and is left to the end-to-end scripts.
"""

from __future__ import annotations

import numpy as np
import pytest

from spatialrefinery.core.utils import (
    _mask_to_gdf as mask_to_gdf,
)
from spatialrefinery.core.utils import (
    bin_centroids,
    hex_grid_centroids,
    human_bytes,
    parse_curl_manifest,
    safe_extract_zip,
    split_study_filename,
    transform_name,
)

# --------------------------------------------------------------------- #
# Names and manifests
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("old_name", "expected"),
    [
        ("simple_name", "simple_name"),
        ("has space", "has_space"),
        ("weird/chars:here", "weird_chars_here"),
        ("dots.and-dashes_ok", "dots.and-dashes_ok"),
    ],
)
def test_transform_name(old_name: str, expected: str) -> None:
    assert transform_name(old_name) == expected


def test_parse_curl_manifest_ignores_non_curl_lines(tmp_path) -> None:
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        "# a comment\n"
        "\n"
        "curl -O https://example.com/study_a/file_outs.zip\n"
        "curl -L -O https://example.com/study_b/file_outs.zip\n"
        "not a curl line\n"
    )
    urls = parse_curl_manifest(manifest)
    assert urls == [
        "https://example.com/study_a/file_outs.zip",
        "https://example.com/study_b/file_outs.zip",
    ]


def test_split_study_filename() -> None:
    study, filename = split_study_filename("https://example.com/some/path/study_a/file_outs.zip")
    assert (study, filename) == ("study_a", "file_outs.zip")


def test_split_study_filename_too_short_raises() -> None:
    with pytest.raises(ValueError, match="Cannot determine study"):
        split_study_filename("https://example.com/file_outs.zip")


@pytest.mark.parametrize(
    ("n", "expected_prefix"),
    [
        (500, "500.0B"),
        (1536, "1.5KB"),
        (1024**3, "1.0GB"),
    ],
)
def test_human_bytes(n: int, expected_prefix: str) -> None:
    assert human_bytes(n) == expected_prefix


def test_safe_extract_zip_rejects_zip_slip(tmp_path) -> None:
    import zipfile

    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../escaped.txt", "gotcha")

    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(ValueError, match="zip-slip"):
        safe_extract_zip(archive, dest)
    assert not (tmp_path / "escaped.txt").exists()


def test_safe_extract_zip_extracts_safe_members(tmp_path) -> None:
    import zipfile

    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("nested/file.txt", "hello")

    dest = tmp_path / "dest"
    dest.mkdir()
    safe_extract_zip(archive, dest)
    assert (dest / "nested" / "file.txt").read_text() == "hello"


# --------------------------------------------------------------------- #
# Spot-grid geometry
# --------------------------------------------------------------------- #


def test_bin_centroids_empty_input() -> None:
    cx, cy = bin_centroids(np.array([]), np.array([]), spot_size_um=5.0)
    assert cx.size == 0
    assert cy.size == 0


def test_bin_centroids_exact_multiple_extent_no_aliasing() -> None:
    """Regression test for the off-by-one in `n_cols` (see core/utils.py).

    With `x_min=0, x_max=10, spot_size_um=5`, the old `ceil`-based `n_cols`
    was 2, so the point at `x=10` (`col_idx=2`) aliased onto row+1/col 0,
    producing a spurious centroid at the wrong y. Both points share `y=0`,
    so both centroids must land in row 0.
    """
    x = np.array([0.0, 10.0])
    y = np.array([0.0, 0.0])
    cx, cy = bin_centroids(x, y, spot_size_um=5.0)
    assert len(cx) == 2
    assert np.allclose(cy, 2.5)
    assert sorted(cx.tolist()) == [2.5, 12.5]


def test_hex_grid_centroids_shape() -> None:
    bounds = (0.0, 0.0, 100.0, 100.0)
    centroids = hex_grid_centroids(bounds, spot_size_um=20.0, overlap=0.0)
    assert centroids.ndim == 2
    assert centroids.shape[1] == 2
    assert len(centroids) > 0


def test_hex_grid_centroids_overlap_increases_density() -> None:
    bounds = (0.0, 0.0, 100.0, 100.0)
    no_overlap = hex_grid_centroids(bounds, spot_size_um=20.0, overlap=0.0)
    with_overlap = hex_grid_centroids(bounds, spot_size_um=20.0, overlap=0.3)
    assert len(with_overlap) >= len(no_overlap)


def test_hex_grid_centroids_none_overlap_treated_as_zero() -> None:
    bounds = (0.0, 0.0, 100.0, 100.0)
    from_none = hex_grid_centroids(bounds, spot_size_um=20.0, overlap=None)
    from_zero = hex_grid_centroids(bounds, spot_size_um=20.0, overlap=0.0)
    assert from_none.shape == from_zero.shape
    assert np.allclose(from_none, from_zero)


@pytest.mark.parametrize("overlap", [1.0, 1.5, -0.1])
def test_hex_grid_centroids_invalid_overlap_raises(overlap: float) -> None:
    with pytest.raises(ValueError, match="overlap"):
        hex_grid_centroids((0.0, 0.0, 100.0, 100.0), spot_size_um=20.0, overlap=overlap)


# --------------------------------------------------------------------- #
# Tissue mask -> polygons (regression net for the hestcore -> cv2 rewrite)
# --------------------------------------------------------------------- #


def test_mask_to_gdf_extracts_polygons_with_holes_and_scales_by_pixel_size() -> None:
    cv2 = pytest.importorskip("cv2")

    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (60, 60), 40, 1, -1)  # solid blob
    cv2.circle(mask, (60, 60), 12, 0, -1)  # hole punched in the blob
    cv2.rectangle(mask, (120, 120), (170, 170), 1, -1)  # separate solid blob

    pixel_size = 0.5
    gdf = mask_to_gdf(mask, pixel_size=pixel_size)

    assert len(gdf) == 2
    assert list(gdf["tissue_id"]) == [0, 1]

    # coordinates must be scaled by pixel_size: mask is 200px wide -> bounds <= 100
    minx, miny, maxx, maxy = gdf.total_bounds
    assert 0 <= minx and maxx <= 200 * pixel_size
    assert 0 <= miny and maxy <= 200 * pixel_size

    n_holes = [len(list(geom.interiors)) for geom in gdf.geometry]
    assert sorted(n_holes) == [0, 1]


def test_mask_to_gdf_empty_mask_returns_empty_geodataframe() -> None:
    mask = np.zeros((50, 50), dtype=np.uint8)
    gdf = mask_to_gdf(mask, pixel_size=1.0)
    assert len(gdf) == 0


# --------------------------------------------------------------------- #
# Table validation
# --------------------------------------------------------------------- #


def test_fix_table_validation_errors_renames_invalid_var_columns(adata) -> None:
    from spatialdata._core.validation import ValidationError, validate_table_attr_keys

    from spatialrefinery.core.utils import fix_table_validation_errors

    adata.var_names = ["gene1", "gene2"]
    adata.var["bad col"] = [1, 2]  # a space is not a valid SpatialData column name

    # Sanity check: confirm this is actually the failure `fix_table_validation_errors` fixes.
    with pytest.raises(ValidationError):
        validate_table_attr_keys(adata)

    fixed = fix_table_validation_errors(adata)

    assert "bad col" not in fixed.var.columns
    assert "bad_col" in fixed.var.columns
    validate_table_attr_keys(fixed)  # must no longer raise
