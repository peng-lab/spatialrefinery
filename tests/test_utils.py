"""Tests for `spatialrefinery.core.utils`.

Only exercises the dependency-light surface (pure helpers, spot-grid
geometry, and the mask->polygon extraction used by `segment_tissue`) so this
file runs fast and offline; `segment_tissue` itself needs a real
openslide-readable image on disk and is left to the end-to-end scripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spatialrefinery.core.utils import (
    _mask_to_gdf as mask_to_gdf,
)
from spatialrefinery.core.utils import (
    assign_points_to_hexes,
    bin_centroids,
    bin_points_to_hex_counts,
    hex_candidate_offsets,
    hex_grid_centroids,
    hex_lattice_params,
    human_bytes,
    parse_curl_manifest,
    safe_extract_zip,
    slide_stem,
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


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        # The case this helper exists for: `Path.stem` would leave "slide.ome".
        ("slide.ome.tif", "slide"),
        ("slide.ome.tiff", "slide"),
        ("slide.OME.TIF", "slide"),
        ("slide.ome.zarr", "slide"),
        ("slide.svs", "slide"),
        ("slide.tif", "slide"),
        ("slide.NDPI", "slide"),
        # Only the extension goes; dots inside the name are part of it.
        ("Xenium_V1.2_section.ome.tif", "Xenium_V1.2_section"),
        ("a.b.svs", "a.b"),
        ("no_extension", "no_extension"),
    ],
)
def test_slide_stem(filename: str, expected: str) -> None:
    assert slide_stem(filename) == expected
    assert slide_stem(Path("/some/dir") / filename) == expected


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
# Analytic hex membership (replaces the shapely sjoin in the transcript
# aggregation -- these are the tests that pin it to the old semantics)
# --------------------------------------------------------------------- #


def _shapely_reference(x, y, bounds, spot_size_um, overlap):
    """Ground truth: build the polygons and sjoin, exactly as before."""
    geopandas = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely.geometry")

    from spatialrefinery.core.utils import hexagon_vertices

    centroids = hex_grid_centroids(bounds, spot_size_um, overlap)
    hexes = geopandas.GeoDataFrame(
        geometry=[shapely.Polygon(hexagon_vertices(cx, cy, spot_size_um / 2.0)) for cx, cy in centroids]
    )
    points = geopandas.GeoDataFrame(geometry=geopandas.points_from_xy(x, y))
    joined = hexes.sjoin(points)
    return set(zip(joined.index_right.tolist(), joined.index.tolist(), strict=True))


@pytest.mark.parametrize(
    ("spot_size_um", "overlap"),
    [(55.0, 0.0), (55.0, 0.06), (100.0, 0.06), (55.0, 0.25), (30.0, 0.4)],
)
def test_assign_points_to_hexes_matches_shapely_sjoin(spot_size_um: float, overlap: float) -> None:
    """The analytic test must reproduce `sjoin` pair-for-pair.

    This is the contract that lets `_aggregate_transcripts_hex` stand in for
    `SpatialData.aggregate`: same pairs means same counts, including the
    duplicates that overlapping hexagons produce.
    """
    bounds = (0.0, 0.0, 600.0, 600.0)
    rng = np.random.default_rng(11)
    x = rng.uniform(-spot_size_um, 600.0 + spot_size_um, 20_000)
    y = rng.uniform(-spot_size_um, 600.0 + spot_size_um, 20_000)

    lattice = hex_lattice_params(bounds, spot_size_um, overlap)
    positions, spot_ids = assign_points_to_hexes(x, y, lattice, overlap)

    assert set(zip(positions.tolist(), spot_ids.tolist(), strict=True)) == _shapely_reference(
        x, y, bounds, spot_size_um, overlap
    )


def test_assign_points_to_hexes_covers_the_plane_without_overlap() -> None:
    """With overlap=0 the hexagons tile, so every interior point lands in exactly one."""
    bounds = (0.0, 0.0, 400.0, 400.0)
    lattice = hex_lattice_params(bounds, 55.0, 0.0)
    rng = np.random.default_rng(3)
    x = rng.uniform(50.0, 350.0, 5_000)
    y = rng.uniform(50.0, 350.0, 5_000)

    positions, _ = assign_points_to_hexes(x, y, lattice, 0.0)
    assert np.array_equal(np.bincount(positions, minlength=5_000), np.ones(5_000, dtype=int))


def test_assign_points_to_hexes_overlap_assigns_some_points_twice() -> None:
    """Overlapping spots genuinely double-count, which the sjoin also did."""
    bounds = (0.0, 0.0, 400.0, 400.0)
    lattice = hex_lattice_params(bounds, 55.0, 0.06)
    rng = np.random.default_rng(5)
    x = rng.uniform(50.0, 350.0, 5_000)
    y = rng.uniform(50.0, 350.0, 5_000)

    positions, _ = assign_points_to_hexes(x, y, lattice, 0.06)
    multiplicity = np.bincount(positions, minlength=5_000)
    assert multiplicity.min() >= 1
    assert multiplicity.max() > 1


def test_hex_grid_centroids_matches_lattice_params_indexing() -> None:
    """Spot id `i * nx + j` must address the centroid the lattice describes."""
    bounds = (-13.0, 7.5, 500.0, 300.0)
    lattice = hex_lattice_params(bounds, 55.0, 0.06)
    centroids = hex_grid_centroids(bounds, 55.0, 0.06)

    assert len(centroids) == lattice["nx"] * lattice["ny"]
    for row in (0, 1, lattice["ny"] - 1):
        for col in (0, lattice["nx"] - 1):
            expected_x = lattice["x_range"][col] + (lattice["dx"] / 2 if row % 2 else 0.0)
            assert centroids[row * lattice["nx"] + col] == pytest.approx((expected_x, lattice["y_range"][row]))


@pytest.mark.parametrize(
    ("overlap", "rows", "cols"),
    [(0.0, [0, 1], [0, 1]), (0.06, [0, 1], [0, 1]), (0.4, [-1, 0, 1, 2], [0, 1])],
)
def test_hex_candidate_offsets_widen_with_overlap(overlap, rows, cols) -> None:
    row_offsets, col_offsets = hex_candidate_offsets(overlap)
    assert list(row_offsets) == rows
    assert list(col_offsets) == cols


def test_bin_points_to_hex_counts_matches_a_dense_reference() -> None:
    """Streaming in batches must give the same matrix as counting in one go."""
    bounds = (0.0, 0.0, 300.0, 300.0)
    overlap, spot_size, n_genes = 0.06, 55.0, 7
    lattice = hex_lattice_params(bounds, spot_size, overlap)
    n_spots = lattice["nx"] * lattice["ny"]

    rng = np.random.default_rng(17)
    x = rng.uniform(0.0, 300.0, 30_000)
    y = rng.uniform(0.0, 300.0, 30_000)
    codes = rng.integers(0, n_genes, 30_000)

    counts = bin_points_to_hex_counts(
        ((x[s], y[s], codes[s]) for s in (slice(0, 7_000), slice(7_000, 21_000), slice(21_000, None))),
        lattice,
        n_spots=n_spots,
        n_genes=n_genes,
        overlap=overlap,
    )

    expected = np.zeros((n_spots, n_genes), dtype=np.int64)
    positions, spot_ids = assign_points_to_hexes(x, y, lattice, overlap)
    np.add.at(expected, (spot_ids, codes[positions]), 1)

    assert np.array_equal(counts.toarray(), expected)
    assert counts.sum() == len(positions)


def test_bin_points_to_hex_counts_flushes_without_changing_the_result() -> None:
    """`flush_pairs` bounds the merge buffer only; it must not alter counts."""
    bounds = (0.0, 0.0, 200.0, 200.0)
    lattice = hex_lattice_params(bounds, 55.0, 0.06)
    n_spots = lattice["nx"] * lattice["ny"]

    rng = np.random.default_rng(23)
    batches = [(rng.uniform(0, 200, 4_000), rng.uniform(0, 200, 4_000), rng.integers(0, 5, 4_000)) for _ in range(5)]

    unflushed = bin_points_to_hex_counts(iter(batches), lattice, n_spots, 5, 0.06, flush_pairs=10**9)
    flushed = bin_points_to_hex_counts(iter(batches), lattice, n_spots, 5, 0.06, flush_pairs=100)
    assert (unflushed != flushed).nnz == 0


def test_bin_points_to_hex_counts_drops_negative_gene_codes() -> None:
    """A -1 code is how an unmapped feature is skipped, not counted into gene 0."""
    bounds = (0.0, 0.0, 200.0, 200.0)
    lattice = hex_lattice_params(bounds, 55.0, 0.0)
    n_spots = lattice["nx"] * lattice["ny"]

    x = np.array([100.0, 100.0, 100.0])
    y = np.array([100.0, 100.0, 100.0])
    kept = bin_points_to_hex_counts([(x, y, np.array([0, 0, 0]))], lattice, n_spots, 3, 0.0)
    dropped = bin_points_to_hex_counts([(x, y, np.array([0, -1, -1]))], lattice, n_spots, 3, 0.0)

    assert kept.sum() == 3
    assert dropped.sum() == 1


def test_bin_points_to_hex_counts_empty_input_returns_empty_matrix() -> None:
    lattice = hex_lattice_params((0.0, 0.0, 200.0, 200.0), 55.0, 0.06)
    n_spots = lattice["nx"] * lattice["ny"]
    counts = bin_points_to_hex_counts([], lattice, n_spots, 4, 0.06)
    assert counts.shape == (n_spots, 4)
    assert counts.nnz == 0


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


# --------------------------------------------------------------------- #
# Points: bytes-column decoding
# --------------------------------------------------------------------- #


def _bytes_points():
    """A points frame shaped like an older Xenium `transcripts.parquet`.

    `cell_id` / `fov_name` hold `bytes` under a bare `object` dtype, which is
    what the "Preview" / "With_Addon" bundles produce.

    `convert-string` is switched off while the frame is built because
    `dd.from_pandas` otherwise re-types object columns to `StringDtype` and the
    bug disappears. The real frames come from `dd.read_parquet` with a declared
    object-dtype meta, which is what this reproduces.
    """
    import dask
    import dask.dataframe as dd
    import pandas as pd
    from spatialdata.models import PointsModel

    df = pd.DataFrame(
        {
            "x": np.array([1.0, 2.0], dtype="float32"),
            "y": np.array([3.0, 4.0], dtype="float32"),
            "cell_id": np.array([b"UNASSIGNED", b"abcd-1"], dtype=object),
            "fov_name": np.array([b"A5", b"B7"], dtype=object),
        }
    )
    with dask.config.set({"dataframe.convert-string": False}):
        return PointsModel.parse(dd.from_pandas(df, npartitions=1))


def test_decode_bytes_columns_makes_points_parquet_writable() -> None:
    """The decoded frame's dummy meta must be inferable by pyarrow.

    This is the actual failure mode: `to_parquet` types the schema from
    `meta_nonempty`, which fills a bare `object` column with an `object()`
    sentinel that pyarrow rejects.
    """
    import pyarrow as pa
    from dask.dataframe.utils import meta_nonempty

    from spatialrefinery.core.utils import decode_bytes_columns

    points = _bytes_points()

    with pytest.raises(pa.ArrowInvalid):
        pa.Schema.from_pandas(meta_nonempty(points._meta), preserve_index=False)

    decoded = decode_bytes_columns(points)

    pa.Schema.from_pandas(meta_nonempty(decoded._meta), preserve_index=False)


def test_decode_bytes_columns_decodes_values_and_keeps_transformations() -> None:
    """Values become real strings and the coordinate transforms survive."""
    from spatialdata.transformations import Scale, get_transformation, set_transformation

    from spatialrefinery.core.utils import decode_bytes_columns

    points = _bytes_points()
    set_transformation(points, Scale([2.0, 2.0], axes=("x", "y")), to_coordinate_system="global")
    before = get_transformation(points, get_all=True)

    decoded = decode_bytes_columns(points)
    result = decoded.compute()

    assert list(result["cell_id"]) == ["UNASSIGNED", "abcd-1"]
    assert list(result["fov_name"]) == ["A5", "B7"]
    assert str(get_transformation(decoded, get_all=True)) == str(before)


def test_decode_bytes_columns_no_object_columns_is_a_noop() -> None:
    """A frame with no `object` columns is returned untouched."""
    import dask.dataframe as dd
    import pandas as pd
    from spatialdata.models import PointsModel

    from spatialrefinery.core.utils import decode_bytes_columns

    points = PointsModel.parse(
        dd.from_pandas(
            pd.DataFrame({"x": np.array([1.0], dtype="float32"), "y": np.array([2.0], dtype="float32")}),
            npartitions=1,
        )
    )

    assert decode_bytes_columns(points) is points
