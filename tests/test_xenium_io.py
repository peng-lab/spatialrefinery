"""Tests for `spatialrefinery.io.xenium`'s reader-facing helpers.

Covers the H&E channel-layout probe, which exists because
`spatialdata_io.xenium_aligned_image` infers the axis order from the array
shape alone and asserts on anything that is not 10x's interleaved layout.
Slides are synthesised as small OME-TIFFs written in each layout.

Also covers bundle verification, whose Xenium shape is the inverse of
Visium's: the payload arrives inside `*_outs.zip` with unprefixed member
names, so kind classification only ever sees `outs` and the real
completeness signal is the extracted-member check. Bundles are synthesised
as empty files, since verification reads names rather than contents.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import tifffile as tf

from spatialrefinery.io.xenium import XeniumDownloader, _aligned_image_dims


def _write_rgb(path: Path, planar: bool) -> Path:
    """Write a small RGB OME-TIFF in either planar or interleaved layout."""
    image = np.random.default_rng(0).integers(0, 255, (3, 64, 96), dtype=np.uint8)
    if planar:
        tf.imwrite(path, image, photometric="rgb", planarconfig="separate", metadata={"axes": "SYX"})
    else:
        tf.imwrite(
            path,
            np.moveaxis(image, 0, -1),
            photometric="rgb",
            planarconfig="contig",
            metadata={"axes": "YXS"},
        )
    return path


def test_aligned_image_dims_planar_returns_explicit_dims(tmp_path: Path) -> None:
    """`planarconfig=SEPARATE` reads back as (1, c, y, x), which the reader mis-parses."""
    path = _write_rgb(tmp_path / "planar.ome.tif", planar=True)

    from dask_image.imread import imread

    assert imread(str(path)).shape == (1, 3, 64, 96)
    assert _aligned_image_dims(str(path)) == ("dummy", "c", "y", "x")


def test_aligned_image_dims_interleaved_defers_to_the_reader(tmp_path: Path) -> None:
    """10x's own layout is (1, y, x, c); the reader already handles it, so return None."""
    path = _write_rgb(tmp_path / "interleaved.ome.tif", planar=False)

    from dask_image.imread import imread

    assert imread(str(path)).shape == (1, 64, 96, 3)
    assert _aligned_image_dims(str(path)) is None


def test_aligned_image_dims_rejects_an_unrecognisable_layout(tmp_path: Path) -> None:
    """A single-channel image has no channel axis to find; fail loudly, not with an assert."""
    path = tmp_path / "grey.ome.tif"
    tf.imwrite(path, np.zeros((64, 96), dtype=np.uint8))

    with pytest.raises(ValueError, match="Cannot infer the channel axis"):
        _aligned_image_dims(str(path))


def _tiny_sdata(spot_size: float = 55.0, overlap: float = 0.06):
    """A minimal SpatialData with transcripts and a matching hex-spot element."""
    import dask.dataframe as dd
    import pandas as pd
    import spatialdata as sd
    from spatialdata.models import PointsModel, ShapesModel

    from spatialrefinery.core.utils import create_hexagonal_spots, hex_lattice_params, xy_bounds

    rng = np.random.default_rng(4)
    n = 5_000
    genes = ["GeneA", "GeneB", "GeneC"]
    frame = pd.DataFrame(
        {
            "x": rng.uniform(0.0, 400.0, n),
            "y": rng.uniform(0.0, 400.0, n),
            "feature_name": pd.Categorical(rng.choice(genes, n), categories=genes),
        }
    )
    points = PointsModel.parse(dd.from_pandas(frame, npartitions=3))

    bounds = xy_bounds(points, "x", "y")
    lattice = hex_lattice_params(bounds, spot_size, overlap)
    spots = ShapesModel.parse(create_hexagonal_spots(points, spot_size_um=spot_size, overlap=overlap, bounds=bounds))
    sdata = sd.SpatialData(points={"transcripts": points}, shapes={"spots_55um": spots})
    return sdata, lattice, overlap


def test_aggregate_transcripts_hex_returns_a_table_annotating_the_spots() -> None:
    """The table must be linked to its spots element, not just carry the right counts.

    Regression test: returning a bare `AnnData` left the counts correct but
    unattached, so consumers reported the spots element as having no annotating
    tables. `SpatialData.aggregate`, which this replaced, returned a parsed one.
    """
    from spatialrefinery.io.xenium import _aggregate_transcripts_hex

    sdata, lattice, overlap = _tiny_sdata()
    table = _aggregate_transcripts_hex(sdata, "spots_55um", lattice, overlap)

    assert table.uns["spatialdata_attrs"] == {
        "region": "spots_55um",
        "region_key": "region",
        "instance_key": "instance_id",
    }
    assert list(table.obs["region"].cat.categories) == ["spots_55um"]
    assert np.array_equal(table.obs["instance_id"].to_numpy(), sdata["spots_55um"].index.to_numpy())
    assert table.obs.index.equals(sdata["spots_55um"].index.astype(str))

    # and it must survive being attached to the SpatialData object
    sdata["spots_55um_table"] = table
    assert "spots_55um_table" in sdata.tables


def test_aggregate_transcripts_hex_counts_every_transcript() -> None:
    """Counts must be conserved: each transcript lands in >=1 spot, overlaps included."""
    from spatialrefinery.core.utils import assign_points_to_hexes
    from spatialrefinery.io.xenium import _aggregate_transcripts_hex

    sdata, lattice, overlap = _tiny_sdata()
    table = _aggregate_transcripts_hex(sdata, "spots_55um", lattice, overlap)

    frame = sdata["transcripts"][["x", "y"]].compute()
    positions, _ = assign_points_to_hexes(frame["x"].to_numpy(), frame["y"].to_numpy(), lattice, overlap)
    assert table.X.sum() == len(positions)


def test_aggregate_transcripts_hex_raises_when_nothing_is_counted() -> None:
    """An all-zero result means a broken mapping; it must not be written silently."""
    from spatialrefinery.core.utils import hex_lattice_params
    from spatialrefinery.io.xenium import _aggregate_transcripts_hex

    sdata, _, overlap = _tiny_sdata()
    # a lattice somewhere else entirely: no transcript can fall in any spot
    elsewhere = hex_lattice_params((1e6, 1e6, 1e6 + 400.0, 1e6 + 400.0), 55.0, overlap)
    with pytest.raises(ValueError, match="produced no counts at all"):
        _aggregate_transcripts_hex(sdata, "spots_55um", elsewhere, overlap)


# --------------------------------------------------------------------- #
# Bundle verification
# --------------------------------------------------------------------- #

STUDY = "Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon"

#: The members `spatialdata_io.xenium` opens, i.e. `XeniumDownloader.required_members`.
CONVERTER_MEMBERS = (
    "experiment.xenium",
    "transcripts.parquet",
    "cells.parquet",
    "cell_boundaries.parquet",
    "nucleus_boundaries.parquet",
    "cell_feature_matrix.h5",
)


def _make_bundle(root: Path, *, members: tuple[str, ...] = CONVERTER_MEMBERS, extras: tuple[str, ...] = ()) -> Path:
    """Create a bundle directory holding empty files; verification reads names, not contents."""
    bundle = root / STUDY
    bundle.mkdir(parents=True, exist_ok=True)
    for name in (*members, *extras):
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return bundle


def test_verify_bundle_accepts_an_extracted_bundle(tmp_path) -> None:
    """The six members the converter opens are enough; the .csv.gz/.zarr.zip twins are not required.

    Those duplicates get pruned to reclaim disk on a shared filesystem, so a
    slimmed bundle that still converts must not be reported broken.
    """
    bundle = _make_bundle(tmp_path, extras=(f"{STUDY}_outs.zip", "gene_panel.json", "morphology.ome.tif"))
    check = XeniumDownloader.verify_bundle(bundle)
    assert check.ok
    assert check.missing_required == ()
    assert check.missing_members == ()


def test_verify_bundle_detects_an_unextracted_outs_zip(tmp_path) -> None:
    """A downloaded `_outs.zip` that never unpacked is the failure kinds cannot see.

    Xenium's payload arrives *inside* the archive, so `present` reports
    `outs` either way -- only the member check distinguishes them. Xenium does
    not set `require_extract`, so a failed extraction otherwise only warns.
    """
    bundle = _make_bundle(tmp_path, members=(), extras=(f"{STUDY}_outs.zip",))
    check = XeniumDownloader.verify_bundle(bundle)
    assert check.present == frozenset({"outs"})
    assert check.missing_required == ()
    assert set(check.missing_members) == set(CONVERTER_MEMBERS)
    assert not check.ok


def test_verify_bundle_reports_a_missing_outs_asset(tmp_path) -> None:
    """`outs` is the one required *kind*: without the archive nothing else can exist."""
    check = XeniumDownloader.verify_bundle(_make_bundle(tmp_path, members=()))
    assert check.missing_required == ("outs",)


def test_verify_bundle_accepts_morphology_focus_as_a_directory(tmp_path) -> None:
    """10x ships `morphology_focus` as a flat OME-TIFF in older bundles and a tile directory in newer ones.

    33 of 65 reference bundles use the file form and 32 the directory form, so
    a member glob that only matched files would misreport half the corpus.
    """
    bundle = _make_bundle(tmp_path, extras=(f"{STUDY}_outs.zip", "gene_panel.json", "morphology.ome.tif"))
    (bundle / "morphology_focus").mkdir()
    (bundle / "morphology_focus" / "morphology_focus_0000.ome.tif").touch()
    assert "morphology_focus*" not in XeniumDownloader.verify_bundle(bundle).missing_expected


def test_verify_bundle_does_not_require_an_he_image(tmp_path) -> None:
    """10 of the 65 reference bundles ship no H&E; they convert fine without one.

    Reported through `post_process`, never through `missing_required` -- and
    deliberately not through `expected_kinds` either, since 3 bundles ship
    `_he_unaligned_image.ome.tif` and so lack the `he_image` kind while
    holding a perfectly good image.
    """
    bundle = _make_bundle(tmp_path, extras=(f"{STUDY}_outs.zip", "gene_panel.json", "morphology.ome.tif"))
    check = XeniumDownloader.verify_bundle(bundle)
    assert check.ok
    assert "he_image" not in check.missing_expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("study_he_imagealignment.csv", "he_alignment"),  # 10x's own name
        # The Atera "WTA Preview" sets ship this instead. Missing it from the
        # kind map made `kinds=["he_alignment"]` silently skip those files.
        ("study_he_alignment.csv", "he_alignment"),
        ("study_if_imagealignment.csv", "if_alignment"),
        ("study_he_unaligned_image.ome.tif", "he_unaligned_image"),
    ],
)
def test_classify_covers_alignment_naming_variants(filename: str, expected: str) -> None:
    """Both alignment-CSV spellings must classify, or one pipeline's bundles filter out."""
    assert XeniumDownloader.classify(filename) == expected


def test_post_process_flags_an_he_without_an_alignment_file(tmp_path, caplog) -> None:
    """An H&E with no alignment CSV lands on an Identity transform, silently unaligned.

    3 of the 65 reference bundles are in exactly this state. It downloads and
    converts without error, so nothing else catches it.
    """
    _make_bundle(tmp_path, extras=(f"{STUDY}_outs.zip", f"{STUDY}_he_image.ome.tif"))
    with caplog.at_level(logging.ERROR):
        XeniumDownloader(tmp_path).post_process(STUDY, [])
    assert "no alignment CSV" in caplog.text
    assert "Identity transform" in caplog.text


def test_post_process_is_quiet_about_a_correctly_aligned_bundle(tmp_path, caplog) -> None:
    """A complete bundle with both an H&E and its alignment CSV must not warn at all."""
    bundle = _make_bundle(
        tmp_path,
        extras=(
            f"{STUDY}_outs.zip",
            f"{STUDY}_he_image.ome.tif",
            f"{STUDY}_he_imagealignment.csv",
            "gene_panel.json",
            "morphology.ome.tif",
        ),
    )
    (bundle / "morphology_focus").mkdir()
    with caplog.at_level(logging.WARNING):
        XeniumDownloader(tmp_path).post_process(STUDY, [])
    assert caplog.text == ""
