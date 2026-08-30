"""Tests for `spatialrefinery.io.xenium`'s reader-facing helpers.

Covers the H&E channel-layout probe, which exists because
`spatialdata_io.xenium_aligned_image` infers the axis order from the array
shape alone and asserts on anything that is not 10x's interleaved layout.
Slides are synthesised as small OME-TIFFs written in each layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile as tf

from spatialrefinery.io.xenium import _aligned_image_dims


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
