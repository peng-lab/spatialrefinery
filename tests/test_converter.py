"""Tests for `spatialrefinery.core.converter`'s OME-TIFF write path.

The point of interest is that `OpenSlideImageConverter` never materialises
the level-0 plane: it streams bands off the slide and stages sub-levels in
on-disk memmaps. These tests pin that streamed pyramid to the pixels the
in-memory `ArrayPyramidSource` path produces, since a whole-slide image
large enough to *need* streaming is far too large to keep in a test.

Slides are synthesised as generic tiled TIFFs, which openslide reads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile as tf

from spatialrefinery.core.converter import (
    ArrayPyramidSource,
    OpenSlideImageConverter,
    OpenSlideTiledSource,
    write_ome_tiff,
)

openslide = pytest.importorskip("openslide")

SUBRESOLUTIONS = 3


def make_slide(path, height, width, seed=0):
    """Write a random RGB image as a tiled TIFF openslide can open; return the pixels."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    with tf.TiffWriter(path) as writer:
        writer.write(image, tile=(256, 256), photometric="rgb", compression="zlib")
    return image


def levels_of(path):
    """Read back every pyramid level of an OME-TIFF as arrays."""
    with tf.TiffFile(path) as handle:
        return [level.asarray() for level in handle.series[0].levels]


@pytest.mark.parametrize(
    ("height", "width", "tile_size"),
    [
        (512, 768, 256),  # both dimensions a whole number of tiles
        (700, 1100, 256),  # ragged edge tiles in both directions
        (300, 260, 128),  # sub-levels fall below one tile
    ],
)
def test_streamed_pyramid_matches_in_memory(tmp_path, height, width, tile_size):
    """Streaming the slide must produce the same pixels as halving it in memory."""
    slide_path = tmp_path / "slide.tif"
    image = make_slide(slide_path, height, width)

    streamed = tmp_path / "streamed"
    OpenSlideImageConverter(subresolutions=SUBRESOLUTIONS, tile_size=tile_size).convert(slide_path, streamed)

    reference = tmp_path / "reference.ome.tif"
    write_ome_tiff(
        reference,
        image,
        {"PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5},
        "rgb",
        subresolutions=SUBRESOLUTIONS,
        tile_size=tile_size,
    )

    got = levels_of(streamed / "slide.ome.tif")
    want = levels_of(reference)

    assert [level.shape for level in got] == [level.shape for level in want]
    assert np.array_equal(got[0], image)  # level 0 is lossless
    for index, (actual, expected) in enumerate(zip(got, want, strict=True)):
        assert np.array_equal(actual, expected), f"level {index} differs"


def test_streamed_level_shapes_follow_floor_halving(tmp_path):
    """`level_shape` must match what successive `downsample_plane` calls produce."""
    slide_path = tmp_path / "slide.tif"
    image = make_slide(slide_path, 700, 1100)

    source = OpenSlideTiledSource(slide_path, subresolutions=SUBRESOLUTIONS)
    in_memory = ArrayPyramidSource(image, "rgb")
    try:
        for level in range(SUBRESOLUTIONS + 1):
            assert source.level_shape(level) == in_memory.level_shape(level)
    finally:
        source.close()


def test_close_releases_scratch_directory(tmp_path):
    """The staged sub-level memmaps must not outlive the source."""
    slide_path = tmp_path / "slide.tif"
    make_slide(slide_path, 512, 768)

    source = OpenSlideTiledSource(slide_path, subresolutions=SUBRESOLUTIONS)
    list(source.level_data(0, 256))  # drains level 0, which stages level 1

    scratch = source._scratch
    assert scratch is not None and Path(scratch.name).exists()

    source.close()
    assert not Path(scratch.name).exists()
    source.close()  # idempotent


def test_partially_drained_level_still_stages_the_next(tmp_path):
    """A consumer that stops at the last tile must still leave level 1 complete.

    `tifffile` pulls exactly the tile count implied by `shape`, so the
    generator is left suspended on its final `yield` and never runs the code
    after its loop -- which is why the halving happens before the yields.
    """
    slide_path = tmp_path / "slide.tif"
    image = make_slide(slide_path, 512, 768)

    source = OpenSlideTiledSource(slide_path, subresolutions=1)
    try:
        tiles = source.level_data(0, 256)
        expected_tiles = (512 // 256) * (768 // 256)
        for _ in range(expected_tiles):  # stop without exhausting the generator
            next(tiles)

        staged = np.asarray(source._levels[1])
        assert np.array_equal(staged, ArrayPyramidSource(image, "rgb").level_data(1, 0))
    finally:
        source.close()
