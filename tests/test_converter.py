"""Tests for `spatialrefinery.core.converter`'s OME-TIFF write path.

The point of interest is that neither `OpenSlideImageConverter` nor
`SlideioImageConverter` materialises the level-0 plane: both stream bands
off the slide and stage sub-levels in on-disk memmaps. These tests pin that
streamed pyramid to the pixels an in-memory reference path produces, since a
whole-slide image large enough to *need* streaming is far too large to keep
in a test.

Slides are synthesised as generic tiled TIFFs, which openslide reads and
which `slideio`'s GDAL driver reads for the RGB case. `slideio`'s GDAL/OpenCV
backend cannot read a tiled TIFF with a true multi-sample-per-pixel plane
(`PlanarConfig=2`), which is how the only way to write >1 non-RGB channel
into one page; `SlideioTiledSource`'s `"minisblack"` path is exercised
instead against a minimal fake `Scene` (`FakeScene` below), covering exactly
the same `size`/`num_channels`/`get_channel_data_type`/`read_block`/`close`
surface a real `slideio.Scene` exposes.
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
    SlideioImageConverter,
    SlideioTiledSource,
    _resolve_mpp,
    downsample_plane,
    write_ome_tiff,
)
from spatialrefinery.core.registry import list_converters

openslide = pytest.importorskip("openslide")

SUBRESOLUTIONS = 3


def make_slide(path, height, width, seed=0):
    """Write a random RGB image as a tiled TIFF openslide (and slideio's GDAL driver) can open."""
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
    # This synthetic TIFF carries no mpp-x/mpp-y property, which OpenSlideImageConverter
    # now raises on rather than silently assuming 1.0 -- see test_missing_mpp_raises.
    OpenSlideImageConverter(subresolutions=SUBRESOLUTIONS, tile_size=tile_size, mpp=0.5).convert(slide_path, streamed)

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


def test_missing_mpp_raises(tmp_path):
    """A source with no detectable pixel size must fail rather than default to 1.0.

    Silently writing a slide at an assumed scale is exactly the "silent bad
    sample" `CLAUDE.md` warns against -- see `_resolve_mpp`.
    """
    slide_path = tmp_path / "slide.tif"
    make_slide(slide_path, 256, 256)

    with pytest.raises(ValueError, match="mpp"):
        OpenSlideImageConverter(subresolutions=1).convert(slide_path, tmp_path / "out")


class TestResolveMpp:
    """Unit tests for `_resolve_mpp`'s detected/override/missing logic."""

    def test_detected_used_when_no_override(self):
        """A detected pixel size passes through unchanged."""
        assert _resolve_mpp((0.25, 0.26), None, Path("x")) == (0.25, 0.26)

    def test_scalar_override_applies_to_both_axes(self):
        """A single override value is used for both X and Y, regardless of what was detected."""
        assert _resolve_mpp((0.25, 0.26), 0.5, Path("x")) == (0.5, 0.5)

    def test_tuple_override_wins_over_detected(self):
        """A two-value override sets X and Y independently."""
        assert _resolve_mpp((0.25, 0.26), (0.1, 0.2), Path("x")) == (0.1, 0.2)

    def test_missing_detected_and_no_override_raises(self):
        """No detected value and no override must fail rather than assume a scale."""
        with pytest.raises(ValueError, match="mpp"):
            _resolve_mpp(None, None, Path("slide.vsi"))

    @pytest.mark.parametrize(
        "detected",
        [
            (0.0, 0.5),  # zero X
            (0.5, -1.0),  # negative Y
        ],
    )
    def test_non_positive_detected_raises(self, detected):
        """A detected value that isn't a real physical size must also raise."""
        with pytest.raises(ValueError, match="mpp"):
            _resolve_mpp(detected, None, Path("slide.vsi"))

    def test_override_bypasses_a_bad_detected_value(self):
        """The override is the escape hatch even when detection produced nonsense."""
        assert _resolve_mpp((0.0, -1.0), 0.25, Path("x")) == (0.25, 0.25)


def test_downsample_plane_single_channel_minisblack():
    """A single-channel `"minisblack"` plane must halve without swapping height and width.

    Regression test: `cv2.resize` silently drops a trailing size-1 channel
    axis (`(H, W, 1) -> (H, W)`), and the previous `moveaxis(img, -1, 0)` then
    transposed that now-2D result instead of restoring the channel axis --
    invisible in this repo until now because the one existing minisblack
    caller (`BioioImageConverter`) always `np.squeeze`s a single channel down
    to plain `(H, W)` before it reaches `downsample_plane`.
    """
    rng = np.random.default_rng(0)
    image = rng.integers(0, 4000, (1, 100, 60), dtype=np.uint16)  # H != W to catch a swap

    halved = downsample_plane(image, "minisblack", 0.5)

    assert halved.shape == (1, 50, 30)


def test_new_formats_registered_without_disturbing_existing_ones():
    """`.vsi`/`.qptiff`/`.zvi`/`.afi` route to `SlideioImageConverter`; nothing else moves."""
    converters = list_converters()
    for suffix in (".vsi", ".qptiff", ".zvi", ".afi"):
        assert converters[suffix] == "SlideioImageConverter"

    # DICOM WSI is normally a directory of instances, which suffix dispatch
    # doesn't address, and it has not been exercised against a real file.
    assert ".dcm" not in converters

    for suffix in (".svs", ".ndpi", ".tif", ".tiff", ".mrxs", ".scn", ".bif", ".vms", ".svslide"):
        assert converters[suffix] == "OpenSlideImageConverter"


@pytest.mark.parametrize(
    ("height", "width", "tile_size"),
    [
        (512, 768, 256),
        (700, 1100, 256),
    ],
)
def test_slideio_streamed_rgb_pyramid_matches_in_memory(tmp_path, height, width, tile_size):
    """`SlideioTiledSource` streaming an RGB scene must match halving it in memory.

    Uses `slideio`'s own GDAL driver to open a synthetic tiled TIFF, so this
    exercises the real `slideio.Scene` API, not a stand-in.
    """
    slide_path = tmp_path / "slide.tif"
    image = make_slide(slide_path, height, width)

    streamed = tmp_path / "streamed"
    SlideioImageConverter(subresolutions=SUBRESOLUTIONS, tile_size=tile_size, mpp=0.5).convert(slide_path, streamed)

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
    assert np.array_equal(got[0], image)
    for index, (actual, expected) in enumerate(zip(got, want, strict=True)):
        assert np.array_equal(actual, expected), f"level {index} differs"


def test_slideio_missing_mpp_raises(tmp_path):
    """`SlideioImageConverter` must also refuse a scene with no usable pixel size."""
    slide_path = tmp_path / "slide.tif"
    make_slide(slide_path, 256, 256)

    with pytest.raises(ValueError, match="mpp"):
        SlideioImageConverter(subresolutions=1).convert(slide_path, tmp_path / "out")


class FakeScene:
    """A minimal stand-in for `slideio.Scene`, covering only what `SlideioTiledSource` calls.

    `slideio`'s GDAL/OpenCV backend cannot read back a tiled TIFF with a true
    multi-sample-per-pixel plane (see the module docstring), so the
    `"minisblack"` path is tested against this fake instead of a real
    `slideio.Scene` -- it is backed by a plain in-memory `(C, H, W)` array and
    implements `size`, `num_channels`, `get_channel_data_type`,
    `read_block(rect, channel_indices=...)`, and `close`, matching the real
    API surface `SlideioTiledSource` relies on.
    """

    def __init__(self, image: np.ndarray, dtype_name: str, name: str = "fake") -> None:
        self._image = image  # (C, H, W)
        self._dtype_name = dtype_name
        self.name = name
        self.num_channels = image.shape[0]
        self.num_z_slices = 1
        self.num_t_frames = 1
        self.resolution = (0.3e-6, 0.3e-6)
        self.closed = False

    @property
    def size(self):
        """`(width, height)`, matching `slideio.Scene.size`."""
        _, height, width = self._image.shape
        return (width, height)

    def get_channel_data_type(self, channel: int) -> str:
        """Every channel shares one dtype in this fake, as in the real scenes tested here."""
        return self._dtype_name

    def get_channel_name(self, channel: int) -> str:
        """No real slideio scene tested here reports channel names either."""
        return ""

    def read_block(self, rect, channel_indices=None):
        """Slice `self._image` the way `slideio.Scene.read_block` would."""
        x, y, w, h = rect
        if channel_indices is not None:
            return self._image[channel_indices[0], y : y + h, x : x + w]
        return np.moveaxis(self._image[:, y : y + h, x : x + w], 0, -1)

    def close(self) -> None:
        """Record that `close` was called; idempotent, like the real `Scene.close`."""
        self.closed = True


def per_channel_reference(image: np.ndarray, subresolutions: int) -> list[np.ndarray]:
    """Halve each channel of a `(C, H, W)` array independently, stacked back into `(C, h, w)`.

    This -- not a single `ArrayPyramidSource` call on the whole `(C, H, W)`
    array -- is the correct oracle for `SlideioTiledSource`'s `"minisblack"`
    output: `tifffile` requires a `"minisblack"` tile iterator to walk
    page-major (every tile of channel 0, then channel 1, ...), so
    `SlideioTiledSource` downsamples each channel with its own independent
    `cv2.resize` call rather than one call across all channels at once.
    `cv2.resize` is not required to be bit-identical between those two
    call patterns (and empirically is not, by 1-2 counts out of ~4000 on a
    `uint16` test array) -- an inherent, harmless consequence of the
    per-channel-page write, not a bug.
    """
    sources = [ArrayPyramidSource(image[c], "minisblack") for c in range(image.shape[0])]
    levels = []
    for level in range(subresolutions + 1):
        planes = [np.asarray(s.level_data(level, 0)) for s in sources]
        levels.append(np.stack(planes, axis=0))
    return levels


@pytest.mark.parametrize(
    ("channels", "height", "width", "tile_size"),
    [
        (1, 300, 260, 128),  # single channel: exercises the downsample_plane fix above
        (2, 512, 768, 256),
        (4, 700, 1100, 256),  # opencv's resize caps at 4 channels; see per_channel_reference
    ],
)
def test_slideio_streamed_minisblack_pyramid_matches_per_channel_reference(
    tmp_path, channels, height, width, tile_size
):
    """`SlideioTiledSource` streaming a multichannel scene must match `per_channel_reference`."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 4000, (channels, height, width), dtype=np.uint16)
    scene = FakeScene(image, "uint16")

    source = SlideioTiledSource(scene, subresolutions=SUBRESOLUTIONS)
    assert source.photometric == "minisblack"
    try:
        streamed_path = write_ome_tiff(
            tmp_path / "streamed",
            source,
            {"PhysicalSizeX": 0.3, "PhysicalSizeY": 0.3},
            "minisblack",
            subresolutions=SUBRESOLUTIONS,
            tile_size=tile_size,
        )
    finally:
        source.close()
    assert scene.closed

    got = [level.reshape(1, *level.shape) if level.ndim == 2 else level for level in levels_of(streamed_path)]
    want = per_channel_reference(image, SUBRESOLUTIONS)

    assert [g.shape for g in got] == [w.shape for w in want]
    for index, (actual, expected) in enumerate(zip(got, want, strict=True)):
        assert np.array_equal(actual, expected), f"level {index} differs"
