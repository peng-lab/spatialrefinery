"""Converters: image-format -> pyramidal OME-TIFF, and vendor bundle -> SpatialData zarr.

These two conversion shapes are deliberately kept as separate template
hierarchies under one two-method :class:`BaseConverter` (``supports`` /
``convert``), rather than forced into a single ``read/transform/write``
pipeline: OME-TIFF conversion is image-in, image-out and synchronous,
while a vendor bundle -> zarr conversion (see
``spatialrefinery.io.xenium.XeniumConverter``) is a heterogeneous
directory read incrementally into an on-disk store. Sharing more than the
capability check between them would only produce a template with escape
hatches.

Each pyramid level reaches :func:`write_ome_tiff` through a
:class:`PyramidSource`, which supplies a level either as a whole array
(:class:`ArrayPyramidSource`) or as an iterator of tiles
(:class:`OpenSlideTiledSource`). Whole-slide level-0 planes routinely
exceed RAM -- a 100k x 45k RGB slide is 12.5 GiB before openslide's RGBA
read buffer is counted -- so the openslide path never materialises one.
"""

from __future__ import annotations

import logging
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from spatialrefinery.core.registry import register_converter

logger = logging.getLogger(__name__)

DEFAULT_TILE_SIZE = 1024
DEFAULT_SUBRESOLUTIONS = 4

Photometric = Literal["rgb", "minisblack"]


# --------------------------------------------------------------------- #
# OME-TIFF writing (shared by every ImageConverter)
# --------------------------------------------------------------------- #


def img_resize(img: np.ndarray, scale_factor: float) -> np.ndarray:
    """Resize `img` by `scale_factor` using area interpolation."""
    import cv2

    width = int(np.floor(img.shape[1] * scale_factor))
    height = int(np.floor(img.shape[0] * scale_factor))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def downsample_plane(img: np.ndarray, photometric: Photometric, factor: float = 0.5) -> np.ndarray:
    """Halve a pyramid level, keeping a leading channel axis (CYX) in place for `minisblack`."""
    if photometric == "minisblack" and img.ndim >= 3 and img.shape[0] < img.shape[-1]:
        img = np.moveaxis(img, 0, -1)
        img = img_resize(img, factor)
        return np.moveaxis(img, -1, 0)
    return img_resize(img, factor)


def _ome_tiff_path(path: str | Path) -> Path:
    path = Path(path)
    if path.name.endswith((".ome.tif", ".ome.tiff")):
        return path
    return path.with_name(path.name + ".ome.tif")


# --------------------------------------------------------------------- #
# Pyramid levels: whole-array or streamed
# --------------------------------------------------------------------- #


class PyramidSource(ABC):
    """Supplies each pyramid level to :func:`write_ome_tiff`.

    `level_data` may return a whole array or an iterator of tiles; both are
    accepted by `tifffile.TiffWriter.write`, the latter needing an explicit
    `shape`/`dtype`, which is what `level_shape` and `dtype` are for. Tiles
    must arrive in row-major tile order; edge tiles are yielded at their true
    (short) size rather than padded, which `tifffile` accepts.
    """

    photometric: Photometric

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Element type of every level."""

    @abstractmethod
    def level_shape(self, level: int) -> tuple[int, ...]:
        """Shape of pyramid `level` (0 is full resolution)."""

    @abstractmethod
    def level_data(self, level: int, tile_size: int) -> np.ndarray | Iterator[np.ndarray]:
        """Pixel data for pyramid `level`, as an array or an iterator of tiles."""

    def close(self) -> None:  # noqa: B027 - only a streaming source holds anything to release
        """Release any handles or scratch files. Safe to call more than once."""


class ArrayPyramidSource(PyramidSource):
    """A plane already in memory; sub-levels are derived by successive halving."""

    def __init__(self, image: np.ndarray, photometric: Photometric) -> None:
        self._image = image
        self._current = image
        self._current_level = 0
        self.photometric = photometric

    @property
    def dtype(self) -> np.dtype:
        """Element type of every level."""
        return self._image.dtype

    def level_shape(self, level: int) -> tuple[int, ...]:
        """Shape of pyramid `level`, taken from the halved array itself."""
        return self.level_data(level, 0).shape

    def level_data(self, level: int, tile_size: int) -> np.ndarray:
        """Halve repeatedly up to `level`, caching so a full walk stays O(pixels).

        Levels are requested in ascending order, so the cache turns what would
        be a from-scratch downsample per level into one halving per level.
        """
        if level < self._current_level:  # only a re-request can go backwards
            self._current, self._current_level = self._image, 0
        while self._current_level < level:
            self._current = downsample_plane(self._current, self.photometric, 0.5)
            self._current_level += 1
        return self._current


class OpenSlideTiledSource(PyramidSource):
    """Streams an openslide level as tiles, holding at most one band in memory.

    Level 0 is read from the slide in full-width bands exactly `tile_size` tall,
    so each band is one row of the page's tile grid. Every band is halved on its
    way past and appended to an on-disk memmap that becomes level 1, and the same
    band-and-halve pass then runs down the pyramid -- so level 0 is decoded once
    no matter how many sub-levels are requested. Peak memory is a band plus its
    downsample (~420 MiB for a 100k-wide slide at the 1024 default); scratch disk
    is about a third of the level-0 size.

    Bands are halved *before* their tiles are yielded, so the memmap feeding the
    next level is complete as soon as the last band has been read. Deferring that
    to after the final `yield` would strand it: a consumer that pulls exactly the
    tile count implied by `shape` -- as `tifffile` does -- leaves the generator
    suspended on its last `yield` and never runs the code past the loop.
    """

    def __init__(
        self,
        path: Path,
        *,
        level: int = 0,
        subresolutions: int = DEFAULT_SUBRESOLUTIONS,
        read_chunk_cols: int = 8192,
    ) -> None:
        import openslide

        self._slide = openslide.OpenSlide(str(path))
        self._level = level
        self._subresolutions = subresolutions
        self._read_chunk_cols = read_chunk_cols
        self.photometric: Photometric = "rgb"

        width, height = self._slide.level_dimensions[level]
        self._shape0 = (height, width, 3)
        self._scratch: tempfile.TemporaryDirectory[str] | None = None
        self._levels: dict[int, np.memmap] = {}
        self._closed = False

    @property
    def properties(self) -> Any:
        """The underlying slide's openslide property map (mpp, vendor, and so on)."""
        return self._slide.properties

    @property
    def dtype(self) -> np.dtype:
        """Element type of every level; openslide always hands back 8-bit RGB."""
        return np.dtype(np.uint8)

    def level_shape(self, level: int) -> tuple[int, ...]:
        """Halve (flooring) `level` times, matching `img_resize`'s `floor(dim * 0.5)`."""
        height, width, samples = self._shape0
        for _ in range(level):
            height, width = height // 2, width // 2
        return (height, width, samples)

    def level_data(self, level: int, tile_size: int) -> Iterator[np.ndarray]:
        """Tiles of pyramid `level`, in row-major tile order."""
        return self._iter_tiles(level, tile_size)

    def _iter_tiles(self, level: int, tile_size: int) -> Iterator[np.ndarray]:
        height, width, _ = self.level_shape(level)
        nxt = self._allocate(level + 1) if level < self._subresolutions else None
        filled = 0

        for y in range(0, height, tile_size):
            band = self._read_band(level, y, min(tile_size, height - y))

            if nxt is not None:
                small = downsample_plane(band, self.photometric, 0.5)
                nxt[filled : filled + small.shape[0]] = small
                filled += small.shape[0]

            for x in range(0, width, tile_size):
                yield band[:, x : x + tile_size]

    def _read_band(self, level: int, y: int, band_rows: int) -> np.ndarray:
        if level == 0:
            return self._read_band_from_slide(y, band_rows)
        return np.asarray(self._levels[level][y : y + band_rows])

    def _read_band_from_slide(self, y: int, band_rows: int) -> np.ndarray:
        """Read one full-width band, in column chunks to cap the RGBA read buffer."""
        _, width, _ = self._shape0
        # read_region takes its location in the level-0 frame but its size in the
        # target level's, so the origin needs scaling back up by the downsample.
        downsample = self._slide.level_downsamples[self._level]
        band = np.empty((band_rows, width, 3), np.uint8)

        for x in range(0, width, self._read_chunk_cols):
            chunk_cols = min(self._read_chunk_cols, width - x)
            region = self._slide.read_region(
                (int(x * downsample), int(y * downsample)),
                self._level,
                (chunk_cols, band_rows),
            )
            band[:, x : x + chunk_cols] = np.asarray(region.convert("RGB"))
            region.close()

        return band

    def _allocate(self, level: int) -> np.memmap:
        if self._scratch is None:
            self._scratch = tempfile.TemporaryDirectory(prefix="spatialrefinery-pyramid-")
        shape = self.level_shape(level)
        memmap = np.memmap(Path(self._scratch.name) / f"level{level}.raw", dtype=np.uint8, mode="w+", shape=shape)
        self._levels[level] = memmap  # registered up-front so it survives a partial walk
        logger.debug("Staging pyramid level %d (shape: %s) on disk", level, shape)
        return memmap

    def close(self) -> None:
        """Drop the staged memmaps, delete the scratch directory, close the slide."""
        if self._closed:  # openslide raises rather than ignoring a second close
            return
        self._closed = True
        self._levels.clear()
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None
        self._slide.close()


def write_ome_tiff(
    path: str | Path,
    image: np.ndarray | PyramidSource,
    metadata: dict[str, Any],
    photometric: Photometric,
    *,
    subresolutions: int = DEFAULT_SUBRESOLUTIONS,
    tile_size: int = DEFAULT_TILE_SIZE,
    compression: str = "zlib",
    compression_level: int = 8,
    maxworkers: int = 8,
    bigtiff: bool = True,
) -> Path:
    """Write a pyramidal OME-TIFF. `path` may omit the `.ome.tif` suffix.

    Parameters
    ----------
    path : str | Path
        Output path. `.ome.tif` is appended if not already present.
    image : np.ndarray | PyramidSource
        Level-0 image data (YXC for `"rgb"`, YX or CYX for `"minisblack"`), or
        a :class:`PyramidSource` streaming a plane too large to materialise.
    metadata : dict
        OME metadata; `PhysicalSizeX`/`PhysicalSizeY` (default 1.0) set the
        base resolution that every pyramid sub-level is derived from.
    photometric : {"rgb", "minisblack"}
        Photometric interpretation passed to `tifffile`.
    subresolutions : int, optional
        Number of pyramid levels below level 0. Default 4.
    """
    import tifffile as tf

    fn = _ome_tiff_path(path)
    px_size_x = metadata.get("PhysicalSizeX", 1.0)
    px_size_y = metadata.get("PhysicalSizeY", 1.0)
    source = image if isinstance(image, PyramidSource) else ArrayPyramidSource(image, photometric)

    options: dict[str, Any] = {
        "photometric": photometric,
        "tile": (tile_size, tile_size),
        "maxworkers": maxworkers,
        "compression": compression,
        "compressionargs": {"level": compression_level},
        "resolutionunit": "CENTIMETER",
    }

    def write_level(tif: tf.TiffWriter, level: int, **kwargs: Any) -> None:
        shape = source.level_shape(level)
        logger.info("Writing pyramid level %d (shape: %s)", level, shape)
        data = source.level_data(level, tile_size)
        # An iterator carries no shape of its own, so tifffile needs it spelled
        # out; passing it alongside an ndarray would be rejected as redundant.
        if not isinstance(data, np.ndarray):
            kwargs |= {"shape": shape, "dtype": source.dtype}
        tif.write(data, **kwargs, **options)

    with tf.TiffWriter(str(fn), bigtiff=bigtiff) as tif:
        write_level(
            tif,
            0,
            subifds=subresolutions,
            resolution=(1e4 / px_size_x, 1e4 / px_size_y),
            metadata=metadata,
        )
        # Tracks how much the pixel size has grown relative to level 0: each
        # level halves image dimensions, so pixel size doubles. This is the
        # formula the svs/ndpi converters used; the old czi converter instead
        # tracked `scale /= 2` and computed `1e4 * 2**(i+1) / px`, which is
        # inverted -- every CZI sub-level claimed *finer* spacing than level 0.
        pixel_size_multiplier = 1.0
        for i in range(subresolutions):
            pixel_size_multiplier *= 2
            write_level(
                tif,
                i + 1,
                subfiletype=1,
                resolution=(
                    1e4 / (px_size_x * pixel_size_multiplier),
                    1e4 / (px_size_y * pixel_size_multiplier),
                ),
            )

    return fn


# --------------------------------------------------------------------- #
# Base converter
# --------------------------------------------------------------------- #


class BaseConverter(ABC):
    """Common surface for every converter: what it handles, and how to run it."""

    name: ClassVar[str]
    input_suffixes: ClassVar[tuple[str, ...]] = ()
    output_suffix: ClassVar[str] = ""

    @classmethod
    def supports(cls, path: str | Path) -> bool:
        """Return whether this converter can handle `path`, by suffix."""
        return Path(path).suffix.lower() in cls.input_suffixes

    @abstractmethod
    def convert(self, source: str | Path, output_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
        """Convert `source`, writing into `output_dir`. Returns every file/store produced."""


# --------------------------------------------------------------------- #
# Image branch: vendor image -> pyramidal OME-TIFF
# --------------------------------------------------------------------- #


@dataclass(slots=True)
class ImagePlane:
    """One writable 2D(+C) plane pulled out of a source image."""

    data: np.ndarray | PyramidSource
    photometric: Photometric
    metadata: dict[str, Any] = field(default_factory=dict)
    name: str | None = None  # scene/series name; None if the source has a single plane

    def close(self) -> None:
        """Release a streaming `data` source; a no-op for a plain array."""
        if isinstance(self.data, PyramidSource):
            self.data.close()


class ImageConverter(BaseConverter):
    """Template: read planes from a vendor image, write each as a pyramidal OME-TIFF."""

    output_suffix = ".ome.tif"

    def __init__(self, *, subresolutions: int = DEFAULT_SUBRESOLUTIONS, tile_size: int = DEFAULT_TILE_SIZE) -> None:
        self.subresolutions = subresolutions
        self.tile_size = tile_size

    @abstractmethod
    def read_planes(self, source: Path):
        """Yield one `ImagePlane` per scene/series found in `source`."""

    def output_basename(self, source: Path, plane: ImagePlane, n_planes: int) -> str:
        """`<stem>` for a single-plane source, `<stem>_<sanitised plane name>` otherwise."""
        if n_planes <= 1 or not plane.name:
            return source.stem

        from spatialrefinery.core.utils import transform_name

        return f"{source.stem}_{transform_name(plane.name)}"

    def convert(self, source: str | Path, output_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
        """Read every plane from `source` and write each as a pyramidal OME-TIFF."""
        source = Path(source)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        planes = list(self.read_planes(source))
        n_planes = len(planes)
        outputs: list[Path] = []

        try:
            for plane in planes:
                out_path = output_dir / self.output_basename(source, plane, n_planes)
                fn = _ome_tiff_path(out_path)
                if fn.exists() and not overwrite:
                    logger.info("Skipping existing output: %s", fn)
                    outputs.append(fn)
                    continue

                written = write_ome_tiff(
                    out_path,
                    plane.data,
                    plane.metadata,
                    plane.photometric,
                    subresolutions=self.subresolutions,
                    tile_size=self.tile_size,
                )
                outputs.append(written)
        finally:
            # A streamed plane keeps a slide handle and scratch memmaps open until
            # the write has drained it, so the close cannot live in `read_planes`.
            for plane in planes:
                plane.close()

        return outputs


@register_converter(overwrite=True)
class OpenSlideImageConverter(ImageConverter):
    """Reads whole-slide images via `openslide` (SVS, NDPI, and other Aperio/Hamamatsu formats)."""

    name = "openslide"
    input_suffixes = (".svs", ".ndpi", ".tif", ".tiff", ".mrxs", ".scn", ".bif", ".vms", ".svslide")

    def __init__(self, *, level: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.level = level

    def read_planes(self, source: Path):
        """Yield the single RGB plane at `self.level` from an openslide-readable WSI.

        The plane is yielded as a streaming :class:`OpenSlideTiledSource` rather
        than an array: a level-0 whole slide is routinely tens of GiB once read,
        and the caller closes the source when the write has drained it.
        """
        import openslide

        tiled = OpenSlideTiledSource(source, level=self.level, subresolutions=self.subresolutions)
        properties = tiled.properties
        mpp_x = float(properties.get(openslide.PROPERTY_NAME_MPP_X, 1.0))
        mpp_y = float(properties.get(openslide.PROPERTY_NAME_MPP_Y, 1.0))

        metadata = {
            "PhysicalSizeX": mpp_x,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": mpp_y,
            "PhysicalSizeYUnit": "µm",
            "Channel": {"Name": ["Red", "Green", "Blue"]},
        }
        yield ImagePlane(data=tiled, photometric="rgb", metadata=metadata, name=None)


@register_converter(overwrite=True)
class BioioImageConverter(ImageConverter):
    """Reads Carl Zeiss CZI (and other Bio-Formats-backed) images via `bioio`.

    Requires the optional `spatialrefinery[czi]` extra (`bioio`, `bioio-czi`);
    the import is function-local so registering this class never requires the
    extra to be installed.
    """

    name = "bioio"
    input_suffixes = (".czi",)

    def read_planes(self, source: Path):
        """Yield one plane per CZI scene; photometric interpretation inferred per-scene."""
        from bioio import BioImage

        from spatialrefinery.core.utils import transform_name

        img = BioImage(source)
        for scene_idx, scene_name in enumerate(img.scenes):
            img.set_scene(scene_name)

            data = np.squeeze(img.data)
            pixel_sizes = img.physical_pixel_sizes
            channel_names = img.channel_names

            is_rgb = (len(channel_names) == 3 and data.dtype == np.uint8) or (
                data.ndim >= 3 and data.shape[-1] == 3 and data.dtype == np.uint8
            )
            if is_rgb:
                photometric: Photometric = "rgb"
                if data.ndim == 3 and data.shape[0] == 3:
                    data = np.moveaxis(data, 0, -1)  # CYX -> YXC
            else:
                photometric = "minisblack"

            metadata: dict[str, Any] = {
                "PhysicalSizeX": pixel_sizes.X or 1.0,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": pixel_sizes.Y or 1.0,
                "PhysicalSizeYUnit": "µm",
            }
            if channel_names:
                metadata["Channel"] = {"Name": channel_names}

            name = transform_name(scene_name) if scene_name else f"scene{scene_idx:03d}"
            yield ImagePlane(data=data, photometric=photometric, metadata=metadata, name=name)


def convert_to_ometiff(
    source: str | Path,
    output_dir: str | Path,
    *,
    subresolutions: int = DEFAULT_SUBRESOLUTIONS,
    tile_size: int = DEFAULT_TILE_SIZE,
    overwrite: bool = False,
    converter: type[ImageConverter] | None = None,
) -> list[Path]:
    """Convert one image file to pyramidal OME-TIFF, dispatching on suffix.

    Parameters
    ----------
    source : str | Path
        Path to the source image (e.g. `.svs`, `.ndpi`, `.czi`).
    output_dir : str | Path
        Directory to write the `.ome.tif` file(s) into.
    subresolutions : int, optional
        Number of pyramid sub-levels. Default 4.
    tile_size : int, optional
        TIFF tile edge length in pixels. Default 1024.
    overwrite : bool, optional
        Whether to regenerate an output that already exists. Default False.
    converter : type[ImageConverter], optional
        Force a specific converter class instead of registry dispatch.

    Returns
    -------
    list[Path]
        Every `.ome.tif` file written (or found, if `overwrite=False`).

    Raises
    ------
    spatialrefinery.core.registry.RegistryError
        If `source`'s suffix has no registered converter, or `source` is
        itself an `.ome.tif`/`.ome.tiff` file.

    Examples
    --------
    >>> from spatialrefinery import convert_to_ometiff
    >>> ometiff_paths = convert_to_ometiff(
    ...     source="/path/to/slide.svs",
    ...     output_dir="/path/to/output",
    ...     subresolutions=4,
    ... )
    """
    from spatialrefinery.core.registry import get_converter_for

    source = Path(source)
    converter_cls = converter if converter is not None else get_converter_for(source)
    instance = converter_cls(subresolutions=subresolutions, tile_size=tile_size)
    return instance.convert(source, output_dir, overwrite=overwrite)


# --------------------------------------------------------------------- #
# SpatialData branch: raw vendor bundle directory -> .zarr
# --------------------------------------------------------------------- #


class SpatialDataConverter(BaseConverter):
    """Converts a raw vendor bundle directory into a SpatialData `.zarr` store.

    Subclassed per technology (see `spatialrefinery.io.xenium.XeniumConverter`).
    """

    technology: ClassVar[str]
    output_suffix = ".zarr"

    @classmethod
    def supports(cls, path: str | Path) -> bool:
        """Default: a vendor bundle is a directory (subclasses may narrow this)."""
        return Path(path).is_dir()

    @abstractmethod
    def convert(
        self,
        source: str | Path,
        output_dir: str | Path,
        *,
        output_name: str | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> list[Path]:
        """Convert the bundle at `source` into one or more `.zarr` stores under `output_dir`."""
