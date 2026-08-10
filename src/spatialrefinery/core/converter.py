"""Converters: image-format -> pyramidal OME-TIFF, and vendor bundle -> SpatialData zarr.

These two conversion shapes are deliberately kept as separate template
hierarchies under one two-method :class:`BaseConverter` (``supports`` /
``convert``), rather than forced into a single ``read/transform/write``
pipeline: OME-TIFF conversion is image-in, image-out, synchronous, and
entirely in memory, while a vendor bundle -> zarr conversion (see
``spatialrefinery.io.xenium.XeniumConverter``) is a heterogeneous
directory read incrementally into an on-disk store. Sharing more than the
capability check between them would only produce a template with escape
hatches.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
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


def write_ome_tiff(
    path: str | Path,
    image: np.ndarray,
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
    image : np.ndarray
        Level-0 image data (YXC for `"rgb"`, YX or CYX for `"minisblack"`).
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

    options: dict[str, Any] = {
        "photometric": photometric,
        "tile": (tile_size, tile_size),
        "maxworkers": maxworkers,
        "compression": compression,
        "compressionargs": {"level": compression_level},
        "resolutionunit": "CENTIMETER",
    }

    with tf.TiffWriter(str(fn), bigtiff=bigtiff) as tif:
        logger.info("Writing pyramid level 0 (shape: %s)", image.shape)
        tif.write(
            image,
            subifds=subresolutions,
            resolution=(1e4 / px_size_x, 1e4 / px_size_y),
            metadata=metadata,
            **options,
        )

        current = image
        # Tracks how much the pixel size has grown relative to level 0: each
        # level halves image dimensions, so pixel size doubles. This is the
        # formula the svs/ndpi converters used; the old czi converter instead
        # tracked `scale /= 2` and computed `1e4 * 2**(i+1) / px`, which is
        # inverted -- every CZI sub-level claimed *finer* spacing than level 0.
        pixel_size_multiplier = 1.0
        for i in range(subresolutions):
            current = downsample_plane(current, photometric, 0.5)
            pixel_size_multiplier *= 2

            logger.info("Writing pyramid level %d (shape: %s)", i + 1, current.shape)
            tif.write(
                current,
                subfiletype=1,
                resolution=(
                    1e4 / (px_size_x * pixel_size_multiplier),
                    1e4 / (px_size_y * pixel_size_multiplier),
                ),
                **options,
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

    data: np.ndarray
    photometric: Photometric
    metadata: dict[str, Any] = field(default_factory=dict)
    name: str | None = None  # scene/series name; None if the source has a single plane


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
        """Yield the single RGB plane at `self.level` from an openslide-readable WSI."""
        import openslide

        from spatialrefinery.core.utils import slide_to_numpy

        slide = openslide.OpenSlide(str(source))
        try:
            properties = slide.properties
            mpp_x = float(properties.get(openslide.PROPERTY_NAME_MPP_X, 1.0))
            mpp_y = float(properties.get(openslide.PROPERTY_NAME_MPP_Y, 1.0))
            image_data = slide_to_numpy(slide, level=self.level)

            metadata = {
                "PhysicalSizeX": mpp_x,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": mpp_y,
                "PhysicalSizeYUnit": "µm",
                "Channel": {"Name": ["Red", "Green", "Blue"]},
            }
            yield ImagePlane(data=image_data, photometric="rgb", metadata=metadata, name=None)
        finally:
            slide.close()


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
