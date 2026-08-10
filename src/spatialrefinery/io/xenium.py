"""
Module for converting 10x Xenium raw data to SpatialData zarr format.

This module provides functions to:
- Read 10x Xenium bundled outputs
- Create pseudo-spots mimicking Visium-like data
- Align H&E images with the spatial data
- Export to SpatialData zarr format
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import spatialdata as sd
from PIL import Image
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity, Scale, get_transformation, set_transformation
from spatialdata.transformations.transformations import Affine
from spatialdata_io import xenium, xenium_aligned_image

from spatialrefinery.core.converter import SpatialDataConverter
from spatialrefinery.core.downloader import BaseDownloader, DownloadResult, RemoteAsset
from spatialrefinery.core.registry import TechnologySpec, register_technology
from spatialrefinery.core.utils import (
    create_circular_spots,
    create_hexagonal_spots,
    fix_table_validation_errors,
    parse_curl_manifest,
    segment_tissue,
    slide_to_numpy,
    transform_name,
)

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

# create_circular_spots, create_hexagonal_spots, fix_table_validation_errors,
# segment_tissue, slide_to_numpy, and transform_name now live in
# spatialrefinery.core.utils (technology-agnostic geometry/validation
# helpers); they are imported and re-exported above so existing callers of
# `spatialrefinery.io.xenium.transform_name` etc. keep working unchanged.


def read_xenium_alignment(alignment_file_path: str):
    """
    Read a Xenium alignment file and convert it to a 3x3 affine matrix.

    Parameters
    ----------
    alignment_file_path : str
        Path to the Xenium alignment CSV file.

    Returns
    -------
    np.ndarray
        A 3x3 affine transformation matrix.
    """
    import numpy as np

    alignment_file = pd.read_csv(alignment_file_path, header=None)
    alignment_matrix = alignment_file.values

    # Xenium explorer >= v2.0
    if isinstance(alignment_matrix[0][0], str) and "fixedX" in alignment_matrix[0]:
        from spatialrefinery.core.utils import affine_from_point_pairs

        points = pd.read_csv(alignment_file_path)
        points = points.iloc[:3]
        dst_pts = points[["fixedX", "fixedY"]].values
        src_pts = points[["alignmentX", "alignmentY"]].values
        alignment_matrix = affine_from_point_pairs(src_pts, dst_pts)

    return np.asarray(alignment_matrix)


def create_pseudo_spots(sdata, spot_size_um: int = 55, overlap: float | None = 0.06, values: str = "transcripts"):
    """
    Create pseudo-spots from transcripts and add them to the SpatialData object.

    This function creates pseudo-spots by binning transcripts, determines which
    spots intersect with cell boundaries (are "in tissue"), and adds the results
    to the SpatialData object.

    Parameters
    ----------
    sdata : spatialdata.SpatialData
        The SpatialData object containing transcripts and cell boundaries.
    spot_size_um : int, optional
        The size of pseudo-spots in micrometers. Default is 55.
    overlap : float, optional
        Fractional overlap between adjacent hexagonal spots. Default is 0.06.
    values : str, optional
        What to aggregate into each pseudo-spot: "transcripts" (default) or
        "cell_boundaries".

    Returns
    -------
    spatialdata.SpatialData
        The updated SpatialData object with added pseudo-spots.
    """
    import geopandas as gpd

    if values not in ("transcripts", "cell_boundaries"):
        raise ValueError(f"values can only be one of ['transcripts', 'cell_boundaries'], found {values!r}")

    # Create the pseudo-spots
    hexagonal_spots_gdf = create_hexagonal_spots(
        sdata["transcripts"], key_x="x", key_y="y", spot_size_um=spot_size_um, overlap=overlap
    )

    # Create transformation for the spot polygons
    transform = get_transformation(sdata["cell_boundaries"], to_coordinate_system="global")
    hexagonal_polygons = ShapesModel.parse(hexagonal_spots_gdf, transformations={"global": transform})

    # Add to SpatialData object
    sdata[f"spots_{spot_size_um!s}um"] = hexagonal_polygons
    logger.info("Updating sdata with shape element for spots of size %sµm...", spot_size_um)
    sdata.write_element(f"spots_{spot_size_um!s}um")

    # Clear temporary object to free memory
    del hexagonal_spots_gdf, hexagonal_polygons

    # Adjust transcripts transforms to 2D
    transcripts_transform = get_transformation(sdata["transcripts"], to_coordinate_system="global")
    if isinstance(transcripts_transform, Scale) and len(transcripts_transform.scale) == 3:
        # Create a 2D Scale transformation using only x and y components
        transcripts_2d_transform = Scale(
            scale=transcripts_transform.scale[:2],  # Take only x, y scale values
            axes=("x", "y"),  # Set to 2D axes
        )
        set_transformation(sdata["transcripts"], transcripts_2d_transform, to_coordinate_system="global")

    # Pool transcripts (or cell_boundaries) to pseudo-spots using the hexagonal polygons
    if values == "transcripts":
        aggr_sdata = sdata.aggregate(
            values="transcripts",
            by=f"spots_{spot_size_um!s}um",
            value_key="feature_name",
            agg_func="count",
            table_name="table",
            deepcopy=False,
        )
    else:  # values == "cell_boundaries"
        aggr_sdata = sdata.aggregate(
            values="cell_boundaries",
            by=f"spots_{spot_size_um!s}um",
            value_key=sdata["table"].var.index.tolist(),
            agg_func="sum",
            fractions=True,
            table_name="table",
            deepcopy=False,
        )

    # Ensure only common genes are included
    common_genes = aggr_sdata["table"].var.index.intersection(sdata["table"].var.index)

    # Filter to common genes in-place (creates a view, not a full copy)
    aggr_sdata["table"] = aggr_sdata["table"][:, common_genes]

    # Add spatial coordinates directly to the table
    aggr_sdata["table"].obsm["spatial"] = aggr_sdata[f"spots_{spot_size_um!s}um"][["x_um", "y_um"]].values

    # Determine which spots are "in tissue" by checking intersection with cell boundaries.
    # `tissue_contours` is only written when an aligned H&E image was found and tissue
    # segmentation succeeded (see xenium_to_spatialdata); without it, every spot's
    # in-tissue status is unknown, so default to 1 rather than raising.
    if "tissue_contours" in sdata:
        spot_gdf = sd.transform(sdata[f"spots_{spot_size_um!s}um"], to_coordinate_system="global").copy()
        contour_gdf = sd.transform(sdata["tissue_contours"], to_coordinate_system="global").copy()

        # Perform spatial join to find spots that intersect with tissue contours
        intersecting_polygons = gpd.sjoin(spot_gdf, contour_gdf, how="inner", predicate="intersects")
        # Intersect with obs.index first: `.loc` setitem with unknown labels would
        # silently *enlarge* obs with spurious rows instead of raising.
        in_tissue_indices = intersecting_polygons.index.unique().astype(str).intersection(aggr_sdata["table"].obs.index)
        aggr_sdata["table"].obs["in_tissue"] = aggr_sdata["table"].obs.index.isin(in_tissue_indices).astype(int)

        del spot_gdf, contour_gdf, intersecting_polygons
    else:
        logger.warning(
            "No 'tissue_contours' found in sdata; marking all spots as in_tissue=1 "
            "(no H&E image was aligned, or tissue segmentation failed)."
        )
        aggr_sdata["table"].obs["in_tissue"] = 1

    # Add to SpatialData object
    sdata[f"spots_{spot_size_um!s}um_table"] = aggr_sdata["table"]

    logger.info("Updating sdata with table element for spots of size %sµm...", spot_size_um)
    sdata.write_element(f"spots_{spot_size_um!s}um_table")

    # Clean up aggregated sdata
    del aggr_sdata

    return sdata


def find_xenium_files(dataset_path: Path) -> dict:
    """
    Find all relevant Xenium files in the dataset directory.

    Parameters
    ----------
    dataset_path : Path
        Path to the Xenium dataset directory.

    Returns
    -------
    dict
        Dictionary containing paths to found files (None if not found).
    """
    file_patterns = {
        "img_path": ["*_he_*.tiff", "*_he_*.tif", "*_he_*.svs", "*_he_*.ndpi"],
        "alignment_file_path": ["*he_imagealignment.csv"],
        "experiment_path": ["experiment.xenium"],
    }

    paths = {}
    for file_key, patterns in file_patterns.items():
        found_file = None
        for pattern in patterns:
            matches = list(dataset_path.glob(pattern))
            if matches:
                found_file = matches[0]  # Take the first match
                break
        paths[file_key] = found_file

    return paths


def xenium_to_spatialdata(
    dataset_path: str | Path,
    output_path: str | Path,
    output_name: str | None = None,
    include_aligned_image: bool = True,
    create_spots: bool = True,
    spot_sizes: list | None = None,
    overlap: float | None = 0.0,
    values: str = "transcripts",
    n_jobs: int = 1,
    overwrite: bool = False,
) -> Path:
    """
    Convert 10x Xenium raw data to SpatialData zarr format.

    This is the main function for converting Xenium bundled outputs to a
    SpatialData object and saving it as a zarr file. It can optionally include
    aligned H&E images and create pseudo-spots at specified sizes.

    Parameters
    ----------
    dataset_path : Union[str, Path]
        Path to the Xenium dataset directory containing raw files.
    output_path : Union[str, Path]
        Path to the directory where the zarr file will be saved.
    output_name : Optional[str], optional
        Name for the output zarr file (without .zarr extension).
        If None, uses the dataset directory name. Default is None.
    include_aligned_image : bool, optional
        Whether to include aligned H&E image if available. Default is True.
    create_spots : bool, optional
        Whether to create pseudo-spots. Default is True.
    spot_sizes : list, optional
        List of spot sizes in micrometers to create. Default is None.
        Spots are only created when both `create_spots` is True *and*
        `spot_sizes` is a non-empty list -- the default combination
        (`create_spots=True`, `spot_sizes=None`) creates no spots.
    overlap : float, optional
        Fractional overlap between adjacent hexagonal spots, forwarded to
        `create_pseudo_spots`. Default is 0.0 (no overlap).
    values : str, optional
        Which SpatialData element to aggregate into spots: `"transcripts"`
        or `"cell_boundaries"`. Default is `"transcripts"`.
    n_jobs : int, optional
        Number of workers for parallel processing. Default is 1.
    overwrite : bool, optional
        Whether to overwrite existing zarr file. Default is False.

    Returns
    -------
    Path
        Path to the created zarr file.

    Raises
    ------
    FileNotFoundError
        If `dataset_path` does not contain an `experiment.xenium` file.

    Examples
    --------
    >>> from spatialrefinery import xenium_to_spatialdata
    >>> zarr_path = xenium_to_spatialdata(
    ...     dataset_path="/path/to/xenium/data",
    ...     output_path="/path/to/output",
    ...     output_name="my_sample",
    ...     create_spots=True,
    ...     spot_sizes=[55, 100],
    ... )
    """
    # Convert to Path objects
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    # Determine output name
    if output_name is None:
        output_name = dataset_path.name

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    zarr_path = output_path / f"{output_name}.zarr"

    # Find relevant files
    file_paths = find_xenium_files(Path(dataset_path))

    # Load experiment specifications
    if file_paths["experiment_path"] is None:
        raise FileNotFoundError(f"experiment.xenium file not found in {dataset_path}")

    with open(file_paths["experiment_path"]) as f:
        specs = json.load(f)

    # Check if we should use existing data or create new
    if zarr_path.exists() and not overwrite:
        logger.info("Loading existing zarr file: %s", zarr_path)
        # Skip to spot creation with existing data
        # No need to load the full object here, we'll do it in the spot creation loop
    else:
        # Create new data
        logger.info("Processing Xenium dataset: %s", dataset_path.name)
        logger.info("Loading Xenium data...")

        # Load the base Xenium data
        sdata = xenium(
            str(dataset_path), aligned_images=False, morphology_focus=False, n_jobs=n_jobs, cells_as_circles=False
        )

        # Fix any validation errors in the table
        sdata["table"] = fix_table_validation_errors(sdata["table"])

        sdata["table"].obs["region"] = "cell_boundaries"
        sdata["table"].obs["instance_id"] = sdata["table"].obs["cell_id"]
        sdata.set_table_annotates_spatialelement(
            "table", region="cell_boundaries", instance_key="instance_id", region_key="region"
        )

        # Write the base SpatialData object to disk before spot creation to free up memory
        logger.info("Writing base data to %s...", zarr_path)
        sdata.write(zarr_path)
        del sdata

        # Add aligned H&E image if requested and available
        if include_aligned_image and file_paths["img_path"] is not None:
            logger.info("Adding aligned H&E image...")
            try:
                sdata = sd.read_zarr(zarr_path)
                aligned_he_image = xenium_aligned_image(
                    image_path=str(file_paths["img_path"]),
                    alignment_file=str(file_paths["alignment_file_path"])
                    if file_paths["alignment_file_path"]
                    else None,
                    image_models_kwargs={"scale_factors": [2, 2], "chunks": {"x": 512, "y": 512}},
                )
                sdata["he_image"] = aligned_he_image
                sdata.write_element("he_image")

                sdata.attrs["source_mpp"] = specs["pixel_size"]

                logger.info("H&E image added successfully")

                # Segment tissue in the H&E image's own raw pixel grid (pixel_size=1.0):
                # the alignment matrix below maps that grid straight into the shared
                # global (micron) space, and already carries the H&E image's own
                # pixel size plus rotation -- which is not `specs["pixel_size"]`
                # (that's the *Xenium morphology* image's resolution, a separate scan
                # at a different pixel size). Pre-scaling by it here would double-apply
                # a scale factor and place `tissue_contours` far from the true tissue.
                tissue_contours = segment_tissue(wsi_path=str(file_paths["img_path"]), pixel_size=1.0, method="otsu")
                if file_paths["alignment_file_path"] is not None:
                    # `read_xenium_alignment` (not a raw `pd.read_csv`) so that Xenium
                    # Explorer >= v2.0 keypoint-format alignment files (columns
                    # `fixedX`/`fixedY`/`alignmentX`/`alignmentY`) are converted to an
                    # affine correctly instead of being fed to `Affine` verbatim.
                    alignment = read_xenium_alignment(str(file_paths["alignment_file_path"]))
                    transform = Affine(alignment, input_axes=("x", "y"), output_axes=("x", "y"))
                else:
                    logger.warning("No alignment file found; tissue contours will use an identity transform.")
                    transform = Identity()

                sdata["tissue_contours"] = ShapesModel.parse(tissue_contours, transformations={"global": transform})
                sdata.write_element("tissue_contours")
                sdata.write_metadata()

                logger.info("H&E Tissue Contours added successfully")

            except Exception:  # best-effort: a failure here must not abort the base conversion
                logger.exception("Failed to add H&E image")

    # Create pseudo-spots if requested
    if create_spots and spot_sizes:
        logger.info("Creating pseudo-spots iteratively...")
        # Load the current state from the Zarr store. This is memory-efficient as it supports lazy loading.
        sdata_iter = sd.read_zarr(zarr_path)

        for spot_size in spot_sizes:
            logger.info("Processing spot size: %sµm", spot_size)
            sdata_iter = create_pseudo_spots(sdata_iter, spot_size, overlap, values)

    logger.info("Successfully created %s", zarr_path)

    return zarr_path


def xenium_to_spatialdata_zip(
    dataset_path: str | Path,
    output_path: str | Path,
    output_name: str | None = None,
    include_aligned_image: bool = True,
    create_spots: bool = True,
    spot_sizes: list | None = None,
    overlap: float | None = 0.0,
    values: str = "transcripts",
    n_jobs: int = 1,
    overwrite: bool = False,
    keep_zarr: bool = True,
) -> Path:
    """
    Convert Xenium data to SpatialData zarr and create a zip archive.

    This function is a wrapper around `xenium_to_spatialdata` that additionally
    creates a zip archive of the zarr directory for easier transfer.

    Parameters
    ----------
    dataset_path : Union[str, Path]
        Path to the Xenium dataset directory containing raw files.
    output_path : Union[str, Path]
        Path to the directory where files will be saved.
    output_name : Optional[str], optional
        Name for the output files. If None, uses the dataset directory name.
    include_aligned_image : bool, optional
        Whether to include aligned H&E image if available. Default is True.
    create_spots : bool, optional
        Whether to create pseudo-spots. Default is True.
    spot_sizes : list, optional
        List of spot sizes in micrometers. Default is None.
        Spots are only created when both `create_spots` is True *and*
        `spot_sizes` is a non-empty list -- the default combination
        (`create_spots=True`, `spot_sizes=None`) creates no spots.
    overlap : float, optional
        Fractional overlap between adjacent hexagonal spots, forwarded to
        `create_pseudo_spots`. Default is 0.0 (no overlap).
    values : str, optional
        Which SpatialData element to aggregate into spots: `"transcripts"`
        or `"cell_boundaries"`. Default is `"transcripts"`.
    n_jobs : int, optional
        Number of workers for parallel processing. Default is 1.
    overwrite : bool, optional
        Whether to overwrite existing files. Default is False.
    keep_zarr : bool, optional
        Whether to keep the unzipped zarr directory after creating zip. Default is True.

    Returns
    -------
    Path
        Path to the created zip file.

    Raises
    ------
    FileNotFoundError
        If `dataset_path` does not contain an `experiment.xenium` file
        (raised by the underlying `xenium_to_spatialdata` call).

    Examples
    --------
    >>> from spatialrefinery import xenium_to_spatialdata_zip
    >>> zip_path = xenium_to_spatialdata_zip(
    ...     dataset_path="/path/to/xenium/data",
    ...     output_path="/path/to/output",
    ...     output_name="my_sample",
    ...     keep_zarr=False,  # Remove zarr directory after zipping
    ... )
    """
    # First create the zarr file
    zarr_path = xenium_to_spatialdata(
        dataset_path=dataset_path,
        output_path=output_path,
        output_name=output_name,
        include_aligned_image=include_aligned_image,
        create_spots=create_spots,
        spot_sizes=spot_sizes,
        overlap=overlap,
        values=values,
        n_jobs=n_jobs,
        overwrite=overwrite,
    )

    # Create zip archive
    zip_path = Path(f"{zarr_path}.zip")

    if zip_path.exists() and not overwrite:
        logger.warning("Zip file already exists: %s", zip_path)
        return zip_path

    logger.info("Creating zip archive...")
    shutil.make_archive(
        base_name=str(zarr_path),  # Output path without .zip extension
        format="zip",
        root_dir=zarr_path.parent,
        base_dir=zarr_path.name,
    )

    logger.info("Successfully created %s", zip_path)

    # Optionally remove the zarr directory
    if not keep_zarr:
        logger.info("Removing zarr directory...")
        shutil.rmtree(zarr_path)

    return zip_path


# --------------------------------------------------------------------- #
# Download: fetch a Xenium study's raw asset bundle
# --------------------------------------------------------------------- #

#: Maps a filename suffix (as used in 10x's manifests) to a semantic asset
#: kind. Longest suffix wins -- e.g. "..._xe_outs.zip" must not classify as
#: "outs" just because it also ends in "_outs.zip".
_XENIUM_KINDS: dict[str, str] = {
    "_outs.zip": "outs",
    "_xe_outs.zip": "xe_outs",
    "_he_image.ome.tif": "he_image",
    "_he_unaligned_image.ome.tif": "he_unaligned_image",
    "_he_annotated_image.ome.tif": "he_annotated_image",
    "_if_image.ome.tif": "if_image",
    "_if_imagealignment.csv": "if_alignment",
    "_he_imagealignment.csv": "he_alignment",
    "_gene_groups.csv": "gene_groups",
    "_cell_groups.csv": "cell_groups",
    "_gene_list.csv": "gene_list",
    "_cell_types.csv": "cell_types",
}
_XENIUM_KINDS_BY_LENGTH = sorted(_XENIUM_KINDS, key=len, reverse=True)


class XeniumDownloader(BaseDownloader):
    """Fetch a Xenium study's raw asset bundle from a `curl -O <url>` manifest.

    `source` passed to `iter_assets`/`plan`/`run` may be a manifest file
    path (see `example_data/10x_xenium_human.txt`) or any iterable of URLs.
    """

    technology = "xenium"
    # `*_xe_outs.zip` (10x Xenium Explorer bundles) are large auxiliary
    # archives not needed for SpatialData conversion; never auto-extract them.
    never_extract = frozenset({"xe_outs"})

    @staticmethod
    def classify(filename: str) -> str:
        """Return the asset kind for `filename`, or `"unknown"` if unrecognised."""
        for suffix in _XENIUM_KINDS_BY_LENGTH:
            if filename.endswith(suffix):
                return _XENIUM_KINDS[suffix]
        return "unknown"

    def iter_assets(self, source: str | Path | Iterable[str]) -> Iterator[RemoteAsset]:
        """Yield one `RemoteAsset` per URL in `source` (a manifest path or an iterable of URLs)."""
        if isinstance(source, str | Path) and Path(source).is_file():
            urls: Iterable[str] = parse_curl_manifest(source)
        elif isinstance(source, Path):
            raise FileNotFoundError(f"Manifest file not found: {source}")
        else:
            urls = source

        for url in urls:
            asset = RemoteAsset.from_url(url)
            yield RemoteAsset(
                url=asset.url,
                study=asset.study,
                filename=asset.filename,
                kind=self.classify(asset.filename),
            )


def download_xenium_study(
    source: str | Path | Iterable[str],
    outdir: str | Path,
    *,
    studies: list[str] | None = None,
    kinds: list[str] | None = None,
    max_workers: int = 8,
    retries: int = 3,
    timeout: int = 180,
    extract: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[DownloadResult]:
    """Download a Xenium study's raw asset bundle.

    Parameters
    ----------
    source : str | Path | Iterable[str]
        Path to a manifest file of `curl -O <url>` lines (one per asset,
        as downloaded from 10x's dataset pages), or an iterable of URLs.
    outdir : str | Path
        Base directory; assets land under `outdir/<study>/<filename>`.
    studies : list[str], optional
        Restrict to these study names. Default: all studies in `source`.
    kinds : list[str], optional
        Restrict to these asset kinds (e.g. `["outs"]`). Default: all kinds.
    max_workers : int, optional
        Parallel download workers. Default 8.
    retries : int, optional
        Number of retry attempts per asset on transient failure. Default 3.
    timeout : int, optional
        Per-request timeout in seconds. Default 180.
    extract : bool, optional
        Whether to unzip `.zip` assets in place after downloading
        (`*_xe_outs.zip` is always skipped). Default True.
    overwrite : bool, optional
        Re-download assets that already exist on disk. Default False.
    dry_run : bool, optional
        Resolve and report what would be downloaded without any network
        activity. Default False.

    Returns
    -------
    list[DownloadResult]
        One result per selected asset. Each result's `.status` is one of
        `"downloaded"`, `"cached"` (already present, `overwrite=False`),
        `"skipped"` (excluded by `dry_run`), or `"failed"`; `.ok` is a bool
        convenience property, and `.error` holds the exception message
        for failed downloads.
    """
    downloader = XeniumDownloader(
        outdir,
        max_workers=max_workers,
        retries=retries,
        timeout=timeout,
        extract=extract,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    return downloader.run(source, studies=studies, kinds=kinds)


# --------------------------------------------------------------------- #
# Convert: SpatialDataConverter adapter over xenium_to_spatialdata[_zip]
# --------------------------------------------------------------------- #


class XeniumConverter(SpatialDataConverter):
    """`SpatialDataConverter` adapter delegating to `xenium_to_spatialdata[_zip]`.

    The module-level functions remain the canonical, stable API (used
    directly by `examples/xenium_conversion_example.ipynb`); this class is
    the registry-facing surface for generic tooling that dispatches by
    technology rather than importing `spatialrefinery.io.xenium` directly.
    """

    name = "xenium"
    technology = "xenium"
    input_suffixes = ()  # a Xenium bundle is a directory; see supports()

    def __init__(
        self,
        *,
        create_spots: bool = True,
        spot_sizes: list | None = None,
        overlap: float | None = 0.0,
        values: str = "transcripts",
        include_aligned_image: bool = True,
        n_jobs: int = 1,
    ) -> None:
        self.create_spots = create_spots
        self.spot_sizes = spot_sizes
        self.overlap = overlap
        self.values = values
        self.include_aligned_image = include_aligned_image
        self.n_jobs = n_jobs

    @classmethod
    def supports(cls, path: str | Path) -> bool:
        """Whether `path` looks like a Xenium bundle directory."""
        path = Path(path)
        return path.is_dir() and find_xenium_files(path)["experiment_path"] is not None

    def convert(
        self,
        source: str | Path,
        output_dir: str | Path,
        *,
        output_name: str | None = None,
        overwrite: bool = False,
        zip_output: bool = False,
        **kwargs: Any,
    ) -> list[Path]:
        """Convert a Xenium bundle to a SpatialData zarr (optionally zipped)."""
        fn = xenium_to_spatialdata_zip if zip_output else xenium_to_spatialdata
        params = {
            "dataset_path": source,
            "output_path": output_dir,
            "output_name": output_name,
            "include_aligned_image": self.include_aligned_image,
            "create_spots": self.create_spots,
            "spot_sizes": self.spot_sizes,
            "overlap": self.overlap,
            "values": self.values,
            "n_jobs": self.n_jobs,
            "overwrite": overwrite,
            **kwargs,
        }
        return [fn(**params)]


__all__ = [
    "XeniumConverter",
    "XeniumDownloader",
    "create_circular_spots",
    "create_hexagonal_spots",
    "create_pseudo_spots",
    "download_xenium_study",
    "find_xenium_files",
    "fix_table_validation_errors",
    "read_xenium_alignment",
    "segment_tissue",
    "slide_to_numpy",
    "transform_name",
    "xenium_to_spatialdata",
    "xenium_to_spatialdata_zip",
]

# Register "xenium" as a known technology. overwrite=True because this
# module may be re-imported under pytest's --import-mode=importlib.
register_technology(
    TechnologySpec(
        name="xenium",
        downloader=XeniumDownloader,
        converter=XeniumConverter,
        file_finder=find_xenium_files,
        aliases=("10x_xenium", "xenium_v1", "xenium_prime"),
        description="10x Genomics Xenium in-situ platform",
    ),
    overwrite=True,
)
