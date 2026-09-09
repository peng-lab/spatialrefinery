"""
I/O module for spatial transcriptomics data formats.

This module provides functions for reading and converting various
spatial transcriptomics data formats to SpatialData zarr format.
"""

from .visium import (
    VisiumDownloader,
    classify_visium_asset,
    download_visium_study,
    find_visium_files,
)
from .xenium import (
    XeniumConverter,
    XeniumDownloader,
    create_circular_spots,
    create_hexagonal_spots,
    create_pseudo_spots,
    download_xenium_study,
    find_xenium_files,
    fix_table_validation_errors,
    read_xenium_alignment,
    segment_tissue,
    slide_to_numpy,
    transform_name,
    xenium_to_spatialdata,
    xenium_to_spatialdata_zip,
)

__all__ = [
    "VisiumDownloader",
    "XeniumConverter",
    "XeniumDownloader",
    "classify_visium_asset",
    "create_circular_spots",
    "create_hexagonal_spots",
    "create_pseudo_spots",
    "download_visium_study",
    "download_xenium_study",
    "find_visium_files",
    "find_xenium_files",
    "fix_table_validation_errors",
    "read_xenium_alignment",
    "segment_tissue",
    "slide_to_numpy",
    "transform_name",
    "xenium_to_spatialdata",
    "xenium_to_spatialdata_zip",
]
