"""Technology-agnostic helper functions shared across ``spatialrefinery``.

Two groups of functions live here, split by dependency weight:

- **Pure helpers** (name sanitization, curl-manifest parsing, zip-slip-safe
  extraction, spot-grid geometry) use only the standard library, numpy, and
  pandas at module scope, so they are fast to import and easy to unit test
  offline.
- **Heavy helpers** (table validation, tissue segmentation, spot-shape
  construction) need spatialdata / geopandas / shapely / cv2 / openslide,
  which are imported *inside* each function so that importing this module
  never pulls in the full spatial-omics stack.
"""

from __future__ import annotations

import logging
import math
import re
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Names and manifests
# --------------------------------------------------------------------- #

#: Matches a `curl -O <url>` manifest line, tolerating extra flags
#: (e.g. `curl -L -O <url>`) between `curl` and `-O`.
CURL_RE = re.compile(r"^\s*curl\s+(?:-[A-Za-z]+\s+)*-O\s+(https?://\S+)\s*$", re.IGNORECASE)


def transform_name(old_name: str) -> str:
    """Replace characters invalid for SpatialData names with underscores.

    Parameters
    ----------
    old_name : str
        The original name string.

    Returns
    -------
    str
        The transformed name with only valid characters (alphanumeric, '.', '_', '-').
    """
    return re.sub(r"[^\w\._-]", "_", old_name)


def iter_curl_urls(lines: Iterable[str]) -> Iterator[str]:
    """Yield URLs from `curl -O <url>` lines, ignoring comments/blanks/other commands."""
    for line in lines:
        m = CURL_RE.match(line)
        if m:
            yield m.group(1)


def parse_curl_manifest(path: str | Path) -> list[str]:
    """Parse a text manifest of `curl -O <url>` lines into a list of URLs."""
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return list(iter_curl_urls(f))


def split_study_filename(url: str) -> tuple[str, str]:
    """Split a download URL into `(study, filename)` from its last two path segments."""
    parsed = urlparse(url)
    parts = [p for p in Path(parsed.path).parts if p != "/"]
    if len(parts) < 2:
        raise ValueError(f"Cannot determine study from URL: {url}")
    return parts[-2], parts[-1]


def ensure_dir(path: str | Path) -> Path:
    """Create `path` (and parents) if missing, and return it as a `Path`."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def human_bytes(n: int | float) -> str:
    """Render a byte count as a human-readable string, e.g. `1.5MB`."""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(value) < 1024.0:
            return f"{value:3.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}EB"


def safe_extract_zip(
    archive: str | Path,
    dest_dir: str | Path | None = None,
    *,
    members: Iterable[str] | None = None,
) -> Path:
    """Extract `archive` into `dest_dir`, rejecting any member that would escape it.

    Guards against zip-slip: a crafted archive member with a path like
    `../../evil.txt` or an absolute path is rejected before anything is
    written, rather than being silently extracted outside `dest_dir`.
    """
    archive = Path(archive)
    dest_dir = Path(dest_dir).resolve() if dest_dir is not None else archive.parent.resolve()

    with zipfile.ZipFile(archive) as zf:
        names = list(members) if members is not None else zf.namelist()
        for name in names:
            member_path = (dest_dir / name).resolve()
            if not (member_path == dest_dir or member_path.is_relative_to(dest_dir)):
                raise ValueError(f"Unsafe path in zip archive (zip-slip): {name!r}")
        zf.extractall(dest_dir, members=names)

    return dest_dir


# --------------------------------------------------------------------- #
# Spot-grid geometry
# --------------------------------------------------------------------- #


def xy_bounds(df, key_x: str = "x", key_y: str = "y") -> tuple[float, float, float, float]:
    """Return `(x_min, y_min, x_max, y_max)` for `df[key_x]`/`df[key_y]`.

    Works transparently with eager (pandas) or lazy (dask) frames: any
    result exposing a `.compute()` method (dask scalars) is computed.
    """
    x_min, x_max = df[key_x].min(), df[key_x].max()
    y_min, y_max = df[key_y].min(), df[key_y].max()
    x_min, x_max, y_min, y_max = (v.compute() if hasattr(v, "compute") else v for v in (x_min, x_max, y_min, y_max))
    return float(x_min), float(y_min), float(x_max), float(y_max)


def bin_centroids(x: np.ndarray, y: np.ndarray, spot_size_um: float) -> tuple[np.ndarray, np.ndarray]:
    """Bin `(x, y)` points into a square grid of side `spot_size_um`.

    Returns the centroid coordinates of every grid cell that contains at
    least one point, as `(centroid_x, centroid_y)` arrays.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        return np.empty(0), np.empty(0)

    x_min, x_max = x.min(), x.max()
    y_min = y.min()

    # Guard against a degenerate (zero-width) extent, which would otherwise
    # divide by zero when recovering column indices from flattened bin ids.
    # `+ 1` (not `ceil`) because `col_idx` below is a *floor* division: when
    # the extent is an exact multiple of `spot_size_um`, `ceil` undercounts
    # by one and the rightmost column's points alias onto row+1/col 0.
    n_cols = max(1, math.floor((x_max - x_min) / spot_size_um) + 1)

    col_idx = np.floor((x - x_min) / spot_size_um).astype(int)
    row_idx = np.floor((y - y_min) / spot_size_um).astype(int)
    spot_idx = row_idx * n_cols + col_idx
    unique_idx = np.unique(spot_idx)

    centroid_x = x_min + (unique_idx % n_cols) * spot_size_um + spot_size_um / 2
    centroid_y = y_min + np.floor(unique_idx / n_cols) * spot_size_um + spot_size_um / 2
    return centroid_x, centroid_y


def hex_grid_centroids(
    bounds: tuple[float, float, float, float],
    spot_size_um: float,
    overlap: float | None = 0.0,
) -> np.ndarray:
    """Generate a pointy-top hexagonal grid of centroids covering `bounds`.

    Parameters
    ----------
    bounds : tuple[float, float, float, float]
        `(x_min, y_min, x_max, y_max)`, as returned by :func:`xy_bounds`.
    spot_size_um : float
        Hexagon diameter (distance between opposite vertices), in the same
        units as `bounds`.
    overlap : float, optional
        Fractional overlap between adjacent hexagons, applied by reducing
        centroid spacing while keeping hexagon size constant. Default 0.0.

    Returns
    -------
    np.ndarray
        Array of shape `(N, 2)` with one `(x, y)` centroid per row. Empty
        (`shape (0, 2)`) if the grid has no cells.
    """
    overlap = overlap or 0.0
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0.0, 1.0), found {overlap}")

    x_min, y_min, x_max, y_max = bounds
    s = spot_size_um / 2.0  # side length
    hex_width = np.sqrt(3) * s
    hex_height = 2 * s

    dx = hex_width * (1 - overlap)
    dy = hex_height * 3 / 4 * (1 - overlap)

    x_range = np.arange(x_min - dx, x_max + dx, dx)
    y_range = np.arange(y_min - dy, y_max + dy, dy)

    centroids = []
    for i, y_val in enumerate(y_range):
        x_offset = dx / 2 if i % 2 else 0
        for x_val in x_range:
            centroids.append((x_val + x_offset, y_val))

    if not centroids:
        return np.empty((0, 2))
    return np.asarray(centroids)


def hexagon_vertices(center_x: float, center_y: float, side_length: float) -> np.ndarray:
    """Return the 6 vertices of a pointy-top hexagon as an `(6, 2)` array."""
    angles = np.deg2rad(60 * np.arange(6) + 30)  # +30 degrees for pointy-top orientation
    x = center_x + side_length * np.cos(angles)
    y = center_y + side_length * np.sin(angles)
    return np.column_stack([x, y])


def affine_from_point_pairs(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Compute a 3x3 affine matrix mapping `src_pts` to `dst_pts` (3 point pairs)."""
    import cv2

    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    matrix = cv2.getAffineTransform(src, dst)
    return np.vstack((matrix, [0, 0, 1]))


# --------------------------------------------------------------------- #
# Heavy helpers (spatialdata / geopandas / shapely / hestcore / openslide)
# --------------------------------------------------------------------- #


def fix_table_validation_errors(adata):
    """Rename `adata.var` columns that fail SpatialData table validation.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to validate and fix.

    Returns
    -------
    anndata.AnnData
        The AnnData object with fixed table attributes.
    """
    from spatialdata._core.validation import ValidationError, validate_table_attr_keys

    try:
        validate_table_attr_keys(adata)
    except ValidationError as e:
        invalid_columns = {error.location[1]: transform_name(error.location[1]) for error in e._errors}
        logger.warning("Fixing invalid column names: %s", invalid_columns)
        adata.var = adata.var.rename(columns=invalid_columns, errors="raise")

    return adata


def slide_to_numpy(slide, level: int = 0) -> np.ndarray:
    """Convert an `openslide.OpenSlide` object to a NumPy array.

    Parameters
    ----------
    slide : openslide.OpenSlide
        An open slide handle.
    level : int, optional
        The resolution level to read from the slide. Level 0 is the highest resolution.

    Returns
    -------
    np.ndarray
        The image from the specified level.
    """
    dims = slide.level_dimensions[level]
    image_pil = slide.read_region((0, 0), level, dims).convert("RGB")
    return np.array(image_pil)


def _mask_to_gdf(mask: np.ndarray, *, pixel_size: float):
    """Turn a binary tissue mask into a `GeoDataFrame` of polygons (in physical units).

    Uses `cv2.RETR_CCOMP`, which yields a two-level contour hierarchy: each
    top-level (parent == -1) contour is a tissue piece, and its direct
    children are holes cut out of it. Contour points from `cv2.findContours`
    are already `(x, y)` = `(col, row)`, matching shapely/GeoPandas convention.
    """
    import cv2
    import geopandas as gpd
    from shapely.geometry import Polygon

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return gpd.GeoDataFrame({"tissue_id": []}, geometry=[])
    hierarchy = hierarchy[0]  # (N, 4): (next, previous, first_child, parent)

    polygons = []
    tissue_ids: list[int] = []
    for idx, contour in enumerate(contours):
        if hierarchy[idx][3] != -1 or len(contour) < 3:
            continue  # a hole (has a parent), handled below via its parent; or degenerate

        holes = []
        child_idx = hierarchy[idx][2]
        while child_idx != -1:
            child = contours[child_idx]
            if len(child) >= 3:
                holes.append(child[:, 0, :] * pixel_size)
            child_idx = hierarchy[child_idx][0]

        polygon = Polygon(contour[:, 0, :] * pixel_size, holes=holes or None)
        if not polygon.is_valid:
            # Self-intersections at pixel corners shared between adjacent holes (or a
            # hole and the exterior ring) are common when a contour has many holes --
            # `buffer(0)` is the standard repair and preserves the true hole-subtracted
            # area (may return a Polygon or MultiPolygon). Dropping instead of repairing
            # silently discarded most of the tissue area on real, hole-heavy slides.
            polygon = polygon.buffer(0)
        if polygon.area > 0:
            polygons.append(polygon)
            tissue_ids.append(len(tissue_ids))

    return gpd.GeoDataFrame({"tissue_id": tissue_ids}, geometry=polygons)


def segment_tissue(
    wsi_path: str | Path,
    *,
    pixel_size: float,
    thumbnail_width: int = 2000,
    method: str = "otsu",
):
    """Compute a tissue mask for a whole-slide image.

    Parameters
    ----------
    wsi_path : str | Path
        Path to the whole-slide image.
    pixel_size : float
        Physical size of one pixel (e.g. microns per pixel), used to scale contours.
    thumbnail_width : int, optional
        Size at which the segmentation is performed. Default 2000.
    method : str, optional
        Segmentation method; only `"otsu"` is currently supported. Default `"otsu"`.

    Returns
    -------
    gpd.GeoDataFrame
        Tissue contours, with a `tissue_id` column indicating which tissue
        piece each contour belongs to.
    """
    import cv2
    from openslide import open_slide

    if method not in ("otsu",):
        raise ValueError(f"method can only be one of these: ['otsu'], found {method}")

    numpy_wsi = slide_to_numpy(open_slide(str(wsi_path)))
    height, width = numpy_wsi.shape[:2]
    scale = thumbnail_width / width
    thumbnail = cv2.resize(numpy_wsi, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)

    # H&E tissue is stained (darker/more saturated) against a bright white-glass
    # background, so Otsu on grayscale luminance separates the two cleanly;
    # THRESH_BINARY_INV keeps the darker (tissue) side as the foreground (1).
    gray = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = mask.astype(np.uint8)

    # Close small holes/gaps left by the thresholding before extracting contours.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    tissue_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    return _mask_to_gdf(tissue_mask, pixel_size=pixel_size)


def create_circular_spots(
    df,
    spot_size_um: float = 55.0,
    key_x: str = "x",
    key_y: str = "y",
):
    """Pool points into square-binned pseudo-spots and return them as circles.

    This function bins points into a regular grid, calculates the centroid
    coordinates of grid cells that contain at least one point, and creates
    circular spot polygons of diameter `spot_size_um` at those centroids.

    Parameters
    ----------
    df : pd.DataFrame | dask.dataframe.DataFrame
        A dataframe containing `key_x`/`key_y` coordinate columns.
    spot_size_um : float, optional
        The side length of the square pooling grid. Default 55.0.
    key_x : str, optional
        Column name for the x-coordinate. Default `"x"`.
    key_y : str, optional
        Column name for the y-coordinate. Default `"y"`.

    Returns
    -------
    gpd.GeoDataFrame
        Circular polygons at each occupied grid cell's centroid, with
        centroid coordinates in `x_um`/`y_um`.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    try:
        from dask import dataframe as dd

        is_dask = isinstance(df, dd.DataFrame)
    except ImportError:  # pragma: no cover - dask is a declared dependency
        is_dask = False

    if is_dask:
        x_arr = df[key_x].compute().to_numpy()
        y_arr = df[key_y].compute().to_numpy()
    else:
        x_arr = np.asarray(df[key_x])
        y_arr = np.asarray(df[key_y])

    centroid_x, centroid_y = bin_centroids(x_arr, y_arr, spot_size_um)
    coord_df = pd.DataFrame({"x_um": centroid_x, "y_um": centroid_y})

    radius = spot_size_um / 2
    circles = [Point(xy).buffer(radius) for xy in zip(coord_df["x_um"], coord_df["y_um"], strict=False)]
    gdf = gpd.GeoDataFrame(coord_df, geometry=circles)

    logger.info("Created %d pseudo-spots of size %sµm", len(gdf), spot_size_um)
    return gdf


def create_hexagonal_spots(
    df,
    spot_size_um: float = 55.0,
    key_x: str = "x",
    key_y: str = "y",
    overlap: float | None = 0.0,
):
    """Cover the extent of `df` with hexagonal pseudo-spots.

    Parameters
    ----------
    df : pd.DataFrame | dask.dataframe.DataFrame
        A dataframe containing `key_x`/`key_y` coordinate columns; only its
        coordinate extent is used.
    spot_size_um : float, optional
        The diameter of the hexagonal spot (distance between opposite
        vertices). Default 55.0.
    key_x : str, optional
        Column name for the x-coordinate. Default `"x"`.
    key_y : str, optional
        Column name for the y-coordinate. Default `"y"`.
    overlap : float, optional
        Fractional overlap between adjacent hexagons (e.g. 0.06 for 6%
        overlap), applied by reducing centroid spacing while keeping
        hexagon size constant. Default 0.0.

    Returns
    -------
    gpd.GeoDataFrame
        Hexagonal polygons covering `df`'s extent, with centroid
        coordinates in `x_um`/`y_um`.
    """
    import geopandas as gpd
    from shapely.geometry import Polygon

    overlap = overlap or 0.0
    bounds = xy_bounds(df, key_x, key_y)
    s = spot_size_um / 2.0
    centroids = hex_grid_centroids(bounds, spot_size_um, overlap)

    if centroids.size == 0:
        logger.warning("No hexagons generated. Check input data and spot size.")
        return gpd.GeoDataFrame(columns=["x_um", "y_um", "geometry"])

    hexagons = [Polygon(hexagon_vertices(cx, cy, s)) for cx, cy in centroids]
    coord_df = pd.DataFrame(centroids, columns=["x_um", "y_um"])
    gdf = gpd.GeoDataFrame(coord_df, geometry=hexagons)

    overlap_msg = f" with {overlap * 100:.1f}% overlap" if overlap > 0 else ""
    logger.info("Created %d hexagonal pseudo-spots of size %sµm%s", len(gdf), spot_size_um, overlap_msg)
    return gdf
