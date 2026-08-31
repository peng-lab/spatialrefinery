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


def hex_lattice_params(
    bounds: tuple[float, float, float, float],
    spot_size_um: float,
    overlap: float | None = 0.0,
) -> dict:
    """Describe the pointy-top hexagonal lattice that covers `bounds`.

    This is the single definition of the lattice: :func:`hex_grid_centroids`
    materialises it as centroids, and :func:`assign_points_to_hexes` uses it to
    decide membership analytically. Keeping both on top of this function is what
    guarantees the two can never drift apart.

    Centroid `k` of the grid sits at row `i = k // nx`, column `j = k % nx`, with
    odd rows shifted right by `dx / 2`.

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
    dict
        `s` (circumradius), `dx`/`dy` (centroid spacing), `x_range`/`y_range`
        (row/column origins) and `nx`/`ny` (grid shape).
    """
    overlap = overlap or 0.0
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0.0, 1.0), found {overlap}")

    x_min, y_min, x_max, y_max = bounds
    s = spot_size_um / 2.0  # side length == circumradius
    hex_width = np.sqrt(3) * s
    hex_height = 2 * s

    dx = hex_width * (1 - overlap)
    dy = hex_height * 3 / 4 * (1 - overlap)

    x_range = np.arange(x_min - dx, x_max + dx, dx)
    y_range = np.arange(y_min - dy, y_max + dy, dy)

    return {
        "s": s,
        "dx": dx,
        "dy": dy,
        "x_range": x_range,
        "y_range": y_range,
        "nx": len(x_range),
        "ny": len(y_range),
    }


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
        Array of shape `(N, 2)` with one `(x, y)` centroid per row, ordered
        row-major (all of row 0 left to right, then row 1, ...). Empty
        (`shape (0, 2)`) if the grid has no cells.
    """
    lattice = hex_lattice_params(bounds, spot_size_um, overlap)
    x_range, y_range = lattice["x_range"], lattice["y_range"]
    nx, ny = lattice["nx"], lattice["ny"]

    if nx == 0 or ny == 0:
        return np.empty((0, 2))

    rows = np.repeat(np.arange(ny), nx)
    x_offset = np.where(rows % 2 == 1, lattice["dx"] / 2, 0.0)
    return np.column_stack([np.tile(x_range, ny) + x_offset, np.repeat(y_range, nx)])


def hexagon_vertices(center_x: float, center_y: float, side_length: float) -> np.ndarray:
    """Return the 6 vertices of a pointy-top hexagon as an `(6, 2)` array."""
    angles = np.deg2rad(60 * np.arange(6) + 30)  # +30 degrees for pointy-top orientation
    x = center_x + side_length * np.cos(angles)
    y = center_y + side_length * np.sin(angles)
    return np.column_stack([x, y])


def hex_candidate_offsets(overlap: float | None = 0.0) -> tuple[range, range]:
    """Return the `(row, column)` lattice offsets a point could fall into.

    A point at lattice cell `(i0, j0)` can only lie inside hexagons whose
    centroid is within one circumradius `s` of it. With centroid spacing
    `dy = 1.5 * s * (1 - overlap)` and `dx = sqrt(3) * s * (1 - overlap)`, that
    is always rows `{i0, i0 + 1}` and columns `{j0, j0 + 1}`; wider overlaps
    bring further neighbours into reach, so the span is derived from `overlap`
    rather than hard-coded (the `0.06` used in practice needs only the base
    2x2 block).

    Returns
    -------
    tuple[range, range]
        `(row_offsets, column_offsets)` to try around the containing cell.
    """
    overlap = overlap or 0.0
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0.0, 1.0), found {overlap}")

    # s / dy and (sqrt(3)/2 * s) / dx respectively, i.e. how many extra whole
    # centroid spacings fit inside the hexagon's reach.
    extra_rows = int(np.floor(1.0 / (1.5 * (1 - overlap))))
    extra_cols = int(np.floor(1.0 / (2.0 * (1 - overlap))))
    return range(-extra_rows, 2 + extra_rows), range(-extra_cols, 2 + extra_cols)


def assign_points_to_hexes(
    x: np.ndarray,
    y: np.ndarray,
    lattice: dict,
    overlap: float | None = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Map points to every lattice hexagon that contains them.

    The lattice is regular, so membership is a closed-form test rather than a
    spatial join: no geometry objects are built and nothing is indexed. A point
    on a shared edge is reported for both hexagons, matching the `intersects`
    predicate `GeoDataFrame.sjoin` uses. When `overlap > 0` the hexagons
    genuinely overlap, so a point can be returned for more than one spot -- the
    same duplication the sjoin produces.

    Parameters
    ----------
    x, y : np.ndarray
        Point coordinates, in the units `lattice` was built in.
    lattice : dict
        As returned by :func:`hex_lattice_params`.
    overlap : float, optional
        The overlap `lattice` was built with; sets how far to search.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        `(point_positions, spot_ids)`, where `spot_ids` index
        :func:`hex_grid_centroids` output for the same lattice.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    s, dx, dy = lattice["s"], lattice["dx"], lattice["dy"]
    nx, ny = lattice["nx"], lattice["ny"]
    if nx == 0 or ny == 0 or x.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    x0, y0 = lattice["x_range"][0], lattice["y_range"][0]
    half_width = np.sqrt(3) / 2 * s
    inv_sqrt3 = 1.0 / np.sqrt(3)

    row_offsets, col_offsets = hex_candidate_offsets(overlap)
    base_row = np.floor((y - y0) / dy).astype(np.int64)

    positions: list[np.ndarray] = []
    spot_ids: list[np.ndarray] = []
    for row_offset in row_offsets:
        row = base_row + row_offset
        row_ok = (row >= 0) & (row < ny)
        if not row_ok.any():
            continue
        centre_y = y0 + row * dy
        abs_dy = np.abs(y - centre_y)
        # Odd rows are shifted right by half a spacing, so the column origin
        # depends on the row -- recompute it rather than reusing base_row's.
        x_offset = np.where(row % 2 == 1, dx / 2, 0.0)
        base_col = np.floor((x - x0 - x_offset) / dx).astype(np.int64)
        for col_offset in col_offsets:
            col = base_col + col_offset
            centre_x = x0 + x_offset + col * dx
            abs_dx = np.abs(x - centre_x)
            inside = row_ok & (col >= 0) & (col < nx) & (abs_dx <= half_width) & (abs_dy <= s - abs_dx * inv_sqrt3)
            if inside.any():
                hit = np.flatnonzero(inside)
                positions.append(hit)
                spot_ids.append(row[hit] * nx + col[hit])

    if not positions:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positions), np.concatenate(spot_ids)


def _merge_pair_counts(pending, keys, counts):
    """Fold a list of raw pair keys into running `(unique keys, counts)` totals."""
    stacked = [*pending]
    weights = [np.ones(part.size, dtype=np.int64) for part in pending]
    if keys.size:
        stacked.append(keys)
        weights.append(counts)

    all_keys = np.concatenate(stacked)
    if all_keys.size == 0:
        return keys, counts
    all_weights = np.concatenate(weights)

    order = np.argsort(all_keys, kind="stable")
    all_keys = all_keys[order]
    all_weights = all_weights[order]
    unique_keys, starts = np.unique(all_keys, return_index=True)
    return unique_keys, np.add.reduceat(all_weights, starts)


def bin_points_to_hex_counts(
    batches,
    lattice: dict,
    n_spots: int,
    n_genes: int,
    overlap: float | None = 0.0,
    flush_pairs: int = 256_000_000,
):
    """Accumulate per-(spot, gene) counts by streaming batches of points.

    Peak memory is one batch plus the running set of non-zero cells, so this
    scales to billion-transcript sections that cannot be held in memory at
    once. `spatialdata.aggregate` cannot: it builds one shapely `Point` per
    transcript and then groups with `observed=False`, which materialises the
    full `n_spots x n_genes` product regardless of how sparse the data is.

    Parameters
    ----------
    batches : Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]
        `(x, y, gene_codes)` triples. `gene_codes` index the gene axis;
        negative codes are dropped, which is how unmapped features are skipped.
    lattice : dict
        As returned by :func:`hex_lattice_params`.
    n_spots, n_genes : int
        Shape of the matrix to build.
    overlap : float, optional
        The overlap `lattice` was built with.
    flush_pairs : int, optional
        Merge pending `(spot, gene)` pairs into the running totals once this
        many have accumulated. Bounds the merge buffer; does not change output.
        Each merge re-sorts the running totals, so this trades peak memory
        against the number of merges: a 2.7e9-transcript section flushes ~12
        times at the default and ~48 times at a quarter of it, with the running
        array growing to hundreds of millions of cells either way.

    Returns
    -------
    scipy.sparse.csr_matrix
        Counts of shape `(n_spots, n_genes)`.
    """
    from scipy import sparse

    # One integer key per (spot, gene) cell keeps each merge a single sort
    # rather than a lexsort over two columns.
    running_keys = np.empty(0, dtype=np.int64)
    running_counts = np.empty(0, dtype=np.int64)
    pending: list[np.ndarray] = []
    pending_size = 0

    for batch_x, batch_y, gene_codes in batches:
        positions, spot_ids = assign_points_to_hexes(batch_x, batch_y, lattice, overlap)
        if positions.size == 0:
            continue
        codes = np.asarray(gene_codes)[positions].astype(np.int64)
        keep = codes >= 0
        if not keep.all():
            spot_ids = spot_ids[keep]
            codes = codes[keep]
        if spot_ids.size == 0:
            continue

        pending.append(spot_ids * n_genes + codes)
        pending_size += spot_ids.size
        if pending_size >= flush_pairs:
            running_keys, running_counts = _merge_pair_counts(pending, running_keys, running_counts)
            pending, pending_size = [], 0

    if pending:
        running_keys, running_counts = _merge_pair_counts(pending, running_keys, running_counts)

    if running_keys.size == 0:
        return sparse.csr_matrix((n_spots, n_genes), dtype=np.int64)

    rows, cols = np.divmod(running_keys, n_genes)
    return sparse.coo_matrix((running_counts, (rows, cols)), shape=(n_spots, n_genes), dtype=np.int64).tocsr()


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


def decode_bytes_columns(points):
    """Decode bytes-valued ``object`` columns of a points frame to pandas strings.

    Older Xenium outputs (the "Preview" / "With_Addon" bundles) store
    `cell_id` and `fov_name` in `transcripts.parquet` as parquet *binary*
    rather than *string*, so `spatialdata_io.xenium` hands back a dask frame
    whose columns carry `bytes` under a bare `object` dtype.

    That combination cannot be written. `SpatialData.write` sends points
    through `dask.dataframe.to_parquet`, which infers the Arrow schema from
    `meta_nonempty(df._meta)`; for a bare `object` dtype dask fills the dummy
    frame with a literal `object()` sentinel, and pyarrow raises
    `ArrowInvalid: ... did not recognize Python value type` before any real
    data is touched. Re-typing the columns as `string` makes both the dummy
    frame and the real values inferable.

    Parameters
    ----------
    points : dask.dataframe.DataFrame
        A points element, typically `sdata["transcripts"]`.

    Returns
    -------
    dask.dataframe.DataFrame
        The same element with every `object` column re-typed to `string`, and
        its coordinate transformations preserved. Returned unchanged when
        there are no `object` columns.
    """
    from spatialdata.models import PointsModel
    from spatialdata.transformations import get_transformation, set_transformation

    object_columns = [name for name, dtype in points.dtypes.items() if pd.api.types.is_object_dtype(dtype)]
    if not object_columns:
        return points

    logger.info("Re-typing bytes/object columns to string: %s", object_columns)

    # get_all=True so a frame registered in several coordinate systems keeps
    # every one of them, not just "global".
    transformations = get_transformation(points, get_all=True)

    decoded = points
    for column in object_columns:
        decoded[column] = decoded[column].map(
            lambda value: value.decode("utf-8") if isinstance(value, bytes) else value,
            meta=(column, "string"),
        )

    # map() drops the PointsModel attrs, so re-parse and re-attach the
    # transformations captured above.
    decoded = PointsModel.parse(decoded)
    set_transformation(decoded, transformations, set_all=True)

    return decoded


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
    bounds: tuple[float, float, float, float] | None = None,
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
    bounds : tuple[float, float, float, float], optional
        Precomputed `(x_min, y_min, x_max, y_max)` for `df`. Pass this when the
        caller already has the extent: deriving it from a lazy dask frame costs
        a full scan, and callers that also need the lattice want both to come
        from the same numbers.

    Returns
    -------
    gpd.GeoDataFrame
        Hexagonal polygons covering `df`'s extent, with centroid
        coordinates in `x_um`/`y_um`.
    """
    import geopandas as gpd
    from shapely.geometry import Polygon

    overlap = overlap or 0.0
    if bounds is None:
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
