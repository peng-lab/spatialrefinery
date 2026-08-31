"""Turn a nucleus-segmentation GeoJSON plus its slide into a SpatialData zarr.

The image element is built directly from the slide's own pyramid with
`tifffile`'s zarr interface, which keeps the read lazy -- a level-0 plane on a
typical whole slide is 33427 x 11949 x 3, about 1.1 GiB if materialised.

The table is all-zero counts over a template's `var`; it exists so the shapes
element carries a SpatialData-valid annotation. Nucleus segmentation produces
no expression data, so no cell-type column is written.
"""

from __future__ import annotations

import json
import logging
import os
import zipfile
from pathlib import Path
from typing import cast

logger = logging.getLogger(__name__)

#: Suffixes `tifffile` can open directly. SVS and NDPI are TIFF containers, so
#: the whole family goes through the same lazy path.
_TIFF_SUFFIXES = (".tif", ".tiff", ".svs", ".ndpi", ".scn", ".bif", ".svslide")

DEFAULT_SCALE_FACTORS = (2, 2, 2, 2)


def explode_multipolygons(geojson_path: str | Path, save_path: str | Path):
    """Split multi-polygons into individual polygons, preserving all attributes.

    Returns the exploded frame and writes it to `save_path`.
    """
    import geopandas as gpd
    from shapely.geometry import Polygon, shape

    with open(geojson_path) as f:
        features = json.load(f)

    geometries = []
    properties = []

    for feature in features:
        if "geometry" not in feature:
            continue
        geom = feature["geometry"]
        props = feature.get("properties", {})

        if geom["type"] == "Polygon":
            coords = geom["coordinates"]
            geometries.append(Polygon(coords[0], coords[1:] if len(coords) > 1 else None))
            properties.append(props)
        elif geom["type"] == "MultiPolygon":
            for poly_coords in geom["coordinates"]:
                geometries.append(Polygon(poly_coords[0], poly_coords[1:] if len(poly_coords) > 1 else None))
                properties.append(props)
        else:
            geometries.append(shape(geom))
            properties.append(props)

    gdf = gpd.GeoDataFrame(properties, geometry=geometries)

    # The loop above already flattens, so this is close to a no-op; kept because
    # a non-Polygon geometry can still arrive as a collection via `shape()`.
    exploded_gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    exploded_gdf.to_file(save_path, driver="GeoJSON")

    return exploded_gdf


def wsi_image_element(image_path: str | Path, *, scale_factors=DEFAULT_SCALE_FACTORS):
    """Build a lazy multiscale `Image2DModel` element from a whole-slide image.

    `tifffile.imread(aszarr=True)` exposes the file's own chunking, so `dask`
    reads only the blocks a write actually touches rather than pulling the
    level-0 plane into RAM.
    """
    import dask.array as da
    import tifffile
    from spatialdata.models import Image2DModel

    image_path = Path(image_path)
    if not any(image_path.name.lower().endswith(s) for s in _TIFF_SUFFIXES):
        raise ValueError(
            f"{image_path.name} is not a TIFF-backed slide. Convert it first with "
            "`spatialrefinery.core.converter.convert_to_ometiff`."
        )

    store = tifffile.imread(str(image_path), aszarr=True, level=0)
    arr = da.from_zarr(store)

    # tifffile hands back (y, x, sample) for RGB slides; SpatialData wants (c, y, x).
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3].transpose(2, 0, 1)
    elif arr.ndim == 2:
        arr = arr[None, ...]

    channels = ["r", "g", "b"][: arr.shape[0]]
    return Image2DModel.parse(
        arr,
        dims=("c", "y", "x"),
        c_coords=channels,
        scale_factors=list(scale_factors),
    )


def default_zarr_path(wsi_path: str | Path, zarr_outdir: str | Path) -> Path:
    """Return `<zarr_outdir>/<slide stem>.zarr`, the store name one slide converts to.

    The stem is the prefix before the slide's extension, so `slide.ome.tif`
    gives `slide.zarr` rather than `slide.ome.tif.zarr` or `slide.ome.zarr`.
    Note this differs from the segmentation stage, whose per-slide directory
    keeps the full filename (see `segmentation.instanseg.segment_wsi`).
    """
    from spatialrefinery.core.utils import slide_stem

    return Path(zarr_outdir) / f"{slide_stem(wsi_path)}.zarr"


def geojson_to_spatialdata(
    geojson_path: str | Path,
    zarr_path: str | Path,
    image_path: str | Path,
    template_adata_path: str | Path,
    *,
    write_zip: bool = True,
):
    """Assemble and write the SpatialData zarr for one segmented slide.

    The table is all-zero counts over the template's `var`; it exists so the
    shapes element carries a SpatialData-valid annotation.
    """
    import anndata as ad
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import spatialdata
    from spatialdata.models import ShapesModel

    from spatialrefinery.core.utils import fix_table_validation_errors

    # GDAL truncates large GeoJSON features unless this is lifted.
    os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

    geojson_path = Path(geojson_path)
    zarr_path = Path(zarr_path)
    image_path = Path(image_path)

    temp_geojson = geojson_path.parent / f"{geojson_path.stem}_exploded.geojson"
    if temp_geojson.exists():
        logger.info("Reusing existing exploded GeoJSON: %s", temp_geojson)
    else:
        logger.info("Exploding multi-polygons from %s", geojson_path)
        explode_multipolygons(geojson_path, save_path=temp_geojson)

    logger.info("Reading exploded GeoJSON")
    gdf = gpd.read_file(temp_geojson)
    if gdf.empty:
        raise ValueError(f"No polygons found in {geojson_path}")

    # The GeoJSON spec declares EPSG:4326, so pyogrio tags the frame as
    # geographic -- but these are pixel coordinates on a slide, not lon/lat.
    # Left in place, `.centroid` warns and invites a spherical reprojection
    # that would silently corrupt every centroid.
    gdf = gdf.set_crs(None, allow_override=True)

    centroids = gdf.geometry.centroid
    spatial_coords = np.column_stack([centroids.x, centroids.y])
    gdf.index = gdf.index.astype(str)

    logger.info("Loading template AnnData: %s", template_adata_path)
    template_adata = ad.read_h5ad(template_adata_path)

    # anndata types `.var` as `DataFrame | Dataset2D`; the second arm only
    # occurs for a backed store, and `read_h5ad` above loads into memory.
    var = cast("pd.DataFrame", template_adata.var).copy()

    table = ad.AnnData(
        X=np.zeros((len(gdf), var.shape[0]), dtype=np.float32),
        obs=pd.DataFrame(index=gdf.index),
        var=var,
    )
    table.obsm["spatial"] = spatial_coords

    # InstanSeg tags every feature `object_type`; it carries no information
    # once the polygons are in a shapes element.
    shapes = gdf[[gdf.geometry.name]].copy()

    logger.info("Building SpatialData object (%d nuclei)", len(gdf))
    sdata = spatialdata.SpatialData(
        images={"he_image": wsi_image_element(image_path)},
        shapes={"nucleus_boundaries": ShapesModel.parse(shapes)},
        attrs={"tissue_segmentation_image": "he_image"},
        tables={"table": table},
    )

    sdata["table"] = fix_table_validation_errors(sdata["table"])
    sdata["table"].obs["region"] = "nucleus_boundaries"
    sdata["table"].obs["instance_id"] = sdata["table"].obs.index.astype(str)
    sdata.set_table_annotates_spatialelement(
        "table", region="nucleus_boundaries", instance_key="instance_id", region_key="region"
    )

    logger.info("Writing SpatialData zarr to %s", zarr_path)
    sdata.write(str(zarr_path))

    if write_zip:
        zip_path = Path(f"{zarr_path}.zip")
        if zip_path.exists():
            logger.warning("Zip already exists, leaving it alone: %s", zip_path)
        else:
            logger.info("Creating uncompressed zip archive")
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zipf:
                for root, _, files in os.walk(zarr_path):
                    for file in files:
                        file_path = Path(root) / file
                        zipf.write(file_path, file_path.relative_to(zarr_path.parent))

    temp_geojson.unlink(missing_ok=True)
    return sdata
