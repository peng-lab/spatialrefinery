#!/usr/bin/env python3
r"""Convert a nucleus-segmentation GeoJSON plus its slide into a SpatialData zarr.

Thin CLI wrapper around
`spatialrefinery.segmentation.to_spatialdata.geojson_to_spatialdata`.
Pairs with `instanseg_segment.py`, consuming the `cells.geojson` it writes.

Usage
-----
    python geojson_to_spatialdata.py \\
        --geojson-path results/slide.ome.tif/cells.geojson \\
        --zarr-outdir zarrs/ --wsi-path slide.ome.tif --template-adata template.h5ad

The store is named for the slide's stem, so the example above writes
`zarrs/slide.zarr` (plus `zarrs/slide.zarr.zip`).

Stores written before this convention are named for the full filename
(`slide.ome.tif.zarr`). The skip-existing check only looks for the new name, so
re-running rebuilds them rather than skipping; rename them instead:

    for z in "$ZARR_OUTDIR"/*.ome.tif.zarr; do echo "$z -> ${z%.ome.tif.zarr}.zarr"; done  # dry run
    for z in "$ZARR_OUTDIR"/*.ome.tif.zarr; do mv "$z" "${z%.ome.tif.zarr}.zarr"; done
    for z in "$ZARR_OUTDIR"/*.ome.tif.zarr.zip; do mv "$z" "${z%.ome.tif.zarr.zip}.zarr.zip"; done

A zarr store records no path of its own, so a rename is all that is needed.
"""

import argparse
import logging
import sys
from pathlib import Path

from spatialrefinery.segmentation.to_spatialdata import default_zarr_path, geojson_to_spatialdata

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="SpatialData conversion for a single segmented WSI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--geojson-path", required=True)
    parser.add_argument("--zarr-outdir", required=True)
    parser.add_argument("--wsi-path", required=True)
    parser.add_argument("--template-adata", required=True)
    parser.add_argument("--no-zip", action="store_true", help="Skip the .zarr.zip archive")
    parser.add_argument("--no-skip-existing", action="store_true", help="Rebuild even if the zarr exists")
    return parser


def main() -> None:
    """Convert one slide's segmentation into a SpatialData zarr."""
    args = build_parser().parse_args()

    geojson_path = Path(args.geojson_path)
    wsi_path = Path(args.wsi_path)
    template_adata = Path(args.template_adata)

    for label, path in (("GeoJSON", geojson_path), ("WSI", wsi_path), ("Template AnnData", template_adata)):
        if not path.exists():
            logger.error("%s file not found: %s", label, path)
            sys.exit(1)

    # Named for the stem, so `slide.ome.tif` gives `slide.zarr`. Unlike the
    # segmentation stage's per-slide directory, which keeps the full filename.
    zarr_path = default_zarr_path(wsi_path, args.zarr_outdir)
    if zarr_path.exists() and not args.no_skip_existing:
        logger.info("Skipping %s: %s already exists", wsi_path.name, zarr_path)
        sys.exit(0)

    Path(args.zarr_outdir).mkdir(parents=True, exist_ok=True)

    try:
        geojson_to_spatialdata(
            geojson_path=geojson_path,
            zarr_path=zarr_path,
            image_path=wsi_path,
            template_adata_path=template_adata,
            write_zip=not args.no_zip,
        )
    except Exception:
        logger.exception("Conversion failed for %s", wsi_path.name)
        sys.exit(1)

    print(f"ZARR_PATH={zarr_path}")


if __name__ == "__main__":
    main()
