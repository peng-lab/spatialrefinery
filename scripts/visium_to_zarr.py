"""Convert one extracted Visium / CytAssist bundle to a zipped SpatialData zarr.

Segment the H&E first (``scripts/instanseg_segment.py``) with the pixel size this script prints, then pass the
resulting ``cells.geojson`` as ``--nucleus-geojson``. Without it the store has the spot side only.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

from spatialrefinery.io.visium import (
    DEFAULT_TEMPLATE_ADATA,
    find_visium_files,
    visium_he_mpp,
    visium_to_spatialdata_zip,
)

logger = logging.getLogger("visium_to_zarr")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        help="Directory holding the extracted bundles, one subdirectory per sample.",
        default="/p/project1/hai_1240/datasets/flagship/raw_output_bundle/cytassist_WT/TENX",
    )
    parser.add_argument(
        "--biospecimen-id",
        help="The sample to process (must match the bundle's directory name).",
        type=str,
        required=True,
    )
    parser.add_argument("--output-path", help="Directory to save the zarr and its zip in.", required=True)
    parser.add_argument(
        "--nucleus-geojson",
        help="InstanSeg cells.geojson for the H&E; adds nucleus_boundaries and the zero-filled table.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--template-adata",
        help="h5ad whose var is the gene panel of the nucleus table (default: the multi-tissue panel).",
        type=Path,
        default=DEFAULT_TEMPLATE_ADATA,
    )
    parser.add_argument("--keep-zarr", help="Keep the unzipped .zarr directory next to the zip.", action="store_true")
    parser.add_argument("--overwrite", help="Reconvert even if the output zip already exists.", action="store_true")
    return parser


def main(args) -> int:
    """Convert the specified biospecimen's Visium bundle to a zipped SpatialData zarr store."""
    bundle_path = Path(args.data_path) / args.biospecimen_id
    if not bundle_path.is_dir():
        logger.error("Directory not found for biospecimen ID: %s", bundle_path)
        return 1

    files = find_visium_files(bundle_path)
    if files["tissue_positions"] is None:
        logger.error("No spatial/tissue_positions.csv in %s; extract *_spatial.tar.gz first.", bundle_path)
        return 1
    logger.info("Segment the H&E with pixel_size=%.4f um/px", visium_he_mpp(files["tissue_positions"]))

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    zarr_path = output_path / f"{args.biospecimen_id}.zarr"
    zip_path = output_path / f"{args.biospecimen_id}.zarr.zip"
    if zip_path.exists() and not args.overwrite:
        logger.info("Skipping %s: %s already exists", args.biospecimen_id, zip_path)
        return 0

    # A zarr without its zip is the debris of an interrupted run; clear it so the write below starts clean.
    if zarr_path.exists() and (args.overwrite or not zip_path.exists()):
        logger.warning("Removing existing zarr before conversion: %s", zarr_path)
        shutil.rmtree(zarr_path)

    zip_path = visium_to_spatialdata_zip(
        bundle_path,
        output_path,
        args.biospecimen_id,
        nucleus_geojson=args.nucleus_geojson,
        template_adata_path=args.template_adata,
        overwrite=args.overwrite,
        keep_zarr=args.keep_zarr,
    )

    logger.info("Created zip file at: %s (%.2f GB)", zip_path, Path(zip_path).stat().st_size / (1024**3))
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    sys.exit(main(parse_args().parse_args()))
