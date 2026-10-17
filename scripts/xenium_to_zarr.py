"""Convert one Xenium bundle to a zipped SpatialData zarr.

For batch runs over a panel tree (Slurm arrays, completed/failed logs, mirrored output layout) use
``scripts/xenium_panel_array_to_zarr.py`` instead.
"""

import argparse
import logging
import os
import shutil
import sys
import warnings
from pathlib import Path

from spatialrefinery.io.xenium import xenium_to_spatialdata_zip

# Set environment variable first
os.environ["PYTHONWARNINGS"] = "ignore"

# Suppress all warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger("xenium_to_zarr")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        help="paths to data raw files",
        default="/p/project1/hai_spatial_clip/data/phoenix_datasets/raw_files",
    )

    parser.add_argument(
        "--biospecimen-id",
        help="The ID/name of the single biospecimen to process (must match the directory name).",
        type=str,
        default="Xenium_Prime_Human_Ovary_FF",
        required=False,
    )

    parser.add_argument(
        "--output-path",
        help="paths to save the zarr files",
        default="/p/scratch/hai_spatial_clip/xenium_22_10_2025/ovary_files",
    )

    parser.add_argument(
        "--values",
        help="Which element to aggregate into the pseudo-spots.",
        type=str,
        default="transcripts",
        choices=["transcripts", "cell_boundaries"],
        required=False,
    )

    parser.add_argument(
        "--spot-sizes",
        nargs="+",
        type=int,
        default=[55, 100],
        help="Pseudo-spot diameters in micrometres.",
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.06,
        help="Fractional overlap between adjacent hexagonal spots.",
    )

    parser.add_argument(
        "--num-workers",
        help="Workers passed to spatialdata_io.xenium as n_jobs.",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--keep-zarr",
        help="Keep the unzipped .zarr directory next to the zip.",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        help="Reconvert even if the output zip already exists.",
        action="store_true",
    )

    return parser


def main(args) -> int:
    """Convert the specified biospecimen's Xenium output directory to a zipped SpatialData zarr store."""
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    biospecimen_id = args.biospecimen_id

    biospecimen_path = data_path / biospecimen_id
    if not biospecimen_path.is_dir():
        logger.error("Directory not found for biospecimen ID: %s", biospecimen_path)
        return 1

    output_path.mkdir(parents=True, exist_ok=True)

    # A zarr without its zip is the debris of an interrupted run. Leaving it in place would make
    # xenium_to_spatialdata short-circuit base creation and then fail writing an already-present
    # spots_<size>um element, so clear it first.
    zarr_path = output_path / f"{biospecimen_id}.zarr"
    zip_path = output_path / f"{biospecimen_id}.zarr.zip"
    if zarr_path.exists() and (args.overwrite or not zip_path.exists()):
        logger.warning("Removing existing zarr before conversion: %s", zarr_path)
        shutil.rmtree(zarr_path)

    zip_path = xenium_to_spatialdata_zip(
        dataset_path=biospecimen_path,
        output_path=output_path,
        output_name=biospecimen_id,
        include_aligned_image=True,
        create_spots=True,
        spot_sizes=args.spot_sizes,
        overlap=args.overlap,
        values=args.values,
        n_jobs=args.num_workers,
        keep_zarr=args.keep_zarr,
        overwrite=args.overwrite,
    )

    logger.info("Created zip file at: %s (%.2f GB)", zip_path, Path(zip_path).stat().st_size / (1024**3))
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    parser = parse_args()
    sys.exit(main(parser.parse_args()))
