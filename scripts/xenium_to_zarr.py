import argparse
import os
import warnings
from pathlib import Path

import dask

# Import our custom function
from spatialrefinery.io.xenium import xenium_to_spatialdata_zip

# Set environment variable first
os.environ["PYTHONWARNINGS"] = "ignore"

# Configure dask
dask.config.set({"dataframe.query-planning": False})

# Suppress all warnings
warnings.filterwarnings("ignore")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
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
        help="Values used for aggregation",
        type=str,
        default="transcripts",
        required=False,
    )

    parser.add_argument(
        "--render-figures", help="Plot a figure using spatialdata-plot", action="store_true", default=True
    )

    parser.add_argument(
        "--num-workers",
        help="Number of workers to use for processing images.",
        type=int,
        default=1,
    )

    return parser


def main(args):
    """Convert the specified biospecimen's Xenium output directory to a zipped SpatialData zarr store."""
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    biospecimen_id = args.biospecimen_id

    os.makedirs(output_path, exist_ok=True)

    # Find paths for the single specified sample
    biospecimen_path = data_path / biospecimen_id
    if not biospecimen_path.is_dir():
        print(f"Error: Directory not found for biospecimen ID: {biospecimen_path}")
        return

    zip_path = xenium_to_spatialdata_zip(
        dataset_path=biospecimen_path,
        output_path=output_path,
        output_name=biospecimen_id,
        include_aligned_image=True,
        create_spots=True,
        spot_sizes=[100, 55],
        overlap=0.06,
        values=args.values,
        n_jobs=args.num_workers,
        keep_zarr=True,  # Keep the zarr directory after zipping
        overwrite=False,
    )

    print(f"✅ Created zip file at: {zip_path}")
    print(f"   Zip size: {Path(zip_path).stat().st_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    main(args)
