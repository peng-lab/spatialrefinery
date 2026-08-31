#!/usr/bin/env python3
"""Segment nuclei in one H&E whole-slide image with InstanSeg.

Thin CLI wrapper around `spatialrefinery.segmentation.instanseg.segment_wsi`.

Prints `GEOJSON_PATH=<path>` on success -- the SLURM worker greps for it to
hand the result to the conversion stage.

Usage
-----
    python instanseg_segment.py --wsi-path slide.svs --outdir results/
    python instanseg_segment.py --wsi-path slide.ome.tif --outdir results/ --wsi-mpp 0.27
"""

import argparse
import logging
import sys
from pathlib import Path

from spatialrefinery.segmentation.instanseg import (
    DEFAULT_DETECTION_SIZE,
    DEFAULT_MODEL,
    DEFAULT_OVERLAP,
    DEFAULT_TILE_SIZE,
    segment_wsi,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="InstanSeg nucleus segmentation for a single WSI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--wsi-path", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA index; -1 forces CPU")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--wsi-mpp", type=float, default=None, help="Microns per pixel; read from metadata if omitted")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--detection-size", type=int, default=DEFAULT_DETECTION_SIZE)
    parser.add_argument(
        "--clahe",
        type=float,
        default=None,
        metavar="CLIP",
        help=(
            "Run CLAHE at this clip limit over each tile before inference (e.g. 2.0). "
            "Off by default; helps on weakly haematoxylin-stained slides where pale "
            "nuclei are missed."
        ),
    )
    parser.add_argument(
        "--seed-threshold",
        type=float,
        default=None,
        help="Override the model seed threshold (default 0.7). Lower detects fainter nuclei.",
    )
    parser.add_argument(
        "--no-otsu",
        action="store_true",
        help="Segment every tile instead of only those inside the tissue mask",
    )
    parser.add_argument("--no-skip-existing", action="store_true", help="Re-segment even if cells.geojson exists")
    return parser


def main() -> None:
    """Run segmentation for one slide and print its GeoJSON path."""
    args = build_parser().parse_args()

    wsi_path = Path(args.wsi_path)
    if not wsi_path.exists():
        logger.error("WSI file not found: %s", wsi_path)
        sys.exit(1)

    try:
        geojson_path = segment_wsi(
            wsi_path,
            args.outdir,
            pixel_size=args.wsi_mpp,
            gpu_id=None if args.gpu_id < 0 else args.gpu_id,
            model_type=args.model,
            tile_size=args.tile_size,
            overlap=args.overlap,
            detection_size=args.detection_size,
            use_otsu_threshold=not args.no_otsu,
            clahe_clip=args.clahe,
            seed_threshold=args.seed_threshold,
            skip_existing=not args.no_skip_existing,
        )
    except Exception:
        logger.exception("Segmentation failed for %s", wsi_path.name)
        sys.exit(1)

    # Contract with slurm/segment_node_worker.sh -- keep this line last.
    print(f"GEOJSON_PATH={geojson_path}")


if __name__ == "__main__":
    main()
