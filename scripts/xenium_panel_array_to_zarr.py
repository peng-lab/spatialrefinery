"""Convert a single Xenium bundle from a panel tree to a zipped SpatialData zarr.

Designed to be the body of a SLURM array task: one invocation handles exactly
one sample, mirrors the raw tree's ``<panel>/<cohort>/<sample>`` layout under the
output root, skips samples that are already done, and records success/failure to
shared append-only logs instead of aborting the wider run.
"""

import argparse
import fcntl
import logging
import os
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

from spatialrefinery.io.xenium import xenium_to_spatialdata_zip

# Set environment variable first
os.environ["PYTHONWARNINGS"] = "ignore"

# Suppress all warnings
warnings.filterwarnings("ignore")

# NB: scripts/xenium_to_zarr.py sets dask.config "dataframe.query-planning" to
# False. That is deliberately NOT done here -- with the pinned dask (2026.7.1)
# the legacy dataframe implementation is gone, and actually applying the setting
# before dask.dataframe is imported raises NotImplementedError. It only appears
# harmless in the older script because the top-level imports pull in
# dask.dataframe first, so the assignment never takes effect.

RAW_ROOT = "/p/project1/hai_1240/datasets/flagship/raw_output_bundle"
OUT_ROOT = "/p/scratch/hai_1240/sandbox/zarr_files"

logger = logging.getLogger("xenium_panel_array_to_zarr")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-dir",
        required=True,
        help="Path to a single Xenium bundle root (the directory holding experiment.xenium).",
    )
    parser.add_argument(
        "--raw-root",
        default=RAW_ROOT,
        help="Root of the raw panel tree; --sample-dir is made relative to this to mirror the layout.",
    )
    parser.add_argument(
        "--out-root",
        default=OUT_ROOT,
        help="Root under which the mirrored <panel>/<cohort>/<sample>.zarr.zip is written.",
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
        "--values",
        default="transcripts",
        choices=["transcripts", "cell_boundaries"],
        help="Which element to aggregate into the pseudo-spots.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Workers passed to spatialdata_io.xenium as n_jobs.",
    )
    parser.add_argument(
        "--fail-log",
        default="slurm-logs/failed_samples.tsv",
        help="Append-only TSV of samples whose conversion raised.",
    )
    parser.add_argument(
        "--done-log",
        default="slurm-logs/completed_samples.tsv",
        help="Append-only TSV of samples that produced a zip.",
    )
    parser.add_argument(
        "--max-spots-per-chunk",
        type=int,
        default=50_000,
        help="Deprecated and ignored; transcript aggregation is streamed and no longer tiled.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reconvert even if the output zip already exists.",
    )
    return parser


def append_log(path: str, row: list) -> None:
    """Append one tab-separated row to `path`, holding an exclusive lock.

    Array tasks run concurrently and share these files, so the lock is what
    keeps interleaved writes from corrupting each other's lines.
    """
    line = "\t".join(str(field).replace("\t", " ").replace("\n", " ") for field in row) + "\n"
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main(args) -> int:
    """Convert one sample; return the process exit code."""
    sample_dir = Path(args.sample_dir).resolve()
    raw_root = Path(args.raw_root).resolve()
    out_root = Path(args.out_root).resolve()

    if not sample_dir.is_dir():
        logger.error("Sample directory not found: %s", sample_dir)
        append_log(args.fail_log, [datetime.now(timezone.utc).isoformat(), "", "", str(sample_dir), "not a directory"])
        return 1

    try:
        rel = sample_dir.relative_to(raw_root)
    except ValueError:
        logger.error("--sample-dir %s is not under --raw-root %s", sample_dir, raw_root)
        return 2

    # Mirror <panel>/<cohort>/ on the output side; the sample name becomes the
    # basename of the zarr/zip rather than another directory level.
    out_dir = out_root / rel.parent
    name = rel.name
    panel, cohort = (list(rel.parts) + ["", ""])[:2]

    zarr_path = out_dir / f"{name}.zarr"
    zip_path = out_dir / f"{name}.zarr.zip"

    if zip_path.exists() and not args.overwrite:
        logger.info("Already converted, skipping: %s", zip_path)
        append_log(args.done_log, [datetime.now(timezone.utc).isoformat(), panel, cohort, name, "skipped", zip_path])
        return 0

    # A zarr without its zip is the debris of an interrupted run. Leaving it in
    # place would make xenium_to_spatialdata short-circuit base creation and then
    # fail writing an already-present spots_<size>um element, so clear it first.
    if zarr_path.exists():
        logger.warning("Removing stale partial zarr: %s", zarr_path)
        shutil.rmtree(zarr_path)

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Converting %s -> %s (spot sizes %s)", sample_dir, zip_path, args.spot_sizes)
    try:
        produced = xenium_to_spatialdata_zip(
            dataset_path=sample_dir,
            output_path=out_dir,
            output_name=name,
            include_aligned_image=True,
            create_spots=True,
            spot_sizes=args.spot_sizes,
            overlap=args.overlap,
            values=args.values,
            n_jobs=args.num_workers,
            max_spots_per_chunk=args.max_spots_per_chunk or None,
            keep_zarr=False,
            overwrite=args.overwrite,
        )
    except Exception as exc:  # logged and recorded so the wider array run continues
        logger.exception("Conversion failed for %s", sample_dir)
        append_log(
            args.fail_log,
            [datetime.now(timezone.utc).isoformat(), panel, cohort, name, f"{type(exc).__name__}: {exc}"],
        )
        return 1

    size_gb = Path(produced).stat().st_size / (1024**3)
    logger.info("Created %s (%.2f GB)", produced, size_gb)
    append_log(
        args.done_log,
        [datetime.now(timezone.utc).isoformat(), panel, cohort, name, f"{size_gb:.2f}GB", produced],
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    parser = parse_args()
    sys.exit(main(parser.parse_args()))
