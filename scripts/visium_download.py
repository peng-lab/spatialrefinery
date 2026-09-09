#!/usr/bin/env python3
r"""Fetch a 10x Visium / CytAssist study's raw asset bundle from a `curl -O <url>` manifest.

Thin CLI wrapper around `spatialrefinery.io.visium.download_visium_study`.
Assets land under `<outdir>/<study>/`, with `spatial.tar.gz`,
`analysis.tar.gz` and `deconvolution.tar.gz` unpacked in place beside them;
the two `*_feature_bc_matrix.tar.gz` archives are left packed, being MTX
copies of the `.h5` files fetched alongside.

Usage
-----
    # See what a manifest resolves to, without touching the network
    python visium_download.py --manifest 10x_visium_human.txt -o raw/ --dry-run

    # Fetch just what a conversion needs, for two studies
    python visium_download.py --manifest 10x_visium_human.txt -o raw/ \\
        --studies CytAssist_11mm_FFPE_Human_Kidney CytAssist_FFPE_Human_Colon_Rep1 \\
        --kinds spatial filtered_matrix tissue_image cloupe

Each study logs a completeness report once its assets settle, naming any
required kind that is missing and whether a `.cloupe` was found. A study
with an incomplete upstream manifest is reported, not fatal -- but an asset
whose `spatial` archive will not unpack counts as failed, and exits nonzero.
"""

import argparse
import logging
import sys
from pathlib import Path

from spatialrefinery.io.visium import download_visium_study

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Fetch Visium study files listed as 'curl -O <URL>' lines and unpack tarballs in place.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Text file of lines like: curl -O https://.../<study>_spatial.tar.gz",
    )
    parser.add_argument(
        "-o", "--outdir", type=Path, default=Path.cwd(), help="Output base directory; assets go to <outdir>/<study>/"
    )
    parser.add_argument("--studies", nargs="+", help="Restrict to these study names (default: every study)")
    parser.add_argument("--kinds", nargs="+", help="Restrict to these asset kinds (default: every kind)")
    parser.add_argument("-w", "--workers", type=int, default=8, help="Parallel download workers")
    parser.add_argument("--retries", type=int, default=3, help="Retry attempts per asset on transient failure")
    parser.add_argument("--timeout", type=int, default=180, help="Per-request timeout in seconds")
    parser.add_argument("--no-extract", action="store_true", help="Download tarballs without unpacking them")
    parser.add_argument("--overwrite", action="store_true", help="Re-download assets already on disk")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be fetched, without any network use")
    return parser


def main() -> None:
    """Parse arguments and download the Visium study assets they describe."""
    args = build_parser().parse_args()

    if not args.manifest.is_file():
        logger.error("Manifest file not found: %s", args.manifest)
        sys.exit(1)

    try:
        results = download_visium_study(
            args.manifest,
            args.outdir,
            studies=args.studies,
            kinds=args.kinds,
            max_workers=args.workers,
            retries=args.retries,
            timeout=args.timeout,
            extract=not args.no_extract,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        sys.exit(130)

    failed = [r for r in results if r.status == "failed"]
    if failed:
        logger.error("%d/%d asset(s) failed:", len(failed), len(results))
        for r in failed:
            logger.error("  - %s: %s", r.asset.url, r.error)
        sys.exit(1)


if __name__ == "__main__":
    main()
