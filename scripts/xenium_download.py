#!/usr/bin/env python3
"""Fetch a Xenium study's raw asset bundle from a `curl -O <url>` manifest.

Thin CLI wrapper around `spatialrefinery.io.xenium.download_xenium_study`
(built on `spatialrefinery.core.downloader.BaseDownloader`), which
retries transient failures, writes atomically, and unzips `*_outs.zip`
assets in place (skipping `*_xe_outs.zip`). All of that logic now lives
in the package; this script only parses arguments.
"""

import argparse
import sys
from pathlib import Path

from spatialrefinery.io.xenium import download_xenium_study


def main() -> None:
    """Parse arguments and download the Xenium study assets they describe."""
    parser = argparse.ArgumentParser(
        description="Fetch Xenium study files listed as 'curl -O <URL>' lines and unzip zips in-place."
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Path to the text file containing lines like: curl -O https://.../file.zip",
    )
    parser.add_argument(
        "-o", "--outdir", type=Path, default=Path.cwd(), help="Output base directory (default: current directory)"
    )
    parser.add_argument("-w", "--workers", type=int, default=8, help="Number of parallel download workers (default: 8)")
    args = parser.parse_args()

    try:
        results = download_xenium_study(args.input_file, args.outdir, max_workers=args.workers)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

    failed = [r for r in results if r.status == "failed"]
    if failed:
        print(f"\n{len(failed)}/{len(results)} asset(s) failed:", file=sys.stderr)
        for r in failed:
            print(f"  - {r.asset.url}: {r.error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
