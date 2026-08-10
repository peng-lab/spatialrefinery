#!/usr/bin/env python3
"""Convert whole-slide/microscopy images to pyramidal OME-TIFF.

Thin CLI wrapper around `spatialrefinery.core.converter.convert_to_ometiff`,
which dispatches on file suffix to the registered converter
(`OpenSlideImageConverter` for SVS/NDPI/TIFF/..., `AICSImageConverter` for
CZI). This replaces the three near-identical per-format scripts that used
to live in this directory (`czi_to_ometiff.py`, `svs_to_ometiff.py`,
`ndpi_to_ometiff.py`) -- all of the actual conversion logic now lives in
the package, in one place, with one (fixed) pyramid-resolution formula.

Usage
-----
    python convert_to_ometiff.py --input_path slide.svs --output_dir out/
    python convert_to_ometiff.py --input_path wsi_dir/ --output_dir out/ -p 4
"""

import argparse
import logging
from pathlib import Path

from spatialrefinery.core.converter import convert_to_ometiff
from spatialrefinery.core.registry import RegistryError, get_converter_for, list_converters

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SUBRESOLUTIONS = 4


def find_convertible_files(input_path: Path) -> list[Path]:
    """Return every file under `input_path` with a registered converter suffix."""
    if input_path.is_file():
        return [input_path]

    suffixes = list_converters().keys()
    files = []
    for suffix in suffixes:
        files.extend(input_path.glob(f"*{suffix}"))
        files.extend(input_path.glob(f"*{suffix.upper()}"))
    return sorted(set(files))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert whole-slide/microscopy images to pyramidal OME-TIFF")
    parser.add_argument("--input_path", type=Path, required=True, help="Image file or directory of images")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory (default: next to input)")
    parser.add_argument(
        "-p",
        "--pyramid-levels",
        type=int,
        default=DEFAULT_SUBRESOLUTIONS,
        help=f"Number of pyramid subresolutions (default: {DEFAULT_SUBRESOLUTIONS})",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate outputs that already exist")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def main() -> int:
    """Find convertible files under the given input path and convert each to OME-TIFF."""
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = args.input_path.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_convertible_files(input_path)
    if not files:
        logger.error("No convertible files found: %s (known suffixes: %s)", input_path, ", ".join(list_converters()))
        return 1

    all_outputs: list[Path] = []
    for file in files:
        try:
            get_converter_for(file)  # fail fast with a clear message before doing any work
        except RegistryError as e:
            logger.error("Skipping %s: %s", file, e)
            continue

        logger.info("Processing: %s", file)
        try:
            outputs = convert_to_ometiff(file, output_dir, subresolutions=args.pyramid_levels, overwrite=args.overwrite)
            all_outputs.extend(outputs)
        except Exception as e:  # noqa: BLE001 - one file's failure must not abort the whole batch
            logger.error("Failed to convert %s: %s", file, e)

    logger.info("Created %d OME-TIFF file(s)", len(all_outputs))
    for f in all_outputs:
        logger.info("  %s", f)

    return 0


if __name__ == "__main__":
    exit(main())
