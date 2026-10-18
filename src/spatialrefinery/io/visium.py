"""Download 10x Genomics Visium / CytAssist studies, verify the bundles, and convert them to SpatialData.

This module is the Visium half of what :mod:`spatialrefinery.io.xenium`
does for Xenium, and deliberately mirrors its shape: a suffix -> kind map,
a :class:`~spatialrefinery.core.downloader.BaseDownloader` subclass, and a
module-level ``download_*_study`` function that stays the canonical API.

Three things make Visium different from Xenium, and each is the reason for
a specific piece of code below.

**Bundles are tarballs, not zips.** Space Ranger ships ``spatial``,
``analysis`` and ``deconvolution`` as ``.tar.gz``. `Path.suffix` is
single-part, so `a_spatial.tar.gz` reports `.gz`; the extraction gate in
`core.downloader` therefore matches whole filenames against
`_ARCHIVE_SUFFIXES`, and `core.utils.safe_extract_tar` does the unpacking.
The archives are unpacked *in place*, giving `<bundle>/spatial/` beside the
flat `<study>_*.h5` assets -- which is exactly the layout
`spatialdata_io.visium` expects, so no staging directory and no symlinking
of the counts matrix is needed downstream.

**Two of the tarballs are redundant.** `*_filtered_feature_bc_matrix.tar.gz`
and `*_raw_feature_bc_matrix.tar.gz` are MTX triplets of the `.h5` files
downloaded beside them, so unpacking them roughly doubles a bundle's size
for no new information. They sit in `never_extract`, the same mechanism
that keeps Xenium's `*_xe_outs.zip` packed.

**A manifested study is not necessarily a complete one.** In the
13-study human cohort this module was written against, `deconvolution` is
present for 6 studies, `alignment_file` for 7, and one study
(`Visium_V2_Human_Lymph_Node`) ships neither a `.cloupe` nor any counts
matrix at all. So missing assets are *reported*, never raised: a hard
failure would abort a whole batch on the one study whose upstream manifest
is incomplete. `VisiumDownloader.verify_bundle` does that reporting, and reads
the filesystem rather than the download results, so it confirms the tars
genuinely produced `spatial/scalefactors_json.json` instead of echoing
what was requested.
"""

from __future__ import annotations

import logging
import os
import shutil
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd

from spatialrefinery.core.converter import SpatialDataConverter
from spatialrefinery.core.downloader import BaseDownloader, DownloadResult, RemoteAsset
from spatialrefinery.core.registry import TechnologySpec, register_technology
from spatialrefinery.core.utils import (
    fix_table_validation_errors,
    parse_curl_manifest,
    safe_extract_tar,
    sample_attrs,
    segment_tissue,
    split_feature_types,
    tar_root_dir,
)

#: Physical diameter of a Visium / CytAssist capture spot, 10x's nominal 55 um. It is the radius of the `spots`
#: circles and of the disk a nucleus must fall in to be assigned to a spot.
SPOT_DIAMETER_UM = 55.0

#: Centre-to-centre spot pitch, 100 um on every 4,992- and 14,336-spot chip (64 or 112 spots per row span the 6.5
#: and 11 mm capture areas). It, not the spot diameter, sets the H&E pixel size (see :func:`visium_he_mpp`). With
#: 55 um spots it leaves ~45 um gaps, which is why a cell can belong to no spot at all.
SPOT_PITCH_UM = 100.0

#: The H&E pixel size must land in this range (um/px), or the scale factors are not describing a ~20-40x slide.
HE_MPP_RANGE = (0.1, 1.0)

#: Largest fraction of in-tissue spots allowed to lie more than one spot radius outside the H&E before the store is
#: refused. A few such spots are normal (the microscope image can be cropped inside the capture area; 0.02-0.09% in the
#: 11 mm CytAssist bundles), and they are kept. A misregistered lattice puts a large share of the spots outside.
MAX_SPOTS_OUTSIDE_IMAGE = 0.01

#: Template whose `var` is the gene panel of the (all-zero) nucleus table, unless a caller asks for another.
DEFAULT_TEMPLATE_ADATA = Path("/p/project1/hai_1240/datasets/templates/multi_tissue.h5ad")

#: Width in pixels of the pyramid level `segment_tissue` is asked to read for the tissue contours.
TISSUE_THUMBNAIL_WIDTH = 2000

#: The widest level `segment_tissue` may be left with. A raw 10x tissue image is single-level, so the nearest
#: level to `TISSUE_THUMBNAIL_WIDTH` is level 0 (~30-55k px wide); Otsu then yields so many noisy full-resolution
#: contours that repairing them with `shapely.buffer(0)` had not finished after 30 minutes on a 39k px slide.
MAX_SEGMENTATION_WIDTH = 8000

_COUNTS_SUFFIX = "_filtered_feature_bc_matrix.h5"

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Asset classification
# --------------------------------------------------------------------- #

#: Maps a filename suffix (as used in 10x's Space Ranger manifests) to a
#: semantic asset kind. Longest suffix wins, which two pairs here rely on:
#: `_filtered_feature_bc_matrix.tar.gz` and `_raw_feature_bc_matrix.tar.gz`
#: share the tail `_feature_bc_matrix.tar.gz`, and `_tissue_image.tif` must
#: beat `_image.tif`. Adding a shorter key that is a suffix of an existing
#: one -- a bare `_feature_bc_matrix.tar.gz`, say -- would silently
#: misclassify every longer sibling, so don't.
_VISIUM_KINDS: dict[str, str] = {
    # Archives, unpacked in place beside the flat assets.
    "_spatial.tar.gz": "spatial",
    "_analysis.tar.gz": "analysis",
    "_deconvolution.tar.gz": "deconvolution",
    "_filtered_feature_bc_matrix.tar.gz": "filtered_matrix_mtx",
    "_raw_feature_bc_matrix.tar.gz": "raw_matrix_mtx",
    # Count matrices.
    "_filtered_feature_bc_matrix.h5": "filtered_matrix",
    "_raw_feature_bc_matrix.h5": "raw_matrix",
    "_raw_probe_bc_matrix.h5": "raw_probe_matrix",
    "_molecule_info.h5": "molecule_info",
    # Images. `_tissue_image.*` is the high-resolution microscope image (the
    # H&E, or a multi-channel immunofluorescence stack); `_image.tif` is the
    # CytAssist instrument capture, which is a different, much coarser pixel
    # grid and is not registered to the spots without the alignment JSON.
    # Conflating them would silently train on the wrong image.
    "_tissue_image.ome.tif": "tissue_image",
    "_tissue_image.tif": "tissue_image",
    "_tissue_image.btf": "tissue_image",
    "_image.tif": "cytassist_image",
    # Metadata and provenance.
    "_alignment_file.json": "alignment",
    "_probe_set.csv": "probe_set",
    "_feature_reference.csv": "feature_reference",
    "_isotype_normalization_factors.csv": "isotype_normalization",
    "_metrics_summary.csv": "metrics_summary",
    "_web_summary.html": "web_summary",
    "_cloupe.cloupe": "cloupe",
}
_VISIUM_KINDS_BY_LENGTH = sorted(_VISIUM_KINDS, key=len, reverse=True)


def classify_visium_asset(filename: str) -> str:
    """Return the asset kind for `filename`, or `"unknown"` if unrecognised.

    Module-level rather than a method because bundle verification classifies
    files already on disk, with no downloader in play.

    Parameters
    ----------
    filename : str
        A bare filename, e.g. `"CytAssist_FFPE_Human_Colon_Rep1_spatial.tar.gz"`.

    Returns
    -------
    str
        One of the values of `_VISIUM_KINDS`, or `"unknown"`.
    """
    for suffix in _VISIUM_KINDS_BY_LENGTH:
        if filename.endswith(suffix):
            return _VISIUM_KINDS[suffix]
    return "unknown"


# --------------------------------------------------------------------- #
# Bundle inspection
# --------------------------------------------------------------------- #


def find_visium_files(dataset_path: str | Path) -> dict[str, Path | None]:
    """Locate the members of a flat, `<study>_`-prefixed Visium bundle.

    Unlike the notebook prototype this grew from, nothing here raises on a
    missing file: a bundle lacking its counts matrix is precisely what
    `VisiumDownloader.verify_bundle` exists to report, so absence is data, not
    an error. Every value is therefore `Path | None`, matching
    :func:`spatialrefinery.io.xenium.find_xenium_files` and satisfying
    `TechnologySpec.file_finder`.

    Parameters
    ----------
    dataset_path : str | Path
        The bundle directory, i.e. `outdir/<study>` after a download.

    Returns
    -------
    dict[str, Path | None]
        Paths to the bundle members, `None` where a member is absent. The
        `scalefactors` and `tissue_positions` entries point *inside* the
        unpacked `spatial/`, so a caller can tell a downloaded tarball from
        an extracted one.
    """
    dataset_path = Path(dataset_path)

    def first(*patterns: str, exclude: str | None = None) -> Path | None:
        # sorted() so a directory holding several matches resolves the same
        # way on every filesystem, rather than following readdir order.
        for pattern in patterns:
            hits = sorted(dataset_path.glob(pattern))
            if exclude is not None:
                hits = [p for p in hits if not p.match(exclude)]
            if hits:
                return hits[0]
        return None

    return {
        "counts_h5": first("*_filtered_feature_bc_matrix.h5"),
        "raw_counts_h5": first("*_raw_feature_bc_matrix.h5"),
        # `.ome.tif` first: a pyramidal copy made by `convert_to_ometiff` reads fastest and carries the
        # real pixel size. Then the vendor `.tif`, then `.btf` (OME-BigTIFF, which some bundles ship instead
        # of a `.tif`). Whichever is found, `visium_to_spatialdata` takes the pixel size from the scale
        # factors, never from the TIFF tags.
        "tissue_image": first("*_tissue_image.ome.tif", "*_tissue_image.tif", "*_tissue_image.btf"),
        # `*_image.tif` also matches `*_tissue_image.tif`, hence the exclusion.
        # Without it the microscope image would be reported as the CytAssist
        # capture and the two would be indistinguishable.
        "cytassist_image": first("*_image.tif", exclude="*_tissue_image.tif"),
        "alignment": first("*_alignment_file.json"),
        "probe_set": first("*_probe_set.csv"),
        "feature_reference": first("*_feature_reference.csv"),
        "isotype_factors": first("*_isotype_normalization_factors.csv"),
        "cloupe": first("*_cloupe.cloupe"),
        "spatial_archive": first("*_spatial.tar.gz"),
        "scalefactors": first("spatial/scalefactors_json.json"),
        # `tissue_positions.csv` is Space Ranger >= 2.0; the headerless
        # `tissue_positions_list.csv` is the 1.x name, kept so an older
        # bundle is recognised as extracted rather than reported broken.
        "tissue_positions": first("spatial/tissue_positions.csv", "spatial/tissue_positions_list.csv"),
    }


# --------------------------------------------------------------------- #
# Download: fetch a Visium study's raw asset bundle
# --------------------------------------------------------------------- #


class VisiumDownloader(BaseDownloader):
    """Fetch a Visium study's raw asset bundle from a `curl -O <url>` manifest.

    `source` passed to `iter_assets`/`plan`/`run` may be a manifest file
    path (see `example_data/10x_visium_human.txt`) or any iterable of URLs.
    """

    technology = "visium"
    # The two `*_feature_bc_matrix.tar.gz` archives are MTX triplets of the
    # `.h5` files fetched beside them, so unpacking them roughly doubles a
    # bundle's size and inode count for no new information.
    never_extract = frozenset({"filtered_matrix_mtx", "raw_matrix_mtx"})
    # Everything downstream reads `spatial/`, so a `spatial.tar.gz` that will
    # not unpack is a failed asset, not a warning to scroll past.
    require_extract = frozenset({"spatial"})

    classify: ClassVar = staticmethod(classify_visium_asset)

    # The spot geometry, the counts, and the microscope image: all three are
    # present for every study in the reference 13-study cohort, so a missing
    # one means an incomplete download or an incomplete upstream manifest.
    required_kinds = ("spatial", "filtered_matrix", "tissue_image")
    # `analysis` is absent for 1 of the 13. `deconvolution` (6/13) and
    # `alignment` (7/13) are deliberately excluded -- genuinely optional
    # upstream, and warning every run would train the reader to ignore the
    # warnings that matter. `cloupe` has its own warning below, so listing it
    # here too would log the same fact twice.
    expected_kinds = ("analysis", "cytassist_image")
    # Proves `<study>_spatial.tar.gz` actually unpacked. The glob covers both
    # names: `tissue_positions.csv` from Space Ranger >= 2.0, and the
    # headerless `tissue_positions_list.csv` of 1.x.
    required_members = ("spatial/scalefactors_json.json", "spatial/tissue_positions*.csv")

    def iter_assets(self, source: str | Path | Iterable[str]) -> Iterator[RemoteAsset]:
        """Yield one `RemoteAsset` per URL in `source` (a manifest path or an iterable of URLs)."""
        if isinstance(source, str | Path) and Path(source).is_file():
            urls: Iterable[str] = parse_curl_manifest(source)
        elif isinstance(source, Path):
            raise FileNotFoundError(f"Manifest file not found: {source}")
        else:
            urls = source

        for url in urls:
            asset = RemoteAsset.from_url(url)
            yield RemoteAsset(
                url=asset.url,
                study=asset.study,
                filename=asset.filename,
                kind=classify_visium_asset(asset.filename),
            )

    def extract_archive(self, asset: RemoteAsset, path: Path) -> Path:
        """Unpack a Visium tarball, giving a *flat* archive a directory of its own.

        "Extract in place" only yields `<bundle>/spatial/` if the archive
        brings its own top-level directory -- a property of the archive, not
        of the destination. A flat archive (`matrix.mtx.gz` at its root)
        would overwrite any sibling shipping a member of the same name, and
        because `fetch` runs in a thread pool the two could be writing
        concurrently, so the result would not even be deterministic. When
        the layout is not self-contained the contents go under
        `<bundle>/<kind>/` instead, where nothing else can reach them.
        """
        if path.name.lower().endswith(".zip"):
            return super().extract_archive(asset, path)

        root = tar_root_dir(path)
        if root is not None:
            return safe_extract_tar(path, path.parent)

        target = path.parent / asset.kind
        logger.warning(
            "%s has no single top-level directory; extracting into %s/ so it cannot overwrite a sibling",
            path.name,
            asset.kind,
        )
        return safe_extract_tar(path, target)

    def post_process(self, study: str, results: Sequence[DownloadResult]) -> None:
        """Verify the bundle, then call out a missing `.cloupe` by name.

        The cloupe gets its own line because it is the usual place to look for
        a slide's provenance, and its absence is easy to miss in a list of
        kinds. Note that microns-per-pixel does not depend on it: it is
        derivable from `spatial/scalefactors_json.json`.
        """
        super().post_process(study, results)
        if not self.dry_run and find_visium_files(self.outdir / study)["cloupe"] is None:
            logger.warning("%s: no *_cloupe.cloupe file", study)


def download_visium_study(
    source: str | Path | Iterable[str],
    outdir: str | Path,
    *,
    studies: list[str] | None = None,
    kinds: list[str] | None = None,
    max_workers: int = 8,
    retries: int = 3,
    timeout: int = 180,
    extract: bool = True,
    overwrite: bool = False,
    resume: bool = True,
    dry_run: bool = False,
) -> list[DownloadResult]:
    """Download a Visium / CytAssist study's raw asset bundle.

    Parameters
    ----------
    source : str | Path | Iterable[str]
        Path to a manifest file of `curl -O <url>` lines (one per asset,
        as downloaded from 10x's dataset pages), or an iterable of URLs.
    outdir : str | Path
        Base directory; assets land under `outdir/<study>/<filename>`, with
        `spatial/`, `analysis/` and `deconvolution/` unpacked beside them.
    studies : list[str], optional
        Restrict to these study names. Default: all studies in `source`.
    kinds : list[str], optional
        Restrict to these asset kinds (e.g. `["spatial", "filtered_matrix"]`).
        Default: all kinds.
    max_workers : int, optional
        Parallel download workers. Default 8.
    retries : int, optional
        Number of retry attempts per asset on transient failure. Default 3.
    timeout : int, optional
        Per-request timeout in seconds. Default 180.
    extract : bool, optional
        Whether to unpack `.tar.gz` assets in place after downloading. The
        two `*_feature_bc_matrix.tar.gz` archives are always skipped, being
        MTX copies of the `.h5` files. Default True.
    overwrite : bool, optional
        Re-download assets that already exist on disk. Default False.
    resume : bool, optional
        Continue a partially-downloaded asset from its leftover `.part`
        file using an HTTP `Range` request instead of refetching it from
        the beginning. Default True.
    dry_run : bool, optional
        Resolve and report what would be downloaded without any network
        activity. Bundle verification is skipped, there being nothing on
        disk to verify. Default False.

    Returns
    -------
    list[DownloadResult]
        One result per selected asset. Each result's `.status` is one of
        `"downloaded"`, `"cached"` (already present, `overwrite=False`),
        `"skipped"` (excluded by `dry_run`), or `"failed"`; `.ok` is a bool
        convenience property, and `.error` holds the exception message for
        failed downloads. An asset whose bytes arrived but whose required
        archive would not unpack is `"failed"` with `.path` still set.
    """
    downloader = VisiumDownloader(
        outdir,
        max_workers=max_workers,
        retries=retries,
        timeout=timeout,
        extract=extract,
        overwrite=overwrite,
        resume=resume,
        dry_run=dry_run,
    )
    return downloader.run(source, studies=studies, kinds=kinds)


# --------------------------------------------------------------------- #
# Conversion: a Visium bundle -> one SpatialData zarr store
# --------------------------------------------------------------------- #


def _restore_antibody_names(table, counts_h5: Path):
    """Undo the reader's renaming of antibodies that share a gene's name.

    `spatialdata_io.visium` calls `var_names_make_unique`, which turns an antibody `CD4` (a gene of that
    name comes first in the matrix) into `CD4-1`, and a second antibody of the same name into `CD4-2`. Those
    suffixes would become the `protein_expression` column names, so `CD4` the antibody could no longer be
    matched to `CD4` the gene. The names are read back from the h5 and made unique among antibodies only,
    which the later move to `obsm` makes safe.
    """
    import h5py
    from anndata.utils import make_index_unique

    with h5py.File(counts_h5, "r") as f:
        names = f["matrix/features/name"][:].astype(str)
        is_antibody = f["matrix/features/feature_type"][:].astype(str) == "Antibody Capture"
    # `gex_only=False` keeps every feature in file order, so the h5 rows and `table.var` line up one to one.
    if len(names) != table.n_vars:
        raise ValueError(f"{counts_h5}: {len(names)} features in the file but {table.n_vars} in the table.")
    if is_antibody.any():
        var_names = np.asarray(table.var_names, dtype=object)
        var_names[is_antibody] = make_index_unique(pd.Index(names[is_antibody]))
        table.var_names = var_names
    return table


def _require_segmentable_pyramid(he_path: Path, he_mpp: float) -> None:
    """Raise unless the H&E has a level `segment_tissue` can contour in reasonable time.

    Raw 10x tissue images are single-level, so they must go through `convert_to_ometiff` first; the
    `*_tissue_image.ome.tif` it writes is what :func:`find_visium_files` then prefers.

    Raises
    ------
    ValueError
        If the level nearest to :data:`TISSUE_THUMBNAIL_WIDTH` is wider than :data:`MAX_SEGMENTATION_WIDTH`.
    """
    from tiffslide import TiffSlide

    with TiffSlide(str(he_path)) as slide:
        widths = [width for width, _ in slide.level_dimensions]
    nearest = min(widths, key=lambda width: abs(width - TISSUE_THUMBNAIL_WIDTH))
    if nearest > MAX_SEGMENTATION_WIDTH:
        raise ValueError(
            f"{he_path.name} has no pyramid level near {TISSUE_THUMBNAIL_WIDTH} px wide (levels: {widths}), so "
            "tissue segmentation would run on the full-resolution plane and not finish. Convert it first:\n"
            f"  python scripts/convert_to_ometiff.py --input_path {he_path} --output_dir {he_path.parent} "
            f"--mpp {he_mpp:.4f}\n"
            "(--mpp is required: the file's own resolution tag is the 96-dpi default.)"
        )


def read_tissue_positions(tissue_positions_path: str | Path) -> pd.DataFrame:
    """Read `tissue_positions.csv` (Space Ranger >= 2, with header) or `tissue_positions_list.csv` (1.x, none).

    Returns
    -------
    pandas.DataFrame
        One row per barcode of the whole capture-area lattice, in or out of tissue, indexed by barcode, with
        `in_tissue`, `array_row`, `array_col`, `pxl_row_in_fullres` and `pxl_col_in_fullres`.
    """
    columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    raw = pd.read_csv(tissue_positions_path, header=None, names=columns)
    if raw.iloc[0, 0] == "barcode":
        raw = raw.iloc[1:]
    return raw.set_index("barcode").astype(float)


def visium_he_mpp(tissue_positions_path: str | Path) -> float:
    """Return the H&E pixel size in um/px from the spacing of the spot lattice in `tissue_positions.csv`.

    The lattice has a fixed physical pitch (:data:`SPOT_PITCH_UM`), so the median nearest-neighbour distance
    between spot centres, in the image's own pixels, gives the pixel size. Two tempting alternatives are wrong:

    - the tissue image's TIFF tags hold the 96-dpi default (264.58 um/px);
    - `55 / spot_diameter_fullres` from the scale factors splits the 13 cohort bundles by *chip* (0.215 vs
      0.251 um/px) rather than by image, and would make a 4,992-spot lattice 5.0 mm wide where 64 spots at
      100 um need 6.4 mm. The lattice pitch is a constant ~364 px in all of them.

    The whole lattice is used, off-tissue spots included, so a small or fragmented tissue cannot skew it.

    Raises
    ------
    ValueError
        If the result is outside :data:`HE_MPP_RANGE`.
    """
    from scipy.spatial import cKDTree

    positions = read_tissue_positions(tissue_positions_path)
    centres = positions[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()
    pitch_px = float(np.median(cKDTree(centres).query(centres, k=2)[0][:, 1]))
    he_mpp = SPOT_PITCH_UM / pitch_px
    lo, hi = HE_MPP_RANGE
    if not lo <= he_mpp <= hi:
        raise ValueError(
            f"{tissue_positions_path}: a spot pitch of {pitch_px:.1f} px gives an H&E pixel size of "
            f"{he_mpp:.4f} um/px at {SPOT_PITCH_UM:g} um, outside [{lo}, {hi}]."
        )
    return he_mpp


def assign_cells_to_spots(
    cell_xy: np.ndarray,
    spot_xy: np.ndarray,
    spot_ids: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Give each cell the id of the spot whose disk contains it, or `<NA>` when it is in no disk.

    Visium spots are :data:`SPOT_DIAMETER_UM` wide on a :data:`SPOT_PITCH_UM` pitch, so they never overlap and a
    cell sitting in a gap contributed to no spot. Such cells are left unassigned rather than snapped to the
    nearest spot, so that pseudo-bulking a spot's cells reproduces what that spot actually measured.

    Parameters
    ----------
    cell_xy : numpy.ndarray
        ``(n_cells, 2)`` cell centroids, `(x, y)`.
    spot_xy : numpy.ndarray
        ``(n_spots, 2)`` spot centres in the same units and frame as `cell_xy`.
    spot_ids : numpy.ndarray
        ``(n_spots,)`` the id each spot is known by (the `spots` shapes index).
    radius : float
        Disk radius in the units of `cell_xy`, i.e. :data:`SPOT_DIAMETER_UM` / 2 divided by the pixel size when
        the coordinates are in pixels.

    Returns
    -------
    pandas.arrays.IntegerArray
        ``(n_cells,)`` nullable `Int64`, the containing spot's id or `<NA>`.
    """
    from scipy.spatial import cKDTree

    # One nearest-neighbour query bounded by `radius`; a cell with no spot in range comes back as index n_spots.
    # Disks do not overlap, so the nearest spot is the only candidate and k=1 is exact.
    _, nearest = cKDTree(spot_xy).query(cell_xy, k=1, distance_upper_bound=radius)
    in_disk = nearest < len(spot_xy)
    assigned = pd.array(np.asarray(spot_ids)[np.minimum(nearest, len(spot_xy) - 1)], dtype="Int64")
    assigned[~in_disk] = pd.NA
    return assigned


def visium_to_spatialdata(
    dataset_path: str | Path,
    output_path: str | Path,
    output_name: str | None = None,
    *,
    nucleus_geojson: str | Path | None = None,
    template_adata_path: str | Path | None = DEFAULT_TEMPLATE_ADATA,
    overwrite: bool = False,
) -> Path:
    """Convert one extracted Visium / CytAssist bundle to a SpatialData zarr store.

    The store mirrors the Xenium one: a single `global` coordinate system, here in H&E pixels (`source_he_mpp` um/px), with
    every element, `he_image` included, on an Identity transform. The differences are in the tables. The ground-truth spot
    expression lives in `spots_table`, annotating the `spots` circles, rather than in per-size pseudo-spot
    tables. When `nucleus_geojson` is given, `nucleus_boundaries` and an all-zero `table` over the template's
    gene panel are added too, with `table.obs["spot_id"]` naming the spot each nucleus lies in.

    Parameters
    ----------
    dataset_path : str | Path
        The bundle directory, with `spatial/` already extracted (what `download_visium_study` leaves behind).
    output_path : str | Path
        Directory the zarr is written into.
    output_name : str | None, optional
        Name of the zarr, without `.zarr`. Defaults to the bundle directory's name.
    nucleus_geojson : str | Path | None, optional
        InstanSeg `cells.geojson` for the H&E, in H&E pixels. Segment with
        `pixel_size=visium_he_mpp(...)`. Default `None`
        writes the store without `nucleus_boundaries` and `table`.
    template_adata_path : str | Path | None, optional
        `.h5ad` whose `var` becomes the gene panel of the nucleus `table`. Default is the multi-tissue panel,
        :data:`DEFAULT_TEMPLATE_ADATA`. Unused without `nucleus_geojson`.
    overwrite : bool, optional
        Replace an existing store. Without it an existing store is returned untouched. Default `False`.

    Returns
    -------
    Path
        Path of the zarr store.

    Raises
    ------
    FileNotFoundError
        If the counts matrix, scale factors or tissue positions are missing, or there is no tissue image.
    ValueError
        If an in-tissue spot lies outside the tissue image, the tissue image has no pyramid level near
        :data:`TISSUE_THUMBNAIL_WIDTH` px wide, the spot table is all zeros, no nucleus falls inside any spot,
        or `nucleus_geojson` is given without a `template_adata_path`.

    Notes
    -----
    A protein-capture bundle keeps its antibodies in `spots_table.obsm["protein_expression"]`, not in `var`;
    see :func:`spatialrefinery.core.utils.split_feature_types`.

    Tissue contours come from :func:`spatialrefinery.core.utils.segment_tissue`, which reads only the pyramid
    level nearest :data:`TISSUE_THUMBNAIL_WIDTH` px wide. A raw 10x `.tif`/`.btf` is single-level, so it is
    rejected up front (see :func:`_require_segmentable_pyramid`): convert it with `convert_to_ometiff` first and
    the resulting `*_tissue_image.ome.tif` is picked up automatically.
    """
    import anndata as ad
    import spatialdata as sd
    from spatialdata.models import ShapesModel, TableModel
    from spatialdata.transformations import Identity
    from spatialdata_io import visium

    from spatialrefinery.segmentation.to_spatialdata import read_nucleus_geojson, wsi_image_element

    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_name = output_name or dataset_path.name
    zarr_path = output_path / f"{output_name}.zarr"

    if zarr_path.exists() and not overwrite:
        logger.info("Store already exists, leaving it alone: %s", zarr_path)
        return zarr_path

    files = find_visium_files(dataset_path)
    for key in ("counts_h5", "scalefactors", "tissue_positions", "tissue_image"):
        if files[key] is None:
            raise FileNotFoundError(
                f"{dataset_path}: no {key} found. Run VisiumDownloader.verify_bundle on it; a missing "
                "`spatial/` means `*_spatial.tar.gz` has not been extracted."
            )
    counts_h5, tissue_positions, he_path = files["counts_h5"], files["tissue_positions"], files["tissue_image"]
    assert counts_h5 and tissue_positions and he_path  # the loop above raised if any were missing; narrows for mypy
    he_mpp = visium_he_mpp(tissue_positions)
    logger.info("H&E %s at %.4f um/px (from the spot lattice pitch)", he_path.name, he_mpp)
    _require_segmentable_pyramid(he_path, he_mpp)

    # Only the table is taken from the reader. Its three per-dataset coordinate systems, its hires/lowres
    # images and its circle shapes are all dropped, and everything is rebuilt below in `global` H&E pixels.
    # `gex_only=False` (forwarded to scanpy) keeps the `Antibody Capture` rows of a protein-capture bundle.
    reader_sdata = visium(
        dataset_path,
        dataset_id=counts_h5.name.removesuffix(_COUNTS_SUFFIX),
        fullres_image_file=None,
        gex_only=False,
    )
    spots_table = fix_table_validation_errors(reader_sdata["table"])
    spots_table = _restore_antibody_names(spots_table, counts_h5)
    spots_table = split_feature_types(spots_table, protein_type="Antibody Capture", metric="UMI_COUNTS")
    # Space Ranger's filtered matrix holds only in-tissue spots, but the store must not depend on that: only spots
    # with in_tissue == 1 are kept, so `spots` and `spots_table` are the same set in every sample.
    in_tissue = (spots_table.obs["in_tissue"] == 1).to_numpy()
    if not in_tissue.all():
        logger.warning("Dropping %d of %d spots with in_tissue != 1", int((~in_tissue).sum()), len(in_tissue))
        spots_table = spots_table[in_tissue].copy()
    if cast("Any", spots_table.X).sum() == 0:
        raise ValueError(f"{dataset_path}: the spot table is all zeros; refusing to write it.")

    spot_ids = spots_table.obs["spot_id"].to_numpy()
    spot_xy = np.asarray(spots_table.obsm["spatial"], dtype=float)  # fullres px, already the H&E's pixel grid
    spot_radius = SPOT_DIAMETER_UM / 2 / he_mpp
    spots_table.obs["region"] = "spots"
    spots_table.obs["instance_id"] = spot_ids
    # The reader parsed this table against its own region, so its metadata has to be replaced, not added to.
    spots_table = TableModel.parse(
        spots_table, region="spots", region_key="region", instance_key="instance_id", overwrite_metadata=True
    )

    spots = ShapesModel.parse(
        spot_xy,
        geometry=0,
        radius=spot_radius,
        index=pd.Series(spot_ids),
        transformations={"global": Identity()},
    )

    # The nucleus side is built before the slow image and tissue-segmentation work, so a GeoJSON that does not
    # fit this slide fails in seconds rather than after a whole-slide read.
    nucleus_shapes: dict = {}
    nucleus_tables: dict = {}
    if nucleus_geojson is not None:
        if template_adata_path is None:
            raise ValueError("`template_adata_path` is required with `nucleus_geojson`.")
        nuclei = read_nucleus_geojson(nucleus_geojson)
        centroids = nuclei.geometry.centroid
        cell_xy = np.column_stack([centroids.x, centroids.y])

        template = ad.read_h5ad(template_adata_path, backed="r")
        var = cast("pd.DataFrame", template.var).copy()
        template.file.close()

        table = ad.AnnData(
            X=np.zeros((len(nuclei), len(var)), dtype=np.float32),
            obs=pd.DataFrame(index=nuclei.index),
            var=var,
        )
        table.obsm["spatial"] = cell_xy
        table.obs["spot_id"] = assign_cells_to_spots(cell_xy, spot_xy, spot_ids, spot_radius)
        n_in_spots = int(table.obs["spot_id"].notna().sum())
        if n_in_spots == 0:
            raise ValueError(
                f"No nucleus of {nucleus_geojson} falls inside any spot. The GeoJSON is likely on a different "
                f"pixel grid than {he_path.name} (segment it with pixel_size={he_mpp:.4f})."
            )
        logger.info(
            "%d of %d nuclei lie inside a spot (%.1f%%)", n_in_spots, len(nuclei), 100 * n_in_spots / len(nuclei)
        )

        table = fix_table_validation_errors(table)
        table.obs["region"] = "nucleus_boundaries"
        table.obs["instance_id"] = table.obs.index.astype(str)
        nucleus_shapes["nucleus_boundaries"] = ShapesModel.parse(
            nuclei[[nuclei.geometry.name]], transformations={"global": Identity()}
        )
        nucleus_tables["table"] = TableModel.parse(
            table, region="nucleus_boundaries", region_key="region", instance_key="instance_id"
        )

    # `global` is the H&E's own pixel grid, so the image keeps the Identity it is parsed with, and the contours
    # (segmented at pixel_size=1.0) and every spot and nucleus above are in it without any transform.
    he_image = wsi_image_element(he_path)

    # Nearly every spot the matrix keeps is under tissue and so must be inside the image. A handful beyond the
    # edge is normal and kept; a large share outside means the positions and this image are on different pixel
    # grids, and the whole store would be misregistered without any error.
    sizes = he_image["scale0"]["image"].sizes
    extent_px = np.array([sizes["x"], sizes["y"]])
    outside = ((spot_xy < -spot_radius) | (spot_xy > extent_px + spot_radius)).any(axis=1)
    if outside.mean() > MAX_SPOTS_OUTSIDE_IMAGE:
        raise ValueError(
            f"{outside.sum()} of {len(outside)} in-tissue spots ({100 * outside.mean():.1f}%) lie outside "
            f"{he_path.name} ({sizes['x']} x {sizes['y']} px; spots span x {spot_xy[:, 0].min():.0f}-"
            f"{spot_xy[:, 0].max():.0f}, y {spot_xy[:, 1].min():.0f}-{spot_xy[:, 1].max():.0f} px): "
            "the spot positions are not on this image's pixel grid."
        )
    if outside.any():
        logger.warning(
            "%d of %d in-tissue spots lie more than one spot radius outside %s and are kept (x %.0f-%.0f, y %.0f-%.0f px)",
            outside.sum(),
            len(outside),
            he_path.name,
            spot_xy[outside, 0].min(),
            spot_xy[outside, 0].max(),
            spot_xy[outside, 1].min(),
            spot_xy[outside, 1].max(),
        )
    tissue_contours = ShapesModel.parse(
        segment_tissue(wsi_path=str(he_path), pixel_size=1.0, method="otsu", thumbnail_width=TISSUE_THUMBNAIL_WIDTH),
        transformations={"global": Identity()},
    )

    sdata = sd.SpatialData(
        images={"he_image": he_image},
        shapes={"spots": spots, "tissue_contours": tissue_contours, **nucleus_shapes},
        tables={"spots_table": spots_table, **nucleus_tables},
        attrs={
            **sample_attrs("visium", he_path, source_mpp=he_mpp, he_mpp=he_mpp),
            "spot_diameter_um": SPOT_DIAMETER_UM,
        },
    )

    # Every element above was given only a `global` transformation; a stray reader coordinate system would
    # break the "one comparable frame per sample" guarantee, so fail before writing rather than after.
    if sdata.coordinate_systems != ["global"]:
        raise ValueError(f"Expected only the 'global' coordinate system, found {sdata.coordinate_systems}.")

    output_path.mkdir(parents=True, exist_ok=True)
    if zarr_path.exists():
        shutil.rmtree(zarr_path)
    logger.info("Writing %s...", zarr_path)
    sdata.write(zarr_path)
    return zarr_path


def visium_to_spatialdata_zip(
    dataset_path: str | Path,
    output_path: str | Path,
    output_name: str | None = None,
    *,
    nucleus_geojson: str | Path | None = None,
    template_adata_path: str | Path | None = DEFAULT_TEMPLATE_ADATA,
    overwrite: bool = False,
    keep_zarr: bool = True,
) -> Path:
    """Run :func:`visium_to_spatialdata` and pack the store into an uncompressed `.zarr.zip`.

    Parameters are those of :func:`visium_to_spatialdata`, plus `keep_zarr`: whether to keep the unzipped
    directory afterwards (default `True`). The zip is stored, not deflated, as for the Xenium stores: zarr chunks
    and parquet are already compressed.

    Returns
    -------
    Path
        Path of the zip. An existing zip is returned untouched unless `overwrite` is set.
    """
    zarr_path = visium_to_spatialdata(
        dataset_path,
        output_path,
        output_name,
        nucleus_geojson=nucleus_geojson,
        template_adata_path=template_adata_path,
        overwrite=overwrite,
    )

    zip_path = Path(f"{zarr_path}.zip")
    if zip_path.exists() and not overwrite:
        logger.warning("Zip file already exists: %s", zip_path)
        return zip_path

    logger.info("Creating uncompressed zip archive...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zipf:
        for root, _, names in os.walk(zarr_path):
            for name in names:
                file_path = Path(root) / name
                zipf.write(file_path, file_path.relative_to(zarr_path.parent))

    if not keep_zarr:
        shutil.rmtree(zarr_path)
    return zip_path


class VisiumConverter(SpatialDataConverter):
    """`SpatialDataConverter` adapter delegating to `visium_to_spatialdata[_zip]`.

    The module-level functions stay the canonical API; this class is the registry-facing surface for generic
    tooling that dispatches by technology.
    """

    name = "visium"
    technology = "visium"
    input_suffixes = ()  # a Visium bundle is a directory; see supports()

    def __init__(
        self,
        *,
        nucleus_geojson: str | Path | None = None,
        template_adata_path: str | Path | None = DEFAULT_TEMPLATE_ADATA,
    ) -> None:
        self.nucleus_geojson = nucleus_geojson
        self.template_adata_path = template_adata_path

    @classmethod
    def supports(cls, path: str | Path) -> bool:
        """Whether `path` looks like an extracted Visium bundle directory."""
        path = Path(path)
        if not path.is_dir():
            return False
        files = find_visium_files(path)
        return files["counts_h5"] is not None and files["scalefactors"] is not None

    def convert(
        self,
        source: str | Path,
        output_dir: str | Path,
        *,
        output_name: str | None = None,
        overwrite: bool = False,
        zip_output: bool = False,
        **kwargs: Any,
    ) -> list[Path]:
        """Convert a Visium bundle to a SpatialData zarr (optionally zipped)."""
        fn = visium_to_spatialdata_zip if zip_output else visium_to_spatialdata
        return [
            fn(
                source,
                output_dir,
                output_name,
                nucleus_geojson=self.nucleus_geojson,
                template_adata_path=self.template_adata_path,
                overwrite=overwrite,
                **kwargs,
            )
        ]


__all__ = [
    "VisiumConverter",
    "VisiumDownloader",
    "assign_cells_to_spots",
    "classify_visium_asset",
    "download_visium_study",
    "find_visium_files",
    "visium_he_mpp",
    "visium_to_spatialdata",
    "visium_to_spatialdata_zip",
]

# Register "visium" as a known technology. overwrite=True because this
# module may be re-imported under pytest's --import-mode=importlib.
register_technology(
    TechnologySpec(
        name="visium",
        downloader=VisiumDownloader,
        converter=VisiumConverter,
        file_finder=find_visium_files,
        aliases=("10x_visium", "cytassist", "visium_cytassist", "visium_v2"),
        description="10x Genomics Visium / CytAssist spatial gene expression",
    ),
    overwrite=True,
)
