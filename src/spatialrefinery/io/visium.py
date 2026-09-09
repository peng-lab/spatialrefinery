"""Download 10x Genomics Visium / CytAssist studies and verify the bundles on disk.

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
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import ClassVar

from spatialrefinery.core.downloader import BaseDownloader, DownloadResult, RemoteAsset
from spatialrefinery.core.registry import TechnologySpec, register_technology
from spatialrefinery.core.utils import parse_curl_manifest, safe_extract_tar, tar_root_dir

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
        # `.btf` first: some bundles ship the microscope image as OME-BigTIFF,
        # and it must not be shadowed by a `.tif` of the same sample.
        "tissue_image": first("*_tissue_image.btf", "*_tissue_image.tif", "*_tissue_image.tiff"),
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


__all__ = [
    "VisiumDownloader",
    "classify_visium_asset",
    "download_visium_study",
    "find_visium_files",
]

# Register "visium" as a known technology. overwrite=True because this
# module may be re-imported under pytest's --import-mode=importlib.
# `converter` is left unset: the SpatialData conversion half is not built
# yet, and a downloader-only technology is still useful to dispatch to.
register_technology(
    TechnologySpec(
        name="visium",
        downloader=VisiumDownloader,
        file_finder=find_visium_files,
        aliases=("10x_visium", "cytassist", "visium_cytassist", "visium_v2"),
        description="10x Genomics Visium / CytAssist spatial gene expression",
    ),
    overwrite=True,
)
