"""Tests for `spatialrefinery.io.visium`'s download and verification surface.

Covers asset classification (including the two suffix collisions that
longest-suffix-wins exists to resolve), manifest resolution, the
flat-archive redirect that keeps two tarballs from overwriting each other,
and the bundle completeness report.

Deliberately not covered: any real download. `fetch`/`run` are the shared
`BaseDownloader` engine, tested in `test_downloader.py`; here the network
is avoided via `dry_run` and by calling the pure hooks directly. Bundles
are synthesised as empty files with the right names, since classification
and verification read filenames, never contents.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from spatialrefinery.core.registry import get_technology
from spatialrefinery.io.visium import VisiumDownloader, classify_visium_asset, find_visium_files

STUDY = "CytAssist_11mm_FFPE_Human_Kidney"


def _touch(directory: Path, *names: str) -> None:
    """Create empty files named `<STUDY>_<name>` (or `name` verbatim if it has a `/`)."""
    for name in names:
        path = directory / name if "/" in name else directory / f"{STUDY}_{name}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _write_tar(path: Path, names: tuple[str, ...]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name in names:
            info = tarfile.TarInfo(name)
            info.size = 1
            tf.addfile(info, io.BytesIO(b"x"))


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        ("_spatial.tar.gz", "spatial"),
        ("_analysis.tar.gz", "analysis"),
        ("_deconvolution.tar.gz", "deconvolution"),
        ("_filtered_feature_bc_matrix.tar.gz", "filtered_matrix_mtx"),
        ("_raw_feature_bc_matrix.tar.gz", "raw_matrix_mtx"),
        ("_filtered_feature_bc_matrix.h5", "filtered_matrix"),
        ("_raw_feature_bc_matrix.h5", "raw_matrix"),
        ("_raw_probe_bc_matrix.h5", "raw_probe_matrix"),
        ("_molecule_info.h5", "molecule_info"),
        ("_tissue_image.tif", "tissue_image"),
        ("_tissue_image.btf", "tissue_image"),
        ("_image.tif", "cytassist_image"),
        ("_alignment_file.json", "alignment"),
        ("_probe_set.csv", "probe_set"),
        ("_feature_reference.csv", "feature_reference"),
        ("_isotype_normalization_factors.csv", "isotype_normalization"),
        ("_metrics_summary.csv", "metrics_summary"),
        ("_web_summary.html", "web_summary"),
        ("_cloupe.cloupe", "cloupe"),
    ],
)
def test_classify_covers_every_manifest_suffix(suffix: str, expected: str) -> None:
    """Every filename tail 10x ships must classify; an `unknown` means an unhandled asset."""
    assert classify_visium_asset(f"{STUDY}{suffix}") == expected


@pytest.mark.parametrize(
    ("filename", "expected", "shadowed_by"),
    [
        # Both matrix tarballs end in `_feature_bc_matrix.tar.gz`, so a shorter
        # key would collapse them into one kind.
        (f"{STUDY}_raw_feature_bc_matrix.tar.gz", "raw_matrix_mtx", "filtered_matrix_mtx"),
        (f"{STUDY}_filtered_feature_bc_matrix.tar.gz", "filtered_matrix_mtx", "raw_matrix_mtx"),
        # `_tissue_image.tif` also ends in `_image.tif`: the microscope image
        # must not be mistaken for the coarser CytAssist capture.
        (f"{STUDY}_tissue_image.tif", "tissue_image", "cytassist_image"),
    ],
)
def test_classify_prefers_the_longest_matching_suffix(filename: str, expected: str, shadowed_by: str) -> None:
    """Overlapping suffixes resolve to the most specific kind, not the first match."""
    kind = classify_visium_asset(filename)
    assert kind == expected
    assert kind != shadowed_by


def test_classify_unknown_suffix() -> None:
    """An unrecognised asset is labelled, not guessed at, so `kinds=` filtering stays honest."""
    assert classify_visium_asset(f"{STUDY}_something_new.parquet") == "unknown"


def test_iter_assets_derives_study_and_kind(tmp_path) -> None:
    """Manifest lines resolve to study/kind, including 10x's double-slash URLs.

    Some 10x manifests carry `spatial-exp/3.1.3//<study>/...`. The empty
    path segment must not become the study name, *and* it must be collapsed
    out of the URL: the CDN answers the doubled path with 403, so leaving it
    in makes every asset of that study undownloadable.
    """
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        "# Output Files\n"
        "curl -O https://example.com/samples/2.1.0/study_a/study_a_spatial.tar.gz\n"
        "curl -O https://example.com/samples/3.1.3//study_b/study_b_cloupe.cloupe\n"
        "not a curl line\n"
    )

    assets = list(VisiumDownloader(tmp_path).iter_assets(manifest))
    assert [(a.study, a.kind) for a in assets] == [("study_a", "spatial"), ("study_b", "cloupe")]
    assert assets[1].url == "https://example.com/samples/3.1.3/study_b/study_b_cloupe.cloupe"


def test_plan_filters_by_kind(tmp_path) -> None:
    """`kinds=` narrows a manifest without any network access."""
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        "curl -O https://example.com/s/study_a/study_a_spatial.tar.gz\n"
        "curl -O https://example.com/s/study_a/study_a_web_summary.html\n"
    )

    assets = VisiumDownloader(tmp_path, dry_run=True).plan(manifest, kinds=["spatial"])
    assert [a.kind for a in assets] == ["spatial"]


def test_find_visium_files_does_not_shadow_the_cytassist_image(tmp_path) -> None:
    """`*_image.tif` also matches `*_tissue_image.tif`; the two must stay distinguishable.

    Without the exclusion the microscope H&E would be reported as the
    CytAssist capture, and a conversion would silently use the wrong image.
    """
    bundle = tmp_path / STUDY
    bundle.mkdir()
    _touch(bundle, "image.tif", "tissue_image.tif")

    files = find_visium_files(bundle)
    assert files["cytassist_image"] == bundle / f"{STUDY}_image.tif"
    assert files["tissue_image"] == bundle / f"{STUDY}_tissue_image.tif"


def test_find_visium_files_prefers_bigtiff_for_the_microscope_image(tmp_path) -> None:
    """A `.btf` microscope image must not be shadowed by a `.tif` of the same sample."""
    bundle = tmp_path / STUDY
    bundle.mkdir()
    _touch(bundle, "tissue_image.btf", "tissue_image.tif")
    assert find_visium_files(bundle)["tissue_image"] == bundle / f"{STUDY}_tissue_image.btf"


def test_verify_visium_bundle_accepts_a_complete_bundle(tmp_path) -> None:
    """The happy path: nothing required missing, spatial unpacked, cloupe found."""
    bundle = tmp_path / STUDY
    bundle.mkdir()
    _touch(
        bundle,
        "filtered_feature_bc_matrix.h5",
        "tissue_image.btf",
        "image.tif",
        "cloupe.cloupe",
        "spatial.tar.gz",
        "analysis.tar.gz",
        "spatial/scalefactors_json.json",
        "spatial/tissue_positions.csv",
    )

    check = VisiumDownloader.verify_bundle(bundle)
    assert check.study == STUDY
    assert check.ok
    assert "cloupe" in check.present
    assert check.missing_required == ()
    assert check.missing_members == ()
    assert check.missing_expected == ()


def test_verify_visium_bundle_reports_missing_required_kinds(tmp_path) -> None:
    """An upstream manifest can omit the counts matrix entirely; that is reported, not raised.

    `Visium_V2_Human_Lymph_Node` in the reference cohort ships neither a
    counts matrix nor a `.cloupe`, so raising here would abort a whole batch
    on one study's incomplete manifest.
    """
    bundle = tmp_path / STUDY
    bundle.mkdir()
    _touch(bundle, "spatial.tar.gz", "spatial/scalefactors_json.json", "spatial/tissue_positions.csv")

    check = VisiumDownloader.verify_bundle(bundle)
    assert not check.ok
    assert "cloupe" not in check.present
    assert set(check.missing_required) == {"filtered_matrix", "tissue_image"}
    # `cloupe` gets its own dedicated warning in `post_process`, so listing it
    # in `expected_kinds` too would log the same absence twice.
    assert "cloupe" not in check.missing_expected


def test_verify_visium_bundle_detects_an_unextracted_spatial_archive(tmp_path) -> None:
    """A downloaded-but-unpacked `spatial.tar.gz` is not a usable bundle.

    `present` sees the archive, so only a filesystem check inside `spatial/`
    can tell the two apart -- which is why verification reads the disk
    rather than the download results.
    """
    bundle = tmp_path / STUDY
    bundle.mkdir()
    _touch(bundle, "filtered_feature_bc_matrix.h5", "tissue_image.btf", "spatial.tar.gz")

    check = VisiumDownloader.verify_bundle(bundle)
    assert "spatial" in check.present
    assert check.missing_required == ()
    # The archive is present as a *kind*; only the member check inside
    # `spatial/` can tell a downloaded tarball from an extracted one.
    assert check.missing_members == ("spatial/scalefactors_json.json", "spatial/tissue_positions*.csv")
    assert not check.ok


def test_verify_visium_bundle_tolerates_a_missing_directory(tmp_path) -> None:
    """A study whose every asset failed leaves no directory; that must not raise."""
    check = VisiumDownloader.verify_bundle(tmp_path / "never_downloaded")
    assert check.present == frozenset()
    assert not check.ok


def test_extract_archive_keeps_a_self_contained_archive_in_place(tmp_path) -> None:
    """`<study>_spatial.tar.gz` unpacks to `<bundle>/spatial/`, not a nested duplicate.

    That flat-beside-the-assets layout is what `spatialdata_io.visium`
    expects, so getting it right here removes the need for a staging copy.
    """
    bundle = tmp_path / STUDY
    bundle.mkdir()
    archive = bundle / f"{STUDY}_spatial.tar.gz"
    _write_tar(archive, ("spatial/scalefactors_json.json", "spatial/tissue_positions.csv"))

    downloader = VisiumDownloader(tmp_path)
    asset = next(iter(downloader.iter_assets([f"https://example.com/s/{STUDY}/{archive.name}"])))
    downloader.extract_archive(asset, archive)

    assert (bundle / "spatial" / "scalefactors_json.json").exists()
    assert not (bundle / "spatial" / "spatial").exists()


def test_extract_archive_redirects_a_flat_archive(tmp_path) -> None:
    """Two flat archives sharing a member name must not overwrite each other.

    They extract from a thread pool into one directory, so an in-place
    unpack would be both lossy and non-deterministic. Each therefore gets a
    directory named for its kind.
    """
    bundle = tmp_path / STUDY
    bundle.mkdir()
    members = ("matrix.mtx.gz", "barcodes.tsv.gz", "features.tsv.gz")

    downloader = VisiumDownloader(tmp_path)
    both = (("filtered_feature_bc_matrix", "filtered_matrix_mtx"), ("raw_feature_bc_matrix", "raw_matrix_mtx"))
    for suffix, kind in both:
        archive = bundle / f"{STUDY}_{suffix}.tar.gz"
        _write_tar(archive, members)
        asset = next(iter(downloader.iter_assets([f"https://example.com/s/{STUDY}/{archive.name}"])))
        assert asset.kind == kind
        downloader.extract_archive(asset, archive)

    for kind in ("filtered_matrix_mtx", "raw_matrix_mtx"):
        assert (bundle / kind / "matrix.mtx.gz").exists()
    assert not (bundle / "matrix.mtx.gz").exists()


def test_visium_is_registered_as_a_technology() -> None:
    """`io/__init__.py` must re-export the module, or `_ensure_builtins` never imports it."""
    spec = get_technology("visium")
    assert spec.downloader is VisiumDownloader
    assert spec.file_finder is find_visium_files
    # The conversion half is not built yet; a downloader-only technology is valid.
    assert spec.converter is None
    assert get_technology("cytassist").name == "visium"
