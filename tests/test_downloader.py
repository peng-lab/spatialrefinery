"""Tests for `spatialrefinery.core.downloader`'s extraction hooks.

Covers only the pure, filesystem-local surface: which archives
`should_extract` recognises, how `extract_archive` dispatches between the
zip and tar helpers, and whether a failed extraction escalates to a failed
asset. Nothing here touches the network -- `download_with_retries`,
`plan()` and `run()` are exercised by the per-technology suites and by the
end-to-end scripts, since faking HTTP would test the fake.

`fetch` is reachable offline nonetheless: `download_with_retries` returns
immediately when `dest` already exists and `overwrite` is False, so
pre-creating the destination isolates the extraction branch.
"""

from __future__ import annotations

import io
import tarfile
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest

from spatialrefinery.core.downloader import BaseDownloader, RemoteAsset


class _Downloader(BaseDownloader):
    """Minimal concrete subclass; `BaseDownloader` is an ABC over `iter_assets` alone."""

    def iter_assets(self, source: str | Path | Iterable[str]) -> Iterator[RemoteAsset]:
        return iter(())


class _RequiresSpatial(_Downloader):
    """A downloader for which the `spatial` archive is load-bearing."""

    require_extract = frozenset({"spatial"})


def _asset(filename: str, kind: str = "unknown") -> RemoteAsset:
    return RemoteAsset(url=f"https://example.com/study_a/{filename}", study="study_a", filename=filename, kind=kind)


def _write_tar(path: Path, name: str = "nested/file.txt", payload: bytes = b"hello") -> None:
    with tarfile.open(path, "w:gz") as tf:
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("study_a_outs.zip", True),
        ("study_a_spatial.tar.gz", True),  # the case `Path.suffix == ".gz"` used to miss
        ("study_a.tgz", True),
        ("study_a.tar", True),
        ("study_a.tar.bz2", True),
        ("study_a.tar.xz", True),
        # A bare `.gz` is a single compressed file, not a container: it has no
        # member list, so "extract in place" has no defined output path.
        ("study_a_transcripts.csv.gz", False),
        ("study_a_he_image.ome.tif", False),
        ("study_a_cloupe.cloupe", False),
    ],
)
def test_should_extract_recognises_archives(tmp_path, filename: str, expected: bool) -> None:
    """Archive detection must match whole filenames, since `Path.suffix` is single-part."""
    downloader = _Downloader(tmp_path)
    assert downloader.should_extract(_asset(filename), tmp_path / filename) is expected


def test_should_extract_honours_never_extract(tmp_path) -> None:
    """A kind in `never_extract` stays packed even though its suffix is an archive."""

    class _Skips(_Downloader):
        never_extract = frozenset({"xe_outs"})

    asset = _asset("study_a_xe_outs.zip", kind="xe_outs")
    assert _Skips(tmp_path).should_extract(asset, tmp_path / asset.filename) is False


def test_should_extract_honours_extract_flag(tmp_path) -> None:
    """`extract=False` disables unpacking wholesale, regardless of kind or suffix."""
    downloader = _Downloader(tmp_path, extract=False)
    asset = _asset("study_a_spatial.tar.gz", kind="spatial")
    assert downloader.should_extract(asset, tmp_path / asset.filename) is False


def test_extract_archive_dispatches_by_suffix(tmp_path) -> None:
    """One hook unpacks both formats, so a subclass never re-implements the dispatch."""
    downloader = _Downloader(tmp_path)

    zip_path = tmp_path / "a_outs.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("from_zip.txt", "z")
    downloader.extract_archive(_asset(zip_path.name), zip_path)

    tar_path = tmp_path / "a_spatial.tar.gz"
    _write_tar(tar_path, name="from_tar.txt", payload=b"t")
    downloader.extract_archive(_asset(tar_path.name, kind="spatial"), tar_path)

    assert (tmp_path / "from_zip.txt").read_text() == "z"
    assert (tmp_path / "from_tar.txt").read_bytes() == b"t"


def test_require_extract_escalates_a_corrupt_archive_to_failed(tmp_path) -> None:
    """A required archive that will not unpack fails the asset, at the point of failure.

    Without this the result is `status="downloaded"`, `ok=True`, and the
    breakage only surfaces stages later as a missing file. `path` is kept on
    the failure so the caller can still find the archive to delete.
    """
    downloader = _RequiresSpatial(tmp_path)
    asset = _asset("study_a_spatial.tar.gz", kind="spatial")
    dest = downloader.destination(asset)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"not a tarball")

    result = downloader.fetch(asset)
    assert result.status == "failed"
    assert result.ok is False
    assert result.path == dest
    assert result.error is not None and "Extraction failed" in result.error


def test_failed_extraction_of_an_optional_kind_stays_usable(tmp_path) -> None:
    """With `require_extract` empty -- every pre-existing subclass -- a bad archive only warns.

    Pins that generalising the extraction hook did not change the behaviour
    Xenium already relied on.
    """
    downloader = _Downloader(tmp_path)
    asset = _asset("study_a_analysis.tar.gz", kind="analysis")
    dest = downloader.destination(asset)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"not a tarball")

    result = downloader.fetch(asset)
    assert result.status == "cached"
    assert result.ok is True
    assert result.extracted_to is None


def test_verify_bundle_is_inert_without_configuration(tmp_path) -> None:
    """A subclass declaring no expectations reports nothing, rather than everything missing.

    `post_process` gained a body when verification moved into the base class;
    this pins that a downloader which opts out is unaffected.
    """
    bundle = tmp_path / "study_a"
    bundle.mkdir()
    check = _Downloader.verify_bundle(bundle)
    assert check.ok
    assert check.present == frozenset()
    assert check.missing_required == ()
    assert check.missing_members == ()


def test_verify_bundle_matches_members_by_glob(tmp_path) -> None:
    """Members are globs so one pattern covers a vendor's naming variants."""

    class _Globs(_Downloader):
        required_members = ("spatial/tissue_positions*.csv",)

    bundle = tmp_path / "study_a"
    (bundle / "spatial").mkdir(parents=True)
    # The pre-2.0 Space Ranger name; the modern one is `tissue_positions.csv`.
    (bundle / "spatial" / "tissue_positions_list.csv").touch()
    assert _Globs.verify_bundle(bundle).missing_members == ()


def test_verify_bundle_counts_a_directory_member_as_present(tmp_path) -> None:
    """A member can be a directory: 10x ships `morphology_focus` both ways."""

    class _Dirs(_Downloader):
        required_members = ("morphology_focus*",)

    bundle = tmp_path / "study_a"
    (bundle / "morphology_focus").mkdir(parents=True)
    assert _Dirs.verify_bundle(bundle).missing_members == ()
