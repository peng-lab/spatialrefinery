"""Downloaders: fetch the raw asset bundle for one spatial-omics technology.

`download_with_retries` is the reusable engine: it retries transient
network failures and writes atomically (to `<dest>.part`, then
`os.replace`s onto `dest`) so a killed download never leaves a truncated
file that a later run mistakes for complete. A surviving `<dest>.part` is
resumed via an HTTP `Range` request rather than refetched from byte 0,
which matters for the multi-GB assets this package targets, and every
download whose length the server declares is size-checked before it is
promoted onto `dest`. `BaseDownloader` wraps that
engine with a thread pool and turns per-asset results into a plan/run
workflow that subclasses only need to feed with `RemoteAsset`s (see
`spatialrefinery.io.xenium.XeniumDownloader`).
"""

from __future__ import annotations

import logging
import os
import ssl
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import spatialrefinery
from spatialrefinery.core.utils import (
    collapse_url_slashes,
    ensure_dir,
    safe_extract_tar,
    safe_extract_zip,
    split_study_filename,
)

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = f"spatialrefinery/{spatialrefinery.__version__} (+python-stdlib)"
DEFAULT_CHUNK_SIZE = 1024 * 1024

#: Archive suffixes `BaseDownloader.should_extract` recognises. Matched
#: against the whole lowered filename rather than `Path.suffix`, which is
#: single-part: for `a_spatial.tar.gz` it returns `.gz`, which is why
#: tarballs fell straight through the earlier `suffix == ".zip"` check and
#: were never unpacked. A bare `.gz` is deliberately absent -- a gzip
#: stream carries no member list, so "extract in place" has no defined
#: output path; decompressing `x.csv.gz` is a different operation, and
#: `tarfile.open(..., "r:*")` raises on one anyway.
_ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".tar.bz2", ".tar.xz", ".zip")

Status = Literal["downloaded", "cached", "skipped", "failed"]


@dataclass(frozen=True, slots=True)
class RemoteAsset:
    """A single downloadable file belonging to a study."""

    url: str
    study: str
    filename: str
    kind: str = "unknown"
    size_bytes: int | None = None

    @classmethod
    def from_url(cls, url: str, *, kind: str = "unknown") -> RemoteAsset:
        """Build a `RemoteAsset` from a URL, deriving `study`/`filename` from its path.

        The URL is slash-normalised first, so a manifest typo cannot produce
        an asset that resolves correctly but 403s on download -- see
        :func:`spatialrefinery.core.utils.collapse_url_slashes`.
        """
        url = collapse_url_slashes(url)
        study, filename = split_study_filename(url)
        return cls(url=url, study=study, filename=filename, kind=kind)


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Outcome of fetching one `RemoteAsset`."""

    asset: RemoteAsset
    path: Path | None
    status: Status
    error: str | None = None
    extracted_to: Path | None = None

    @property
    def ok(self) -> bool:
        """Whether this result represents something usable on disk.

        Excludes `"skipped"`: that status is only produced by `dry_run`,
        which never touches the network or checks whether `path` actually
        exists -- reporting it as `ok` would make every dry-run asset look
        usable regardless of whether anything is really on disk.

        Note that `"failed"` does not imply `path is None`: an asset whose
        bytes downloaded cleanly but whose archive would not unpack fails
        with `path` set, so a caller can still find the broken file (see
        `BaseDownloader.require_extract`).
        """
        return self.status in ("downloaded", "cached")


@dataclass(frozen=True, slots=True)
class BundleCheck:
    """What one study's bundle does and does not have on disk after a download.

    Reports two independent vocabularies, because for most technologies they
    differ and their failures have different causes:

    - **kinds** are the semantic labels a downloader's `classify` assigns to
      the *files it fetched*. A missing required kind means an asset was never
      downloaded, or the upstream manifest never listed it.
    - **members** are literal paths expected *inside* the bundle. A missing
      required member means an archive did not unpack, or unpacked partially
      -- which no amount of inspecting the download results can reveal, since
      the bytes arrived correctly.

    For a technology whose downloaded assets *are* its bundle members (10x
    Visium, say) the two nearly coincide. For one that ships its payload
    inside an archive (10x Xenium, whose members arrive unprefixed inside
    `*_outs.zip` and so all classify as `"unknown"`) only the member check
    says anything useful.
    """

    study: str
    present: frozenset[str]
    missing_required: tuple[str, ...]
    missing_members: tuple[str, ...]
    missing_expected: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Whether the bundle carries everything a conversion needs.

        Only the *required* sets count. Expected-but-absent items are
        reported so a caller can see them, never so a usable bundle is
        marked broken: vendors genuinely omit assets for some studies.
        """
        return not self.missing_required and not self.missing_members


def _partial_size(part_path: Path) -> int:
    """Bytes already fetched into `part_path`, or 0 if it is absent/unreadable."""
    try:
        return part_path.stat().st_size
    except OSError:
        return 0


def _declared_total(resp: object, headers: object) -> int | None:
    """Total size of the *whole* resource, or None if the server didn't say.

    For a `206 Partial Content` reply the body is only the requested tail,
    so `Content-Length` describes the tail, not the resource -- the total
    lives in `Content-Range: bytes <start>-<end>/<total>`.
    """
    get = headers.get  # type: ignore[attr-defined]
    if getattr(resp, "status", None) == 206:
        total = get("Content-Range", "").rpartition("/")[2].strip()
    else:
        total = (get("Content-Length") or "").strip()
    return int(total) if total.isdigit() else None


def _content_length(url: str, headers: dict[str, str], timeout: int, ctx: ssl.SSLContext) -> int | None:
    """Resource size via a HEAD request, or None if the server won't say."""
    try:
        req = Request(url, headers=headers, method="HEAD")
        with urlopen(req, timeout=timeout, context=ctx) as resp:
            length = (resp.headers.get("Content-Length") or "").strip()
        return int(length) if length.isdigit() else None
    except OSError:
        return None


def download_with_retries(
    url: str,
    dest: str | Path,
    *,
    retries: int = 3,
    backoff: float = 2.0,
    timeout: int = 180,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overwrite: bool = False,
    resume: bool = True,
    user_agent: str = DEFAULT_USER_AGENT,
) -> Path:
    """Stream `url` to `dest`, retrying transient failures, atomically.

    Writes to `dest` + `.part` and `os.replace`s onto `dest` only on
    success, so an interrupted run never leaves a truncated file that a
    later run's `dest.exists()` check would mistake for complete.

    With `resume=True` a `.part` left by an earlier attempt -- in this call
    or in a previous process -- is continued with a `Range` request instead
    of being refetched from byte 0. Servers that ignore `Range` answer
    `200` with the whole body, which is detected and restarts the write, so
    a resumed file is never a mix of two responses. Whenever the server
    declares a length, the finished `.part` is size-checked before it is
    promoted onto `dest`.

    Parameters
    ----------
    url : str
        Source URL.
    dest : str | Path
        Destination file path.
    retries : int, optional
        Number of attempts before raising. Default 3.
    backoff : float, optional
        Exponential backoff base (seconds) between retries. Default 2.0.
    timeout : int, optional
        Per-attempt socket timeout in seconds. Default 180.
    overwrite : bool, optional
        If False (default) and `dest` already exists, return immediately
        without making a network request.
    resume : bool, optional
        Continue from a leftover `<dest>.part` using an HTTP `Range`
        request, and keep that `.part` when an attempt fails so the next
        attempt (or run) picks up where it stopped. Default True.

    Returns
    -------
    Path
        `dest`, once fully and successfully written.

    Raises
    ------
    RuntimeError
        If every attempt fails, or the fetched byte count disagrees with
        the size the server declared.
    """
    dest = Path(dest)
    if dest.exists() and not overwrite:
        logger.debug("Already exists, skipping: %s", dest)
        return dest

    ensure_dir(dest.parent)
    part_path = dest.with_name(dest.name + ".part")
    ctx = ssl.create_default_context()
    headers = {"User-Agent": user_agent}
    last_err: Exception | None = None

    for attempt in range(1, retries + 1):
        offset = _partial_size(part_path) if resume else 0
        try:
            req_headers = dict(headers)
            if offset:
                req_headers["Range"] = f"bytes={offset}-"
            req = Request(url, headers=req_headers)
            with urlopen(req, timeout=timeout, context=ctx) as resp:
                # A server free to ignore `Range` answers 200 with the entire
                # body; appending that to what we already have would corrupt
                # the file, so only 206 may append.
                appending = bool(offset) and getattr(resp, "status", None) == 206
                if not appending:
                    offset = 0
                total = _declared_total(resp, resp.headers)
                if appending:
                    logger.info("Resuming %s at %d bytes", dest.name, offset)
                with open(part_path, "ab" if appending else "wb") as out:
                    while chunk := resp.read(chunk_size):
                        out.write(chunk)

            written = _partial_size(part_path)
            if total is not None and written != total:
                # Too long means the `.part` is not a prefix of the resource
                # (a stale or corrupt leftover); resuming can never repair
                # that, so drop it and let the next attempt start clean.
                if written > total:
                    part_path.unlink(missing_ok=True)
                raise OSError(f"Incomplete download: got {written} bytes, server declared {total}")

            os.replace(part_path, dest)
            return dest
        except HTTPError as e:
            last_err = e
            if e.code == 416 and offset:
                # The range is past the end of the resource: either the file
                # finished and only the rename was lost, or the `.part` is
                # stale. Only the exact-size case is safe to promote.
                total = _content_length(url, headers, timeout, ctx)
                if total is not None and _partial_size(part_path) == total:
                    os.replace(part_path, dest)
                    return dest
                part_path.unlink(missing_ok=True)
            _backoff(attempt, retries, backoff, url, e)
        except OSError as e:
            # HTTPError, URLError, TimeoutError, and ssl.SSLError are all OSError
            # subclasses; catching OSError directly also cleans up `.part` for a
            # plain local write failure (e.g. disk full), which the previous
            # narrower tuple missed. With `resume` the bytes already on disk are
            # a valid prefix, so they are kept for the next attempt.
            last_err = e
            if not resume:
                part_path.unlink(missing_ok=True)
            _backoff(attempt, retries, backoff, url, e)

    raise RuntimeError(f"Failed to download after {retries} attempts: {url}\nLast error: {last_err}")


def _backoff(attempt: int, retries: int, backoff: float, url: str, err: Exception) -> None:
    """Sleep between attempts, unless `attempt` was the last one."""
    if attempt < retries:
        sleep_for = backoff ** (attempt - 1)
        logger.warning("Retry %d/%d failed for %s: %s. Retrying in %.1fs...", attempt, retries, url, err, sleep_for)
        time.sleep(sleep_for)


class BaseDownloader(ABC):
    """Fetch the asset bundle for one spatial-omics technology.

    Subclasses implement `iter_assets`; everything else (threaded
    fetching, extraction, filtering) has a working default.
    """

    technology: ClassVar[str] = "base"
    #: asset kinds that must never be auto-extracted even if they are archives
    never_extract: ClassVar[frozenset[str]] = frozenset()
    #: asset kinds whose archive is load-bearing: if extraction fails, the
    #: asset itself fails rather than merely warning. Without this a corrupt
    #: archive reports `status="downloaded"`, `ok=True`, and only resurfaces
    #: stages later as a missing file a long way from its cause. Empty by
    #: default, so existing subclasses keep the warn-only behaviour.
    require_extract: ClassVar[frozenset[str]] = frozenset()

    # ---- bundle verification (see `verify_bundle`) ------------------- #
    #: asset kinds without which the bundle cannot be converted at all
    required_kinds: ClassVar[tuple[str, ...]] = ()
    #: asset kinds worth reporting when absent, but which must not block
    expected_kinds: ClassVar[tuple[str, ...]] = ()
    #: glob patterns, relative to the bundle directory, that must match
    #: something once every archive has been unpacked. Globs rather than
    #: exact names so one pattern covers a vendor's naming variants -- e.g.
    #: `tissue_positions*.csv` matches both the Space Ranger >= 2.0 name and
    #: the 1.x `tissue_positions_list.csv`.
    required_members: ClassVar[tuple[str, ...]] = ()
    #: glob patterns reported when they match nothing, but not blocking
    expected_members: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        outdir: str | Path,
        *,
        max_workers: int = 8,
        retries: int = 3,
        backoff: float = 2.0,
        timeout: int = 180,
        extract: bool = True,
        overwrite: bool = False,
        resume: bool = True,
        dry_run: bool = False,
    ) -> None:
        self.outdir = Path(outdir)
        self.max_workers = max_workers
        self.retries = retries
        self.backoff = backoff
        self.timeout = timeout
        self.extract = extract
        self.overwrite = overwrite
        self.resume = resume
        self.dry_run = dry_run

    # ---- must implement --------------------------------------------- #
    @abstractmethod
    def iter_assets(self, source: str | Path | Iterable[str]) -> Iterator[RemoteAsset]:
        """Yield every asset described by `source` (manifest path, study id, or URLs)."""

    # ---- overridable hooks (sane defaults) ---------------------------- #
    def select(self, asset: RemoteAsset) -> bool:
        """Return False to skip `asset`. Default: accept everything."""
        return True

    def destination(self, asset: RemoteAsset) -> Path:
        """Where `asset` is written under `self.outdir`. Default: `outdir/study/filename`."""
        return self.outdir / asset.study / asset.filename

    def should_extract(self, asset: RemoteAsset, path: Path) -> bool:
        """Whether `path` should be unpacked in place after downloading."""
        return self.extract and path.name.lower().endswith(_ARCHIVE_SUFFIXES) and asset.kind not in self.never_extract

    def extract_archive(self, asset: RemoteAsset, path: Path) -> Path:
        """Unpack `path` beside itself, dispatching on its archive format.

        Override this to place an archive's contents somewhere other than
        `path.parent` -- e.g. to give a *flat* archive a directory of its
        own so it cannot overwrite a sibling (see
        `spatialrefinery.io.visium.VisiumDownloader.extract_archive`).
        """
        if path.name.lower().endswith(".zip"):
            return safe_extract_zip(path, path.parent)
        return safe_extract_tar(path, path.parent)

    @staticmethod
    def classify(filename: str) -> str:
        """Return the semantic asset kind for `filename`.

        Override with the technology's suffix map. The default labels
        everything `"unknown"`, which is honest for a subclass that has not
        defined one -- `verify_bundle` then reports on members only.
        """
        return "unknown"

    @classmethod
    def verify_bundle(cls, bundle_dir: str | Path, *, study: str | None = None) -> BundleCheck:
        """Report what one study's bundle has on disk, against this technology's expectations.

        Reads the filesystem rather than a run's `DownloadResult`s, which is
        what makes it a verification instead of an echo: an archive whose
        bytes arrived intact but which unpacked to nothing is indistinguishable
        from a working bundle in the results, and shows up here as a missing
        required member.

        A classmethod because everything it needs is class-level
        configuration, so an already-downloaded bundle can be checked without
        constructing a downloader or knowing its `outdir`.

        Parameters
        ----------
        bundle_dir : str | Path
            The bundle directory, i.e. `outdir/<study>` after a download.
        study : str, optional
            Name recorded on the result. Defaults to the directory's own name.

        Returns
        -------
        BundleCheck
            The report. Never raises for a missing or empty directory -- a
            study whose every asset failed leaves nothing behind, and that is
            a result to report, not an error.
        """
        bundle_dir = Path(bundle_dir)

        present = {cls.classify(p.name) for p in bundle_dir.glob("*") if p.is_file()}
        present.discard("unknown")

        def absent(patterns: tuple[str, ...]) -> tuple[str, ...]:
            # `next(iter(...), None)` so a directory member counts as present:
            # 10x ships `morphology_focus` as a flat OME-TIFF in older bundles
            # and as a directory of tiles in newer ones.
            return tuple(p for p in patterns if next(iter(bundle_dir.glob(p)), None) is None)

        return BundleCheck(
            study=study if study is not None else bundle_dir.name,
            present=frozenset(present),
            missing_required=tuple(k for k in cls.required_kinds if k not in present),
            missing_members=absent(cls.required_members),
            # Kinds and members are merged here: they are different vocabularies,
            # but for something that is only ever reported the distinction would
            # not change what a reader does about it.
            missing_expected=tuple(k for k in cls.expected_kinds if k not in present) + absent(cls.expected_members),
        )

    def post_process(self, study: str, results: Sequence[DownloadResult]) -> None:
        """Verify `study`'s bundle on disk and log what is missing.

        Logged rather than raised so a batch over many studies completes: an
        incomplete upstream manifest is a fact about one study, not a reason
        to discard the others. Subclasses that add technology-specific checks
        should call `super().post_process(...)` first.
        """
        if self.dry_run:
            # Nothing was written, so there is nothing on disk to inspect.
            return
        if not (self.required_kinds or self.required_members or self.expected_kinds or self.expected_members):
            return

        check = self.verify_bundle(self.outdir / study, study=study)
        logger.info(
            "%s: %d/%d asset(s) usable, kinds present: %s",
            study,
            sum(1 for r in results if r.ok),
            len(results),
            ", ".join(sorted(check.present)) or "<none>",
        )
        if check.missing_required:
            logger.warning(
                "%s: missing required asset kind(s) %s; this bundle cannot be converted as-is",
                study,
                ", ".join(check.missing_required),
            )
        if check.missing_members:
            # An archive that downloaded but did not unpack lands here. Logged
            # at error level because every downstream reader opens these paths.
            logger.error(
                "%s: bundle is incomplete on disk, missing %s -- did an archive fail to unpack?",
                study,
                ", ".join(check.missing_members),
            )
        if check.missing_expected:
            logger.warning("%s: missing expected %s", study, ", ".join(check.missing_expected))

    # ---- concrete engine ----------------------------------------------- #
    def fetch(self, asset: RemoteAsset) -> DownloadResult:
        """Download (and optionally extract) a single asset."""
        dest = self.destination(asset)
        if self.dry_run:
            return DownloadResult(asset=asset, path=dest, status="skipped")

        already_present = dest.exists() and not self.overwrite
        try:
            download_with_retries(
                asset.url,
                dest,
                retries=self.retries,
                backoff=self.backoff,
                timeout=self.timeout,
                overwrite=self.overwrite,
                resume=self.resume,
            )
        except Exception as e:  # noqa: BLE001 - one asset's failure must not abort the whole batch
            return DownloadResult(asset=asset, path=None, status="failed", error=str(e))

        extracted_to = None
        if self.should_extract(asset, dest):
            try:
                extracted_to = self.extract_archive(asset, dest)
            except Exception as e:  # noqa: BLE001 - a bad archive shouldn't fail an otherwise-successful download
                if asset.kind in self.require_extract:
                    # `path` is kept so the caller can inspect (or delete) the
                    # archive that would not unpack.
                    return DownloadResult(asset=asset, path=dest, status="failed", error=f"Extraction failed: {e}")
                logger.warning("Extraction failed for %s: %s", dest, e)

        status: Status = "cached" if already_present else "downloaded"
        return DownloadResult(asset=asset, path=dest, status=status, extracted_to=extracted_to)

    def plan(
        self,
        source: str | Path | Iterable[str],
        *,
        studies: Sequence[str] | None = None,
        kinds: Sequence[str] | None = None,
    ) -> list[RemoteAsset]:
        """Resolve and filter assets without touching the network."""
        assets = [a for a in self.iter_assets(source) if self.select(a)]
        if studies is not None:
            wanted = set(studies)
            assets = [a for a in assets if a.study in wanted]
        if kinds is not None:
            wanted_kinds = set(kinds)
            assets = [a for a in assets if a.kind in wanted_kinds]
        return assets

    def run(
        self,
        source: str | Path | Iterable[str],
        *,
        studies: Sequence[str] | None = None,
        kinds: Sequence[str] | None = None,
    ) -> list[DownloadResult]:
        """Plan, then fetch every selected asset with a thread pool."""
        assets = self.plan(source, studies=studies, kinds=kinds)
        if not assets:
            logger.warning("No assets to download.")
            return []

        logger.info("Fetching %d asset(s) with up to %d parallel workers.", len(assets), self.max_workers)
        results: list[DownloadResult] = []
        by_study: dict[str, list[DownloadResult]] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_asset = {executor.submit(self.fetch, asset): asset for asset in assets}
            for future in as_completed(future_to_asset):
                asset = future_to_asset[future]
                try:
                    result = future.result()
                except Exception as e:  # noqa: BLE001 - one asset's failure must not abort the whole batch
                    result = DownloadResult(asset=asset, path=None, status="failed", error=str(e))
                results.append(result)
                by_study.setdefault(asset.study, []).append(result)

        for study, study_results in sorted(by_study.items()):
            self.post_process(study, study_results)

        failed = [r for r in results if r.status == "failed"]
        if failed:
            logger.warning("%d/%d asset(s) failed to download.", len(failed), len(results))

        return results
