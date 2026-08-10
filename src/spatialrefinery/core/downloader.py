"""Downloaders: fetch the raw asset bundle for one spatial-omics technology.

`download_with_retries` is the reusable engine: it retries transient
network failures and writes atomically (to `<dest>.part`, then
`os.replace`s onto `dest`) so a killed download never leaves a truncated
file that a later run mistakes for complete. `BaseDownloader` wraps that
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
from urllib.request import Request, urlopen

import spatialrefinery
from spatialrefinery.core.utils import ensure_dir, safe_extract_zip, split_study_filename

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = f"spatialrefinery/{spatialrefinery.__version__} (+python-stdlib)"
DEFAULT_CHUNK_SIZE = 1024 * 1024

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
        """Build a `RemoteAsset` from a URL, deriving `study`/`filename` from its path."""
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
        """
        return self.status in ("downloaded", "cached")


def download_with_retries(
    url: str,
    dest: str | Path,
    *,
    retries: int = 3,
    backoff: float = 2.0,
    timeout: int = 180,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overwrite: bool = False,
    user_agent: str = DEFAULT_USER_AGENT,
) -> Path:
    """Stream `url` to `dest`, retrying transient failures, atomically.

    Writes to `dest` + `.part` and `os.replace`s onto `dest` only on
    success, so an interrupted run never leaves a truncated file that a
    later run's `dest.exists()` check would mistake for complete.

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

    Returns
    -------
    Path
        `dest`, once fully and successfully written.
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
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout, context=ctx) as resp, open(part_path, "wb") as out:
                while chunk := resp.read(chunk_size):
                    out.write(chunk)
            os.replace(part_path, dest)
            return dest
        except OSError as e:
            # HTTPError, URLError, TimeoutError, and ssl.SSLError are all OSError
            # subclasses; catching OSError directly also cleans up `.part` for a
            # plain local write failure (e.g. disk full), which the previous
            # narrower tuple missed.
            last_err = e
            part_path.unlink(missing_ok=True)
            if attempt < retries:
                sleep_for = backoff ** (attempt - 1)
                logger.warning(
                    "Retry %d/%d failed for %s: %s. Retrying in %.1fs...", attempt, retries, url, e, sleep_for
                )
                time.sleep(sleep_for)

    raise RuntimeError(f"Failed to download after {retries} attempts: {url}\nLast error: {last_err}")


class BaseDownloader(ABC):
    """Fetch the asset bundle for one spatial-omics technology.

    Subclasses implement `iter_assets`; everything else (threaded
    fetching, extraction, filtering) has a working default.
    """

    technology: ClassVar[str] = "base"
    #: asset kinds that must never be auto-extracted even if they are .zip
    never_extract: ClassVar[frozenset[str]] = frozenset()

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
        dry_run: bool = False,
    ) -> None:
        self.outdir = Path(outdir)
        self.max_workers = max_workers
        self.retries = retries
        self.backoff = backoff
        self.timeout = timeout
        self.extract = extract
        self.overwrite = overwrite
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
        """Whether `path` should be unzipped in place after downloading."""
        return self.extract and path.suffix.lower() == ".zip" and asset.kind not in self.never_extract

    def post_process(self, study: str, results: Sequence[DownloadResult]) -> None:  # noqa: B027
        """Hook called once per study after all its assets settle. Default: no-op."""

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
            )
        except Exception as e:  # noqa: BLE001 - one asset's failure must not abort the whole batch
            return DownloadResult(asset=asset, path=None, status="failed", error=str(e))

        extracted_to = None
        if self.should_extract(asset, dest):
            try:
                extracted_to = safe_extract_zip(dest, dest.parent)
            except Exception as e:  # noqa: BLE001 - a bad archive shouldn't fail an otherwise-successful download
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
