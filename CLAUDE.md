# spatialrefinery

This file deliberately describes only *what the repo is and why it exists*. Architecture, public API, code and
testing conventions, commands, and extension points are **not** written here because they change with every PR and
would go stale; get them from the knowledge graph (see "graphify" below), from `README.md`, `docs/`, `pyproject.toml`,
`CHANGELOG.md`, and from reading the surrounding code before editing it.

## What this repo is

`spatialrefinery` (`peng-lab/spatialrefinery`, BSD-3-Clause, Python >= 3.12, docs at
<https://spatialrefinery.readthedocs.io>) turns raw spatial-omics vendor output into analysis-ready data. Today it
covers three jobs:

- **Spatial transcriptomics bundles -> one [SpatialData](https://spatialdata.scverse.org/en/stable/) zarr store per
  sample.** 10x Genomics Xenium is the fully supported technology: transcripts, cell/nucleus boundaries, the aligned
  H&E image, tissue contours, and Visium-like **pseudo-spots** binned at several sizes (e.g. 55um, 100um). Visium /
  CytAssist has a downloader and bundle verification; conversion follows the same pattern as Xenium.
- **Whole-slide / microscopy images** (`.svs`, `.ndpi`, `.tif`, `.mrxs`, `.czi`, ...) **-> tiled, pyramidal OME-TIFF**,
  for fast viewing and as the image half of the stores above.
- **H&E slides -> InstanSeg nucleus boundaries**, packaged as their own SpatialData store with an empty table over a
  template gene panel. This is the optional `segmentation` extra because it pulls torch.

There are also a study **downloader** (fetch a vendor study's raw assets from a `curl -O <url>` manifest, verify them
on disk) and a **registry** so that new technologies and image formats can be added without touching existing code.

## Stack

Python 3.12+, PyTorch 2.x, timm, einops, hydra-core, submitit, wandb, numpy, scipy, scikit-learn, anndata/scanpy,
spatialdata. einops is allowed but every pattern must be explained. Heavy dependencies (torch, openslide, cv2,
geopandas, bioio, ...) are optional extras and must stay lazily imported.

## Why it exists

It is the training-data pipeline for **Phoenix**, pan-cancer virtual spatial transcriptomics from routine histology
(Tran, Gindra et al. 2026, doi 10.64898/2026.04.25.720812). Phoenix learns to predict transcript-derived pseudo-bulk
expression at several resolutions from an H&E image, so it needs *hundreds* of samples assembled identically: aligned
H&E plus pseudo-spots at each resolution, in one SpatialData object per sample, so a batch run yields one coherent
multi-resolution dataset. The nucleus-segmentation path exists so histology *without* a paired spatial run can be put
into the same shape Phoenix predicts into.

## Design principles that follow from that purpose

These are goals, not code locations; they explain why the code looks the way it does and should survive refactors.

- **Uniformity and reproducibility beat convenience.** Every sample must be assembled the same way. When a design
  choice trades per-sample convenience for cross-sample comparability, comparability wins.
- **Samples stay comparable even when the assay is not.** A Xenium protein sub-panel keeps its antibody measurements
  beside the table rather than in the gene panel, so the table's genes are that sample's panel whether or not
  antibodies were run, and downstream code never branches on it.
- **Memory-bounded at Phoenix scale.** Inputs reach ~18k gene targets and ~1e9 transcripts per sample on nodes with
  ~500 GB. Transcript-to-spot aggregation streams and keeps only non-zero cells; whole-slide pyramids are written
  out-of-core. Do not reintroduce dense `n_spots x n_genes` products or whole-slide-in-RAM paths.
- **A silent bad sample is worse than a failed run.** Suspicious outputs (e.g. an all-zero pseudo-spot table, a
  mis-parsed alignment) should raise, not be written.
- **Our own outputs are never re-ingested.** OME-TIFFs written by this package are output only, so re-running a
  conversion over a directory cannot consume its own results.
- **`import spatialrefinery` stays cheap and optional extras stay optional.** Heavy dependencies (torch, openslide,
  cv2, geopandas, bioio, ...) are reached lazily; a missing extra degrades with an install hint, never a top-level
  import error.
- **Scripts are thin.** Logic lives in the package; `scripts/` are argparse wrappers over it. `experimental/`
  and `graphify-out/` are per-checkout scratch and gitignored (`experimental/` is where new technologies are prototyped
  before landing in the package).

## Commit and PR style

Read `git log -3` before writing a message and match it. Subject: sentence-case imperative summary, no `type:` prefix,
no scope. Body: concise, what changed and why, no headed sections. A PR message should describe the combined commits.
Every PR updates `## [Unreleased]` in `CHANGELOG.md` (Keep a Changelog / SemVer). Commit or push only when asked.
Never add Claude as a co-author or credit it anywhere in a commit or PR.

## graphify

This project has a knowledge graph at `graphify-out/` (god nodes, community structure, cross-file relationships). It
is the source of truth for architecture, API surface, conventions and extension points. It is gitignored, so it is
per-checkout and may be absent for teammates -- fall back to normal source navigation when `graphify-out/graph.json`
does not exist.

- For codebase questions, first run `graphify query "<question>"`. Use `graphify path "<A>" "<B>"` for relationships and
  `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than
  `GRAPH_REPORT.md` or raw grep output.
- If `graphify-out/wiki/index.md` exists, use it for broad navigation instead of raw source browsing.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review, or when query/path/explain do not surface
  enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
