# Nucleus segmentation on H&E whole-slide images

Status: **implemented** on branch `worktree-instanseg-segmentation`, 2026-08-31.

Design notes for the segmentation pipeline: `spatialrefinery.segmentation`,
its CLI wrappers in `scripts/`, and the SLURM launchers here.

## Approach

Segmentation uses [InstanSeg](https://github.com/instanseg/instanseg)
(`instanseg-torch`), whose `brightfield_nuclei` model is the default and is
trained for H&E.

The decisive property for this pipeline is that InstanSeg owns the
**whole-slide layer**: `eval_whole_slide_image` handles tiling, cross-tile
label matching (`match_labels`), an Otsu tissue prefilter, and GeoJSON export.
Writing that layer by hand is the expensive part of whole-slide segmentation —
cross-tile instance deduplication in particular — so a model that ships it is
worth considerably more than one that does not.

Its runtime requirements are also undemanding: `numpy>=1.24` and `torch>=2.0`
with no upper bounds, so it installs additively into this project's existing
environment. Verified: 31 packages added, **zero removed**; `numpy`,
`pydantic`, `zarr` and `spatialdata` all untouched.

```python
model = InstanSeg("brightfield_nuclei", image_reader="tiffslide")
model.eval_whole_slide_image(
    image=str(wsi_path),
    pixel_size=wsi_mpp,          # else read from slide metadata
    tile_size=512, overlap=80,
    use_otsu_threshold=True,     # skip background tiles
    save_geojson=True,
)
```

## Three upstream defects had to be bridged

All three are in `instanseg` 0.1.1 and unchanged on `main`. They are patched at
call time in `segmentation/_compat.py`, never vendored, so deleting that module
is all it will take once upstream fixes them. Each is pinned by a test.

1. **zarr 3.** `eval_whole_slide_image` builds its canvas with
   `zarr.DirectoryStore`, which zarr 3 renamed to `zarr.storage.LocalStore`.
   InstanSeg pins `zarr>=2.0.0,<3` in its `io` extra as a result — and that cap
   collides with `spatialdata` (`zarr>=3.0.0`) and `anndata` (`zarr>=3.1`).
   **Installing `instanseg-torch[io]` silently downgrades zarr, numcodecs and
   tiffslide and breaks spatialdata.** The `segmentation` extra therefore names
   `rasterio` and `geojson` directly. The pin's stated reason ("tiffslide
   doesn't support zarr v3 yet") is stale as of tiffslide 4.0
   (Bayer-Group/tiffslide#97); only the *name* is missing, and `LocalStore` is
   a drop-in for all four operations InstanSeg performs on the store.
2. **`TiffSlide` is never imported.** `read_slide` calls `TiffSlide(...)` at
   `inference_class.py:236`, but every import of that name in the module is
   function-local, so the global is unbound and any whole-slide call raises
   `NameError` before reading a single tile.
3. **The GeoJSON it writes is invalid.** The exporter emits a comma after every
   feature and then closes the array, so output ends `...}},\n]`. `json.load`
   rejects it, and so does QuPath. Repaired in the artefact rather than worked
   around at read time, so the published file is valid for any consumer.

Defects 2 and 3 together mean InstanSeg's whole-slide path cannot have been run
end to end upstream. All three are worth reporting.

## Layout

Logic lives in the package; `scripts/` holds thin wrappers, matching
`scripts/convert_to_ometiff.py`.

| File | Purpose |
|---|---|
| `src/spatialrefinery/segmentation/_compat.py` | the three bridges above |
| `src/spatialrefinery/segmentation/instanseg.py` | `segment_wsi()` — slide → `cells.geojson` |
| `src/spatialrefinery/segmentation/to_spatialdata.py` | `geojson_to_spatialdata()` |
| `scripts/instanseg_segment.py` | CLI; emits the `GEOJSON_PATH=` contract |
| `scripts/geojson_to_spatialdata.py` | CLI |
| `slurm/segment_node_worker.sh` | per-node worker, round-robin across GPUs |
| `slurm/segment_slurm.sh` | manifest + sbatch launcher |
| `tests/test_compat.py`, `tests/test_segmentation.py` | 24 tests |

The two stages exchange a `cells.geojson` rather than being merged: it is a
resume boundary, so a conversion failure does not force re-segmentation.

Implementation details worth knowing:

- **Outputs are redirected.** InstanSeg writes its `.zarr` and `.geojson` next
  to the *input* file; slides usually sit on read-only dataset mounts, so
  `segment_wsi` runs the model against a symlink inside the output directory.
- **The image element is built lazily** from the slide's own pyramid via
  `tifffile`'s zarr interface — a level-0 plane on the test slide is
  33427 × 11949 × 3, about 1.1 GiB if materialised.
- **CRS is cleared before centroids are taken.** `gpd.read_file` tags GeoJSON
  as EPSG:4326, but these are pixel coordinates; left in place, `.centroid` is
  computed against a spherical datum and every centroid drifts.
- **`.gitignore`** ignores `/scripts/*.py` and `/slurm/` as scratch; explicit
  negations track this pipeline.

## Sensitivity on pale slides

On weakly haematoxylin-stained slides InstanSeg misses pale nuclei. The cause
is staining, not the model: the crop measured has **median lightness 205/255**
(1st percentile 119), and the nuclei found were the darker ones. InstanSeg
normalises each tile with `percentile_normalize`, a *global* stretch over the
tile; where bright cytoplasm dominates the histogram it does not lift pale
nuclei above the seed threshold (default 0.7).

Measured on one 1000 × 1000 crop:

| variant | nuclei | vs default |
|---|---|---|
| default | 209 | — |
| `seed_threshold=0.3` | 222 | +6% |
| percentile stretch 1–99 | 219 | +5% |
| **CLAHE clip=2** | **254** | **+22%** |
| **CLAHE clip=2 + `seed_threshold=0.4`** | **266** | **+27%** |
| haematoxylin colour deconvolution | 97 | −54% |

Overlays confirmed the extra detections are real nuclei with tight boundaries.
Two things were ruled out: **resolution is not the bottleneck** (InstanSeg
downsamples 1.83× from 0.2738 to its native 0.5 µm; suppressing that gained
+2%), and **colour deconvolution actively hurts** — the brightfield model wants
true H&E appearance.

Confirmed on the **whole slide** (8m52s end to end, versus 8m40s without —
CLAHE costs nothing measurable):

| | default | `--clahe 2.0 --seed-threshold 0.4` |
|---|---|---|
| nuclei | 67,168 | **83,554 (+24.4%)** |
| median area | 318 px² | 339 px² |
| objects < 20 px² | 54 | 106 (0.13% of total) |

The area distribution shifted **up** at every percentile from 1 to 99, which is
the reassuring direction: a flood of spurious fragments would have pulled the
median down. External sanity check: Xenium's own DAPI-based segmentation of
this sample has 97,560 cells, so the default recovered 69% of that from H&E and
CLAHE 86% — different images, so exact agreement is not expected.

Both controls are **opt-in and off by default**, so runs stay reproducible:
`--clahe 2.0 --seed-threshold 0.4` on the CLI, `CLAHE` and `SEED_THRESHOLD` on
`segment_slurm.sh`. CLAHE is applied to L in LAB (hue preserved) by wrapping
`model._to_tensor`, the single point every tile passes through in the
whole-slide loop — no fork required. Per-tile is the right granularity for an
adaptive method; the 80 px overlap and cross-tile label matching absorb seam
discontinuities.

## Verification

1. Driver check — CUDA 13 wheels were the main rollout risk:
   `nvidia-smi --query-gpu=driver_version --format=csv` → **595.71.05**, fine,
   Turing sm_75 included.
2. Environment coherence after install — `numpy 2.4.6`, `pydantic 2.12.5`,
   `zarr 3.3.0`, `spatialdata 0.8.0`, `anndata 0.13.2` all unchanged.
3. `pytest tests/` → 86 passed locally; 84 passed + 2 skipped in the CI hatch
   environments on both py3.12 and py3.14 (the skips are the tests needing the
   optional `segmentation` extra). `prek run --all-files` clean.
4. End to end on `Xenium_V1_hKidney_nondiseased_section_he_image.ome.tif`
   (867 MB, 33427 × 11949, mpp 0.2738 read from metadata):

   | stage | result |
   |---|---|
   | segmentation | 7m49s, 1239 tiles, 51 MB `cells.geojson`, 83,554 nuclei |
   | conversion | 67s, SpatialData zarr |

   Read back: image `DataTree[cyx] (3, 33427, 11949)` with 5 pyramid levels;
   `nucleus_boundaries` 83,554 Polygons, geometry column only; table
   `(83554, 377)` over the Xenium panel; `obs` = `region`, `instance_id`;
   centroids within slide bounds.

## Risks

- **InstanSeg's WSI support is documented as "limited."** It held on a 400 Mpx
  slide; validate on the largest slide in a cohort before batch rollout.
- **The `[io]` extra must never be installed** — see defect 1.
- **The +24.4% is one slide from one tissue.** If staining varies across a
  cohort, `--clahe 2.0` may want tuning per batch before becoming a default.
- **GPU memory** ran ~35 GB with CLAHE versus ~12 GB without. Fine on a 46 GB
  card, but lower `--tile-size` if fanning out 4 GPUs per node hits OOM.
