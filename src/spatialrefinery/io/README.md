# SpatialRefinery I/O Module

This module provides functions for reading and converting various spatial transcriptomics data formats to SpatialData zarr format.

> For the full, versioned API reference and narrated tutorials, see the
> [documentation site](https://spatialrefinery.readthedocs.io/page/api.html). This README is a quick local reference and
> may lag behind it.

## Xenium Data Conversion

The `xenium` module provides functionality to convert 10x Xenium raw bundled outputs to SpatialData zarr format.

### Quick Start

```python
from spatialrefinery.io import xenium_to_spatialdata

# Convert Xenium data to zarr
zarr_path = xenium_to_spatialdata(
    dataset_path="/path/to/xenium/raw/data",
    output_path="/path/to/output",
    output_name="my_sample",
    create_spots=True,
    spot_sizes=[55, 100],
    n_jobs=8,
)
```

### Main Functions

#### `xenium_to_spatialdata()`

Convert 10x Xenium raw data to SpatialData zarr format.

**Parameters:**
- `dataset_path`: Path to the Xenium dataset directory containing raw files
- `output_path`: Path to the directory where the zarr file will be saved
- `output_name`: Name for the output zarr file (without .zarr extension). If None, uses the dataset directory name
- `include_aligned_image`: Whether to include aligned H&E image if available (default: True)
- `create_spots`: Whether to create pseudo-spots (default: True)
- `spot_sizes`: List of spot sizes in micrometers to create (default: None; no spots are created unless a list is given)
- `overlap`: Fractional overlap between adjacent hexagonal spots (default: 0.0)
- `values`: Which element to aggregate into spots, `"transcripts"` or `"cell_boundaries"` (default: `"transcripts"`)
- `n_jobs`: Number of workers for parallel processing (default: 1)
- `overwrite`: Whether to overwrite existing zarr file (default: False)

**Returns:**
- Path to the created zarr file

**Example:**
```python
from spatialrefinery.io import xenium_to_spatialdata

zarr_path = xenium_to_spatialdata(
    dataset_path="/data/xenium_sample",
    output_path="/output",
    output_name="sample1",
    include_aligned_image=True,
    create_spots=True,
    spot_sizes=[55, 100],
    n_jobs=8,
    overwrite=False,
)
```

#### `xenium_to_spatialdata_zip()`

Convert Xenium data to SpatialData zarr and create a zip archive.

**Parameters:**
- Same as `xenium_to_spatialdata()`, plus:
- `keep_zarr`: Whether to keep the unzipped zarr directory after creating zip (default: True)

**Returns:**
- Path to the created zip file

**Example:**
```python
from spatialrefinery.io import xenium_to_spatialdata_zip

zip_path = xenium_to_spatialdata_zip(
    dataset_path="/data/xenium_sample",
    output_path="/output",
    output_name="sample1",
    keep_zarr=False,  # Remove zarr directory after zipping
    n_jobs=8,
)
```

### Downloading raw data

```python
from spatialrefinery.io.xenium import download_xenium_study

results = download_xenium_study(
    source="example_data/10x_xenium_human.txt",  # a `curl -O <url>` manifest
    outdir="/path/to/raw_files",
    max_workers=8,
)
```

Built on `spatialrefinery.core.downloader.BaseDownloader`: retries transient
failures, writes atomically so an interrupted download never leaves a
truncated file, and unzips `*_outs.zip` assets in place (skipping
`*_xe_outs.zip`).

## Visium Data Download

```python
from spatialrefinery.io.visium import VisiumDownloader, download_visium_study

results = download_visium_study(
    source="example_data/10x_visium_human.txt",  # a `curl -O <url>` manifest
    outdir="/path/to/raw_files",
    kinds=["spatial", "filtered_matrix", "tissue_image", "cloupe"],  # optional
    max_workers=8,
)
```

Same engine as above, with two differences that matter. Visium ships
`.tar.gz` rather than `.zip`, and those archives are unpacked **in place**,
so a bundle ends up flat with `spatial/` beside it -- exactly the layout
`spatialdata_io.visium` expects, which is why no staging copy or symlink of
the counts matrix is needed:

```
raw_files/CytAssist_FFPE_Human_Colon_Rep1/
├── CytAssist_FFPE_Human_Colon_Rep1_filtered_feature_bc_matrix.h5
├── CytAssist_FFPE_Human_Colon_Rep1_tissue_image.btf     # microscope H&E
├── CytAssist_FFPE_Human_Colon_Rep1_image.tif            # CytAssist capture
├── CytAssist_FFPE_Human_Colon_Rep1_cloupe.cloupe
├── CytAssist_FFPE_Human_Colon_Rep1_alignment_file.json
├── CytAssist_FFPE_Human_Colon_Rep1_probe_set.csv
├── CytAssist_FFPE_Human_Colon_Rep1_spatial.tar.gz       # kept
├── spatial/                                             # unpacked
│   ├── scalefactors_json.json
│   ├── tissue_positions.csv
│   └── tissue_hires_image.png, tissue_lowres_image.png, ...
├── analysis/                                            # unpacked
├── deconvolution/                                       # unpacked
└── ..._filtered_feature_bc_matrix.tar.gz                # NOT unpacked
```

The two `*_feature_bc_matrix.tar.gz` archives stay packed: they are MTX
triplets of the `.h5` files fetched alongside, so unpacking them roughly
doubles a bundle's size for no new information.

Second, each study is checked on disk once its assets settle. That check is
shared -- `BaseDownloader.verify_bundle` -- and each technology configures it
with four class-level tuples:

```python
VisiumDownloader.verify_bundle("/path/to/raw_files/CytAssist_FFPE_Human_Colon_Rep1")
# BundleCheck(study=..., present=frozenset({...}), missing_required=(),
#             missing_members=(), missing_expected=())
```

`BundleCheck` reports two vocabularies, because their failures have different
causes. **Kinds** come from classifying the files that were fetched, so a
missing required kind means an asset was never downloaded. **Members** are
globs expected *inside* the bundle, so a missing required member means an
archive did not unpack -- which nothing in the download results can reveal,
since the bytes arrived intact. For Visium the two nearly coincide; for
Xenium, whose payload arrives inside `*_outs.zip` with unprefixed member
names, only the member check says anything useful.

| | `required_kinds` | `required_members` |
| --- | --- | --- |
| Xenium | `outs` | `experiment.xenium`, `transcripts.parquet`, `cells.parquet`, `cell_boundaries.parquet`, `nucleus_boundaries.parquet`, `cell_feature_matrix.h5` |
| Visium | `spatial`, `filtered_matrix`, `tissue_image` | `spatial/scalefactors_json.json`, `spatial/tissue_positions*.csv` |

Each downloader then adds the one check its technology needs and the generic
pass cannot express: Visium calls out a missing `.cloupe` by name, and Xenium
flags an H&E that has no alignment CSV -- which converts without error but
lands on an Identity transform, silently unaligned.

Missing assets are logged, not raised: 10x's own manifests are incomplete for
some studies, and one study's gap is no reason to abandon the rest of a batch.

### Helper Functions

Most helpers below are technology-agnostic and actually live in
`spatialrefinery.core.utils`; `spatialrefinery.io.xenium` re-exports them
so the imports below keep working.

#### `create_circular_spots()` / `create_hexagonal_spots()`

Pool points into square-binned circular, or hexagonal, pseudo-spots.

**Parameters:**
- `df`: transcripts dataframe (pandas or dask) containing x, y coordinates
- `spot_size_um`: spot diameter in micrometers (default: 55.0)
- `key_x`, `key_y`: column names for the x/y coordinates (default: `"x"`, `"y"`)
- `overlap` (hexagonal only): fractional overlap between adjacent hexagons (default: 0.0)

**Returns:**
- `gpd.GeoDataFrame` of spot polygons with centroid coordinates in `x_um`/`y_um`

#### `fix_table_validation_errors()`

Rename `AnnData.var` columns that fail SpatialData table validation.

#### `create_pseudo_spots()`

Create pseudo-spots from transcripts (or cell boundaries) and add them to the SpatialData object.

**Parameters:**
- `sdata`: The SpatialData object containing transcripts and cell boundaries
- `spot_size_um`: The size of pseudo-spots in micrometers (default: 55)
- `overlap`: Fractional overlap between adjacent hexagonal spots (default: 0.06)
- `values`: Which element to aggregate into spots, `"transcripts"` or `"cell_boundaries"` (default: `"transcripts"`)

**Returns:**
- The updated SpatialData object with added pseudo-spots

### Batch Processing Example

```python
from pathlib import Path
from spatialrefinery.io import xenium_to_spatialdata

raw_data_dir = Path("/path/to/multiple/samples")
output_dir = Path("/path/to/output")

# Process all subdirectories as separate samples
for sample_dir in raw_data_dir.iterdir():
    if sample_dir.is_dir():
        try:
            zarr_path = xenium_to_spatialdata(dataset_path=sample_dir, output_path=output_dir, n_jobs=8)
            print(f"✅ Processed {sample_dir.name}")
        except Exception as e:
            print(f"❌ Failed to process {sample_dir.name}: {e}")
```

### Features

- ✅ Loads Xenium transcripts, cells, and boundaries
- ✅ Optionally includes aligned H&E images
- ✅ Creates pseudo-spots at custom sizes (mimics Visium)
- ✅ Determines which spots are "in tissue" based on cell boundaries
- ✅ Validates and fixes column names for SpatialData compatibility
- ✅ Splits protein sub-panels: antibodies to `obsm`, `var` kept to Gene Expression
- ✅ Parallel processing support
- ✅ Optional zip archive creation
- ✅ Automatic cleanup of temporary files
- ✅ Skip processing if output already exists

### Requirements

For Xenium dataset directory structure requirements, see the [Xenium Onboard Analysis Output documentation](https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-at-a-glance).

### Output Structure

A multi-tissue panel (377 genes) converted with `spot_sizes=[55, 100]`:

```python
example.zarr
├── Images
│   ├── 'he_image': DataTree[cyx] (3, 27323, 8832), (3, 13661, 4416), (3, 6830, 2208)
│   └── 'morphology_mip': DataTree[cyx] (1, 10346, 36943), (1, 5173, 18471), (1, 2586, 9235), ...
├── Labels
│   ├── 'cell_labels': DataTree[yx] (10346, 36943), (5173, 18471), (2586, 9235), (1293, 4617), (646, 2308)
│   └── 'nucleus_labels': DataTree[yx] (10346, 36943), (5173, 18471), (2586, 9235), (1293, 4617), (646, 2308)
├── Points
│   └── 'transcripts': DataFrame with shape: (5115684, 10) (3D points)
├── Shapes
│   ├── 'cell_boundaries': GeoDataFrame shape: (56510, 1) (2D shapes)
│   ├── 'nucleus_boundaries': GeoDataFrame shape: (56510, 1) (2D shapes)
│   ├── 'spots_55um': GeoDataFrame shape: (10502, 3) (2D shapes)
│   ├── 'spots_100um': GeoDataFrame shape: (3366, 3) (2D shapes)
│   └── 'tissue_contours': GeoDataFrame shape: (3501, 2) (2D shapes)
└── Tables
    ├── 'spots_55um_table': AnnData (10502, 377)
    ├── 'spots_100um_table': AnnData (3366, 377)
    └── 'table': AnnData (56510, 377)
```

Sizes vary by sample: 377 is this panel's gene count, not a fixed one. What is fixed is that `var`
holds the gene panel and nothing else -- the control and codeword feature types the cell-feature
matrix also carries are dropped, their per-cell totals already being in `obs`.

The morphology image depends on the bundle: Xenium Analyzer 2.0 and later write `morphology_focus`
with one channel per stain, older bundles the single `morphology_mip` above.

A **protein sub-panel** adds to this rather than changing its shape. Each antibody gets its own
`morphology_focus` channel, and the per-cell measurements sit beside the gene table, addressable by
antibody name:

```python
sdata["table"].obsm["protein_expression"]   # cells x antibodies DataFrame, columns are antibody names
sdata["table"].uns["protein_expression"]    # {"names", "gene_ids", "metric"}
```

`var` is the gene panel either way, so nothing downstream has to branch on whether a sample carried
antibodies. Those values are `MEAN_PER_CELL_STAIN` intensities, not counts, so they must not be
normalised the way transcript counts are. The pseudo-spot tables carry no protein channel: proteins
are measured per cell by antibody stain, and there is nothing in `transcripts` to aggregate onto the
hex lattice.
