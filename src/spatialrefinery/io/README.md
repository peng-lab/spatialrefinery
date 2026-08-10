# SpatialRefinery I/O Module

This module provides functions for reading and converting various spatial transcriptomics data formats to SpatialData zarr format.

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

Create pseudo-spots from transcripts and add them to the SpatialData object.

**Parameters:**
- `sdata`: The SpatialData object containing transcripts and cell boundaries
- `specs`: Xenium experiment specifications (must contain 'pixel_size')
- `spot_size_um`: The size of pseudo-spots in micrometers (default: 55)

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
- ✅ Parallel processing support
- ✅ Optional zip archive creation
- ✅ Automatic cleanup of temporary files
- ✅ Skip processing if output already exists

### Requirements

For Xenium dataset directory structure requirements, see the [Xenium Onboard Analysis Output documentation](https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/analysis/xoa-output-at-a-glance).

### Output Structure

```python
example.zarr
├── Images
│   └── 'he_image': DataTree[cyx] (3, 27502, 14896), (3, 13751, 7448), (3, 6875, 3724)
├── Labels
│   ├── 'cell_labels': DataTree[yx] (13770, 34155), (6885, 17077), (3442, 8538), (1721, 4269), (860, 2134)
│   └── 'nucleus_labels': DataTree[yx] (13770, 34155), (6885, 17077), (3442, 8538), (1721, 4269), (860, 2134)
├── Points
│   └── 'transcripts': DataFrame with shape: (<Delayed>, 11) (3D points)
├── Shapes
│   ├── 'cell_boundaries': GeoDataFrame shape: (140702, 1) (2D shapes)
│   ├── 'cell_circles': GeoDataFrame shape: (140702, 2) (2D shapes)
│   ├── 'nucleus_boundaries': GeoDataFrame shape: (136531, 1) (2D shapes)
│   ├── 'spots_55um': GeoDataFrame shape: (6705, 3) (2D shapes)
│   └── 'spots_100um': GeoDataFrame shape: (2086, 3) (2D shapes)
└── Tables
    ├── 'spots_55um_table': AnnData (6705, 377)
    ├── 'spots_100um_table': AnnData (2086, 377)
    └── 'table': AnnData (140702, 377)
```
