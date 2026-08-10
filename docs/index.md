# spatialrefinery

A toolkit for turning raw spatial-omics vendor outputs into analysis-ready data: 10x Genomics Xenium bundles become
[SpatialData](https://spatialdata.scverse.org/en/stable/) zarr stores, and whole-slide/microscopy images become pyramidal
OME-TIFF.

## What spatialrefinery does, end to end

```python
from spatialrefinery import download_xenium_study, xenium_to_spatialdata, convert_to_ometiff

# 1. Fetch a Xenium study's raw asset bundle from a `curl -O <url>` manifest
download_xenium_study("manifest.txt", outdir="raw_files", kinds=["outs"])

# 2. Convert the raw bundle into a SpatialData zarr store, with pseudo-spots
xenium_to_spatialdata(
    dataset_path="raw_files/my_study",
    output_path="processed",
    create_spots=True,
    spot_sizes=[55, 100],
)

# 3. Convert an associated whole-slide image to pyramidal OME-TIFF
convert_to_ometiff(source="raw_files/my_study/slide.svs", output_dir="processed")
```

## Highlights

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`download;1.5em` Download
Fetch a Xenium study's raw assets from a `curl -O <url>` manifest, with retries, atomic writes, and parallel workers.
:::

:::{grid-item-card} {octicon}`file-binary;1.5em` Convert to SpatialData
Turn a raw Xenium bundle -- transcripts, cell/nucleus boundaries, aligned H&E -- into one
[SpatialData](https://spatialdata.scverse.org/en/stable/) zarr store.
:::

:::{grid-item-card} {octicon}`apps;1.5em` Pseudo-spots
Bin transcripts or cell boundaries into Visium-like circular or hexagonal spots at any size, with configurable overlap.
:::

:::{grid-item-card} {octicon}`image;1.5em` Pyramidal imaging
Convert whole-slide images (`.svs`, `.ndpi`, `.tif`, `.czi`, ...) to tiled, multi-resolution OME-TIFF for fast viewing at
any zoom level.
:::

::::

## Which entry point do I want?

```{list-table}
:header-rows: 1

* - Task
  - Function
  - Tutorial
* - Download a Xenium study's raw assets
  - {py:obj}`~spatialrefinery.download_xenium_study`
  - [Downloading Xenium data](notebooks/xenium_download)
* - Convert a Xenium bundle to a SpatialData zarr store
  - {py:obj}`~spatialrefinery.xenium_to_spatialdata`
  - [Xenium to SpatialData zarr](notebooks/xenium_to_zarr)
* - ...and package it as a zip for transfer
  - {py:obj}`~spatialrefinery.xenium_to_spatialdata_zip`
  - [Xenium to SpatialData zarr](notebooks/xenium_to_zarr)
* - Convert a whole-slide image to pyramidal OME-TIFF
  - {py:obj}`~spatialrefinery.convert_to_ometiff`
  - [Converting images to OME-TIFF](notebooks/convert_to_ometiff)
```

```{toctree}
:hidden: true
:maxdepth: 1

installation.md
tutorials.md
api.md
citation.md
changelog.md
references.md
```
