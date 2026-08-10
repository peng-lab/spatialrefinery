# Tutorials

Practical, task-oriented notebooks -- what most users actually do with `spatialrefinery`. Each notebook is a narrated
version of the equivalent script in [`scripts/`](https://github.com/peng-lab/spatialrefinery/tree/main/scripts), and ends
with the plain command-line invocation if you'd rather run it as a batch job.

1. **[Downloading Xenium data](notebooks/xenium_download)** -- fetch a 10x Genomics Xenium study's raw asset bundle from a
   `curl -O <url>` manifest.
2. **[Xenium to SpatialData zarr](notebooks/xenium_to_zarr)** -- convert a raw Xenium bundle into a SpatialData zarr
   store, with aligned H&E images and Visium-like pseudo-spots.
3. **[Converting images to OME-TIFF](notebooks/convert_to_ometiff)** -- convert whole-slide/microscopy images to
   pyramidal OME-TIFF for fast, multi-resolution viewing.

```{toctree}
:hidden: true
:maxdepth: 1

notebooks/xenium_download
notebooks/xenium_to_zarr
notebooks/convert_to_ometiff
```
