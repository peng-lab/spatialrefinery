# Installation

You need Python 3.12 or newer. We recommend managing dependencies in a project-specific virtual environment using
[uv](https://github.com/astral-sh/uv), to avoid dependency conflicts with the rest of your system.

## Development install

`spatialrefinery` is not yet published to PyPI. Install the latest development version directly from GitHub:

::::{tab-set}
:::{tab-item} uv
:sync: uv

```bash
uv add git+https://github.com/peng-lab/spatialrefinery.git
```

:::
:::{tab-item} pip
:sync: pip

```bash
pip install git+https://github.com/peng-lab/spatialrefinery.git
```

:::
::::

## Optional: CZI support

Reading Zeiss CZI images (via [bioio](https://github.com/bioio-devs/bioio)) is an optional extra, kept separate because it
pulls in additional native dependencies:

```bash
pip install "spatialrefinery[czi] @ git+https://github.com/peng-lab/spatialrefinery.git"
```

Without the `czi` extra, {py:obj}`~spatialrefinery.convert_to_ometiff` still handles every other registered format
(`.svs`, `.ndpi`, `.tif`, `.tiff`, `.mrxs`, `.scn`, `.bif`, `.vms`, `.svslide`) -- only `.czi` requires it.

## Optional: nucleus segmentation

Nucleus segmentation on H&E whole-slide images needs the `segmentation` extra, which pulls
[InstanSeg](https://github.com/instanseg/instanseg) (`instanseg-torch`) plus `rasterio`, `geojson` and `tiffslide`:

```bash
pip install "spatialrefinery[segmentation] @ git+https://github.com/peng-lab/spatialrefinery.git"
```

It is kept optional because `instanseg-torch` brings torch and therefore multi-GB CUDA wheels, which most
`spatialrefinery` uses do not need. A CUDA GPU is strongly recommended -- see
[Nucleus segmentation to SpatialData zarr](notebooks/segment_nuclei).

## Verifying the install

```python
import spatialrefinery

print(spatialrefinery.__version__)
```

If this prints a version string without raising an `ImportError`, you're ready for the [tutorials](tutorials.md).
