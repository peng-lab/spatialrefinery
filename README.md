# spatialrefinery

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]
[![Coverage][badge-coverage]][coverage]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/peng-lab/spatialrefinery/test.yaml?branch=main
[badge-docs]: https://app.readthedocs.org/projects/spatialrefinery/badge/
[badge-coverage]: https://codecov.io/github/peng-lab/spatialrefinery/branch/main/graph/badge.svg

A toolkit for turning raw spatial-omics vendor outputs into analysis-ready data: 10x Genomics Xenium bundles become
[SpatialData][] zarr stores, whole-slide/microscopy images become pyramidal OME-TIFF, and H&E slides can be segmented
into nucleus boundaries packaged as a SpatialData store of their own.

## Why spatialrefinery

`spatialrefinery` was built for [Phoenix][], which predicts spatial transcriptomics from routine histology and needs
training data assembled the same way across every Xenium sample: an aligned H&E image and transcript-derived
pseudo-bulk spots at several resolutions, all in one SpatialData object. Rather than re-deriving that pipeline per
sample, `spatialrefinery` turns a raw Xenium bundle -- transcripts, cell/nucleus boundaries, aligned H&E -- into a
zarr store with pseudo-spots binned at whatever sizes a given resolution needs (e.g. 55um, 100um), and converts the
paired whole-slide image to a pyramidal OME-TIFF for fast viewing. Run it over a batch of raw bundles and you get one
common, multi-resolution dataset to train Phoenix on.

The same pipeline also runs on histology that has no paired Xenium run: nucleus segmentation turns an H&E slide into a
SpatialData store of nucleus boundaries with an empty table over a gene panel -- the shape Phoenix predicts into.

## Getting started

Please refer to the [documentation][], in particular the [installation guide][], [tutorials][], and
[API documentation][].

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

Nucleus segmentation needs the optional `segmentation` extra, and is reached through `spatialrefinery.segmentation`
rather than the top-level package:

```python
from spatialrefinery.segmentation.instanseg import segment_wsi
from spatialrefinery.segmentation.to_spatialdata import default_zarr_path, geojson_to_spatialdata

slide = "processed/slide.ome.tif"

# 4. Segment nuclei with InstanSeg -> segmentation/slide.ome.tif/cells.geojson
cells = segment_wsi(slide, outdir="segmentation")

# 5. Package the boundaries with the slide -> processed/slide.zarr (+ .zarr.zip)
geojson_to_spatialdata(
    geojson_path=cells,
    zarr_path=default_zarr_path(slide, "processed"),
    image_path=slide,
    template_adata_path="panel_template.h5ad",
)
```

## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

We recommend managing dependencies in project-specific virtual environments to avoid dependency conflicts.
This is most convenient using package managers such as [uv][].
Choose from the options below to install spatialrefinery:

<!--
1. Add the latest release of `spatialrefinery` from [PyPI][] to your `uv` project:

   ```bash
   uv add spatialrefinery
   ```

1. Install the latest release into a [standard virtual environment][venv]:

   ```bash
   (after activating your venv)
   pip install spatialrefinery
   ```

-->

1. Install the latest development version:

   ```bash
   pip install git+https://github.com/peng-lab/spatialrefinery.git  # (or `uv add`)
   ```

### Optional extras

```bash
# Zeiss CZI images, via bioio
pip install "spatialrefinery[czi] @ git+https://github.com/peng-lab/spatialrefinery.git"

# H&E nucleus segmentation, via InstanSeg
pip install "spatialrefinery[segmentation] @ git+https://github.com/peng-lab/spatialrefinery.git"
```

`segmentation` is kept out of the base install because it pulls `instanseg-torch`, and therefore torch's multi-GB CUDA
wheels, which most uses do not need. A CUDA GPU is strongly recommended for it.

See the [installation guide][] for details and a verification snippet.

## Release notes

See the [changelog][].

## Contact

For questions, help requests, or to report a bug, please use the [issue tracker][].

## Citation

If you use `spatialrefinery`, please cite the Phoenix paper it was built for:

```bibtex
@article{tran/gindra2026.04.25.720812,
    author = {Tran, Manuel and Gindra, Rushin H. and Putze, Philipp and Senbai, Kang and Palla, Giovanni and Kos, Tina and Falcomat{\`a}, Chiara and Wang, Chen and Guo, Ruifeng (Ray) and Boxberg, Melanie and Berclaz, Luc M. and Lindner, Lars H. and Bergmayr, Linda and Kn{\"o}sel, Thomas and Jurmeister, Philipp and Klauschen, Frederick and Homicsko, Krisztian and Gottardo, Raphael and Eckstein, Markus and Matek, Christian and Mock, Andreas and Theis, Fabian J. and Saur, Dieter and Peng, Tingying},
    title = {Pan-cancer virtual spatial transcriptomics from routine histology with Phoenix},
    year = {2026},
    journal = {bioRxiv},
    doi = {https://doi.org/10.64898/2026.04.25.720812},
}
```

`spatialrefinery` also builds on [SpatialData][] and [spatialdata-io][]; if your work relies on the underlying data
model, please also cite the [scverse paper][]. See [Citation & License][] for details.

## License

`spatialrefinery` is released under the [BSD 3-Clause License][license].

[phoenix]: https://doi.org/10.64898/2026.04.25.720812
[spatialdata]: https://spatialdata.scverse.org/en/stable/
[spatialdata-io]: https://spatialdata.scverse.org/projects/io/en/stable/
[scverse paper]: https://doi.org/10.1038/s41587-023-01733-8
[license]: https://github.com/peng-lab/spatialrefinery/blob/main/LICENSE
[installation guide]: https://spatialrefinery.readthedocs.io/page/installation.html
[tutorials]: https://spatialrefinery.readthedocs.io/page/tutorials.html
[citation & license]: https://spatialrefinery.readthedocs.io/page/citation.html
[uv]: https://github.com/astral-sh/uv
[issue tracker]: https://github.com/peng-lab/spatialrefinery/issues
[tests]: https://github.com/peng-lab/spatialrefinery/actions/workflows/test.yaml
[coverage]: https://app.codecov.io/github/peng-lab/spatialrefinery
[documentation]: https://spatialrefinery.readthedocs.io
[changelog]: https://spatialrefinery.readthedocs.io/page/changelog.html
[api documentation]: https://spatialrefinery.readthedocs.io/page/api.html
[pypi]: https://pypi.org/project/spatialrefinery
[venv]: https://docs.python.org/3/tutorial/venv.html
