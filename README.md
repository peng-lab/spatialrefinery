# spatialrefinery

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/peng-lab/spatialrefinery/test.yaml?branch=main
[badge-docs]: https://app.readthedocs.org/projects/spatialrefinery/badge/

A toolkit for turning raw spatial-omics vendor outputs into analysis-ready data: 10x Genomics Xenium bundles become
[SpatialData][] zarr stores, and whole-slide/microscopy images become pyramidal OME-TIFF.

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

See the [installation guide][] for the optional `czi` extra and a verification snippet.

## Release notes

See the [changelog][].

## Contact

For questions, help requests, or to report a bug, please use the [issue tracker][].

## Citation

A citation for `spatialrefinery` is forthcoming (t.b.a.). `spatialrefinery` builds on [SpatialData][]; if your work
relies on the underlying data model, please also cite the [scverse paper][]. See [Citation & License][] for details.

[spatialdata]: https://spatialdata.scverse.org/en/stable/
[scverse paper]: https://doi.org/10.1038/s41587-023-01733-8
[installation guide]: https://spatialrefinery.readthedocs.io/page/installation.html
[tutorials]: https://spatialrefinery.readthedocs.io/page/tutorials.html
[citation & license]: https://spatialrefinery.readthedocs.io/page/citation.html
[uv]: https://github.com/astral-sh/uv
[issue tracker]: https://github.com/peng-lab/spatialrefinery/issues
[tests]: https://github.com/peng-lab/spatialrefinery/actions/workflows/test.yaml
[documentation]: https://spatialrefinery.readthedocs.io
[changelog]: https://spatialrefinery.readthedocs.io/page/changelog.html
[api documentation]: https://spatialrefinery.readthedocs.io/page/api.html
[pypi]: https://pypi.org/project/spatialrefinery
[venv]: https://docs.python.org/3/tutorial/venv.html
