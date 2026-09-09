# API reference

This page documents the functions used in the [tutorial notebooks](tutorials.md): the two entry points for pulling raw
vendor data, the two for converting a Xenium bundle to a
[SpatialData](https://spatialdata.scverse.org/en/stable/) zarr store, the one for converting whole-slide images to
pyramidal OME-TIFF, and the two that carry an H&E slide through nucleus segmentation. All but the last pair are
importable directly off the top-level package; the segmentation pair is reached through
`spatialrefinery.segmentation`, since it needs the optional `segmentation` extra.

## Downloading

Fetch a 10x Genomics study's raw asset bundle from a `curl -O <url>` manifest. Both downloaders retry transient
failures and write atomically, and unpack their technology's archives in place -- `*_outs.zip` for Xenium,
`spatial`/`analysis`/`deconvolution` tarballs for Visium.

```{eval-rst}
.. currentmodule:: spatialrefinery

.. autosummary::
    :toctree: generated
    :nosignatures:

    download_xenium_study
    download_visium_study
```

## Converting

Convert raw vendor bundles and whole-slide images into analysis-ready formats: a
[SpatialData](https://spatialdata.scverse.org/en/stable/) zarr store for Xenium, and pyramidal OME-TIFF for microscopy images.

```{eval-rst}
.. currentmodule:: spatialrefinery

.. autosummary::
    :toctree: generated
    :nosignatures:

    xenium_to_spatialdata
    xenium_to_spatialdata_zip
    convert_to_ometiff
```

## Segmenting

Segment nuclei in an H&E whole-slide image with [InstanSeg](https://github.com/instanseg/instanseg), then package the
boundaries as a [SpatialData](https://spatialdata.scverse.org/en/stable/) zarr store. Needs the optional `segmentation`
extra -- see [Installation](installation.md).

```{eval-rst}
.. currentmodule:: spatialrefinery.segmentation.instanseg

.. autosummary::
    :toctree: generated
    :nosignatures:

    segment_wsi
```

```{eval-rst}
.. currentmodule:: spatialrefinery.segmentation.to_spatialdata

.. autosummary::
    :toctree: generated
    :nosignatures:

    geojson_to_spatialdata
```

:::{note}
This reference is deliberately narrow. `spatialrefinery.core` and `spatialrefinery.io` contain additional lower-level
machinery -- the technology/converter registry, `BaseDownloader`, `XeniumConverter`, and the pseudo-spot helpers in
`core.utils` -- that other modules build on but that isn't part of the documented, stable surface yet. Expect it to change
without notice; the functions above are the supported way to use `spatialrefinery`.
:::
