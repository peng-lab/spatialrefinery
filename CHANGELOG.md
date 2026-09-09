# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [Unreleased]

### Added

- `spatialrefinery.core`: a technology/converter registry (`registry`), shared
  spatial-omics helpers (`utils`), an image-to-pyramidal-OME-TIFF converter
  (`converter`, with an optional `czi` extra for Zeiss CZI via `bioio`), and a
  retrying, atomic-write asset downloader (`downloader`).
- `spatialrefinery.io.xenium`: convert 10x Genomics Xenium bundles to
  SpatialData zarr stores, optionally with aligned H&E images, tissue
  segmentation, and Visium-like pseudo-spots; download a Xenium study's raw
  assets from a `curl -O <url>` manifest.
- `spatialrefinery.io.visium`: download a 10x Visium / CytAssist study's raw
  assets from a `curl -O <url>` manifest, unpacking the `spatial`, `analysis`
  and `deconvolution` tarballs in place. The two `*_feature_bc_matrix.tar.gz`
  archives are left packed, being MTX copies of the `.h5` files fetched
  alongside. Each study is verified on disk afterwards
  reporting missing required assets and whether a `.cloupe` was found rather
  than failing a batch on one incomplete study.
- `BaseDownloader.verify_bundle` and `BundleCheck`: per-study bundle
  verification driven by four class-level tuples (`required_kinds`,
  `expected_kinds`, `required_members`, `expected_members`), logged from a
  `post_process` that is no longer a no-op. It reports asset *kinds* and
  extracted *members* separately, because a missing kind means an asset was
  never downloaded whereas a missing member means an archive did not unpack --
  a failure invisible in the download results, since the bytes arrived intact.
- `spatialrefinery.core.utils.collapse_url_slashes`, applied in
  `RemoteAsset.from_url`: some published 10x manifests carry a doubled path
  separator (`.../spatial-exp/3.1.3//<study>/...`), which the CDN answers
  with `403 Forbidden`. It was invisible before because the study name still
  parsed correctly, so all 8 assets of one Visium study failed as an
  apparent permissions error.
- `spatialrefinery.core.utils.safe_extract_tar` and `tar_root_dir`, and
  `BaseDownloader.extract_archive`/`require_extract`: tar archives are now
  recognised alongside zips (`Path.suffix` reports `.gz` for `a.tar.gz`, so
  tarballs were previously never unpacked), a failed extraction of a
  load-bearing archive fails its asset instead of only warning, and a *flat*
  archive is given a directory of its own so concurrent extractions cannot
  overwrite each other.
- `spatialrefinery.segmentation`: nucleus segmentation on H&E whole-slide
  images via InstanSeg (`instanseg`), and export of the resulting boundaries
  as a SpatialData zarr store carrying the slide image, the nucleus polygons,
  and a table over a template gene panel (`to_spatialdata`). Needs the
  optional `segmentation` extra. The store is named for the slide's stem, so
  `slide.ome.tif` yields `slide.zarr`.
- Tutorial notebook for the segmentation pipeline, taking an OME-TIFF through
  segmentation to a written SpatialData zarr.
