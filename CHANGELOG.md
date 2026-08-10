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
