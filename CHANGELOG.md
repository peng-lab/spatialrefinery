# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [Unreleased]

### Fixed

- `spatialrefinery.io.xenium`: `*_he_alignment.csv` -- the name the Atera "WTA
  Preview" bundles use instead of 10x's `*_he_imagealignment.csv` -- was
  missing from the asset-kind map, so those files classified as `"unknown"`
  and `kinds=["he_alignment"]` silently skipped them. `find_xenium_files`
  already recognised both spellings.
- `spatialrefinery.core.converter.downsample_plane`: halving a single-channel
  `"minisblack"` plane silently swapped height and width from the second
  sub-level onward. `cv2.resize` drops a trailing size-1 channel axis
  (`(H, W, 1) -> (H, W)`), and the subsequent `moveaxis(img, -1, 0)` then
  transposed that now-2D result instead of restoring the channel axis it had
  removed. Invisible until now because the one existing `"minisblack"`
  caller, `BioioImageConverter`, always `np.squeeze`s a single channel down
  to plain `(H, W)` before it reaches `downsample_plane`.

### Added

- `spatialrefinery.core`: a technology/converter registry (`registry`), shared
  spatial-omics helpers (`utils`), an image-to-pyramidal-OME-TIFF converter
  (`converter`, with an optional `czi` extra for Zeiss CZI via `bioio`), and a
  retrying, atomic-write asset downloader (`downloader`).
- `spatialrefinery.io.xenium`: convert 10x Genomics Xenium bundles to
  SpatialData zarr stores, optionally with aligned H&E images, tissue
  segmentation, and Visium-like pseudo-spots; download a Xenium study's raw
  assets from a `curl -O <url>` manifest.
- `spatialrefinery.io.xenium`: Xenium protein sub-panels keep their antibody
  measurements. The cell-feature matrix is now read with `gex_only=False` and
  the table is split by feature type: `var` is restricted to the gene panel,
  and any `Protein Expression` rows move into
  `table.obsm["protein_expression"]` -- a cells x antibodies DataFrame columned
  by antibody name, with `uns["protein_expression"]` recording their `names`,
  `gene_ids` and `metric`. Antibodies cannot stay in `var`: one can carry the
  same name as a gene targeting the same molecule, which makes `var_names`
  non-unique, `table[:, name]` ambiguous, and would have `create_pseudo_spots`
  emit every such gene twice into the spot table. Their values are
  `MEAN_PER_CELL_STAIN` intensities, not counts, so a mixed table also made
  `X.sum(axis=1)` meaningless against the transcript-only `obs["total_counts"]`.
  Control and codeword feature types are dropped, as before, their per-cell
  totals already being in `obs`. Samples without a protein sub-panel are
  unaffected -- their table is bit-identical to what the previous
  `gex_only=True` read produced, and gains no `obsm` entry -- and panel sizes
  are read off the data, so nothing assumes a particular gene count.
  Pseudo-spot tables carry no protein channel: proteins are measured per cell
  by antibody stain, with nothing in `transcripts` to aggregate. A Xenium
  protein store used as a `geojson_to_spatialdata` template therefore
  contributes a gene-panel-only `var`.
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
  Xenium verifies the six members `spatialdata_io.xenium` opens, and flags an
  H&E that has no alignment CSV: it converts without error but lands on an
  Identity transform, silently unaligned, and 3 of 65 sample bundles are in
  that state.
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
- `spatialrefinery.core.converter`: read Olympus cellSens (`.vsi`),
  PerkinElmer/Akoya QPTIFF (`.qptiff`), and Zeiss ZVI/AFI (`.zvi`, `.afi`)
  whole-slide/microscopy images via `slideio`, joining `openslide` and
  `bioio` as a third `ImageConverter` backend (`SlideioImageConverter`).
  Streams level 0 the same way `OpenSlideImageConverter` does -- full-width
  bands halved on the way past into an on-disk memmap -- via a new
  `SlideioTiledSource`, so a whole-slide plane is never materialised in
  full. A multichannel (non-RGB) scene is written as one page per channel,
  which `tifffile` requires its tile iterator to fill in page-major order
  (every tile of channel 0 before any of channel 1); `SlideioTiledSource`
  reads and stages one channel at a time to match. `slideio` is a required
  dependency, not an optional extra: its wheels are `numpy`-only and
  BSD-3-Clause, matching this project's license, though it currently ships
  none for `manylinux` aarch64 or glibc < 2.28. DICOM WSI (`.dcm`) is
  deliberately not registered: it is normally a directory of instances,
  which suffix-based dispatch does not address, and it has not been
  exercised against a real file.

### Changed

- `spatialrefinery.core.converter.OpenSlideImageConverter` and the new
  `SlideioImageConverter` now raise if the source reports no physical pixel
  size (`mpp-x`/`mpp-y`, or `slideio`'s exactly-`1.0`-metre/pixel
  no-metadata fallback), instead of silently writing the slide at an assumed
  1.0 um/px. A slide converted at the wrong scale is exactly the "silent bad
  sample" this project's `CLAUDE.md` says must not happen -- every
  downstream pseudo-spot size and Phoenix training target derives from it.
  Both converters, and `convert_to_ometiff`, take an `mpp=` override (a
  single value, or an `(x, y)` pair, in micrometres) for a source that
  genuinely carries none; `scripts/convert_to_ometiff.py` exposes it as
  `--mpp`.
