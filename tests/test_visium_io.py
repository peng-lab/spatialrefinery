"""Tests for `spatialrefinery.io.visium`'s download and verification surface.

Covers asset classification (including the two suffix collisions that
longest-suffix-wins exists to resolve), manifest resolution, the
flat-archive redirect that keeps two tarballs from overwriting each other,
and the bundle completeness report.

Deliberately not covered: any real download. `fetch`/`run` are the shared
`BaseDownloader` engine, tested in `test_downloader.py`; here the network
is avoided via `dry_run` and by calling the pure hooks directly. Bundles
are synthesised as empty files with the right names, since classification
and verification read filenames, never contents.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spatialrefinery.core.registry import get_technology
from spatialrefinery.io.visium import (
    VisiumConverter,
    VisiumDownloader,
    assign_cells_to_spots,
    classify_visium_asset,
    find_visium_files,
    visium_he_mpp,
    visium_to_spatialdata,
)

STUDY = "CytAssist_11mm_FFPE_Human_Kidney"


def _touch(directory: Path, *names: str) -> None:
    """Create empty files named `<STUDY>_<name>` (or `name` verbatim if it has a `/`)."""
    for name in names:
        path = directory / name if "/" in name else directory / f"{STUDY}_{name}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def _write_tar(path: Path, names: tuple[str, ...]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name in names:
            info = tarfile.TarInfo(name)
            info.size = 1
            tf.addfile(info, io.BytesIO(b"x"))


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        ("_spatial.tar.gz", "spatial"),
        ("_analysis.tar.gz", "analysis"),
        ("_deconvolution.tar.gz", "deconvolution"),
        ("_filtered_feature_bc_matrix.tar.gz", "filtered_matrix_mtx"),
        ("_raw_feature_bc_matrix.tar.gz", "raw_matrix_mtx"),
        ("_filtered_feature_bc_matrix.h5", "filtered_matrix"),
        ("_raw_feature_bc_matrix.h5", "raw_matrix"),
        ("_raw_probe_bc_matrix.h5", "raw_probe_matrix"),
        ("_molecule_info.h5", "molecule_info"),
        ("_tissue_image.ome.tif", "tissue_image"),
        ("_tissue_image.tif", "tissue_image"),
        ("_tissue_image.btf", "tissue_image"),
        ("_image.tif", "cytassist_image"),
        ("_alignment_file.json", "alignment"),
        ("_probe_set.csv", "probe_set"),
        ("_feature_reference.csv", "feature_reference"),
        ("_isotype_normalization_factors.csv", "isotype_normalization"),
        ("_metrics_summary.csv", "metrics_summary"),
        ("_web_summary.html", "web_summary"),
        ("_cloupe.cloupe", "cloupe"),
    ],
)
def test_classify_covers_every_manifest_suffix(suffix: str, expected: str) -> None:
    """Every filename tail 10x ships must classify; an `unknown` means an unhandled asset."""
    assert classify_visium_asset(f"{STUDY}{suffix}") == expected


@pytest.mark.parametrize(
    ("filename", "expected", "shadowed_by"),
    [
        # Both matrix tarballs end in `_feature_bc_matrix.tar.gz`, so a shorter
        # key would collapse them into one kind.
        (f"{STUDY}_raw_feature_bc_matrix.tar.gz", "raw_matrix_mtx", "filtered_matrix_mtx"),
        (f"{STUDY}_filtered_feature_bc_matrix.tar.gz", "filtered_matrix_mtx", "raw_matrix_mtx"),
        # `_tissue_image.tif` also ends in `_image.tif`: the microscope image
        # must not be mistaken for the coarser CytAssist capture.
        (f"{STUDY}_tissue_image.tif", "tissue_image", "cytassist_image"),
    ],
)
def test_classify_prefers_the_longest_matching_suffix(filename: str, expected: str, shadowed_by: str) -> None:
    """Overlapping suffixes resolve to the most specific kind, not the first match."""
    kind = classify_visium_asset(filename)
    assert kind == expected
    assert kind != shadowed_by


def test_classify_unknown_suffix() -> None:
    """An unrecognised asset is labelled, not guessed at, so `kinds=` filtering stays honest."""
    assert classify_visium_asset(f"{STUDY}_something_new.parquet") == "unknown"


def test_iter_assets_derives_study_and_kind(tmp_path) -> None:
    """Manifest lines resolve to study/kind, including 10x's double-slash URLs.

    Some 10x manifests carry `spatial-exp/3.1.3//<study>/...`. The empty
    path segment must not become the study name, *and* it must be collapsed
    out of the URL: the CDN answers the doubled path with 403, so leaving it
    in makes every asset of that study undownloadable.
    """
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        "# Output Files\n"
        "curl -O https://example.com/samples/2.1.0/study_a/study_a_spatial.tar.gz\n"
        "curl -O https://example.com/samples/3.1.3//study_b/study_b_cloupe.cloupe\n"
        "not a curl line\n"
    )

    assets = list(VisiumDownloader(tmp_path).iter_assets(manifest))
    assert [(a.study, a.kind) for a in assets] == [("study_a", "spatial"), ("study_b", "cloupe")]
    assert assets[1].url == "https://example.com/samples/3.1.3/study_b/study_b_cloupe.cloupe"


def test_plan_filters_by_kind(tmp_path) -> None:
    """`kinds=` narrows a manifest without any network access."""
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(
        "curl -O https://example.com/s/study_a/study_a_spatial.tar.gz\n"
        "curl -O https://example.com/s/study_a/study_a_web_summary.html\n"
    )

    assets = VisiumDownloader(tmp_path, dry_run=True).plan(manifest, kinds=["spatial"])
    assert [a.kind for a in assets] == ["spatial"]


def test_find_visium_files_does_not_shadow_the_cytassist_image(tmp_path) -> None:
    """`*_image.tif` also matches `*_tissue_image.tif`; the two must stay distinguishable.

    Without the exclusion the microscope H&E would be reported as the
    CytAssist capture, and a conversion would silently use the wrong image.
    """
    bundle = tmp_path / STUDY
    bundle.mkdir()
    _touch(bundle, "image.tif", "tissue_image.tif")

    files = find_visium_files(bundle)
    assert files["cytassist_image"] == bundle / f"{STUDY}_image.tif"
    assert files["tissue_image"] == bundle / f"{STUDY}_tissue_image.tif"


def test_find_visium_files_prefers_ome_tif_then_tif_then_btf_for_the_microscope_image(tmp_path) -> None:
    """A pyramidal `.ome.tif` beats the vendor `.tif`, which beats `.btf`, whatever else is in the bundle."""
    bundle = tmp_path / STUDY
    bundle.mkdir()
    _touch(bundle, "tissue_image.btf")
    assert find_visium_files(bundle)["tissue_image"] == bundle / f"{STUDY}_tissue_image.btf"
    _touch(bundle, "tissue_image.tif")
    assert find_visium_files(bundle)["tissue_image"] == bundle / f"{STUDY}_tissue_image.tif"
    _touch(bundle, "tissue_image.ome.tif")
    assert find_visium_files(bundle)["tissue_image"] == bundle / f"{STUDY}_tissue_image.ome.tif"


def test_verify_visium_bundle_accepts_a_complete_bundle(tmp_path) -> None:
    """The happy path: nothing required missing, spatial unpacked, cloupe found."""
    bundle = tmp_path / STUDY
    bundle.mkdir()
    _touch(
        bundle,
        "filtered_feature_bc_matrix.h5",
        "tissue_image.btf",
        "image.tif",
        "cloupe.cloupe",
        "spatial.tar.gz",
        "analysis.tar.gz",
        "spatial/scalefactors_json.json",
        "spatial/tissue_positions.csv",
    )

    check = VisiumDownloader.verify_bundle(bundle)
    assert check.study == STUDY
    assert check.ok
    assert "cloupe" in check.present
    assert check.missing_required == ()
    assert check.missing_members == ()
    assert check.missing_expected == ()


def test_verify_visium_bundle_reports_missing_required_kinds(tmp_path) -> None:
    """An upstream manifest can omit the counts matrix entirely; that is reported, not raised.

    `Visium_V2_Human_Lymph_Node` in the reference cohort ships neither a
    counts matrix nor a `.cloupe`, so raising here would abort a whole batch
    on one study's incomplete manifest.
    """
    bundle = tmp_path / STUDY
    bundle.mkdir()
    _touch(bundle, "spatial.tar.gz", "spatial/scalefactors_json.json", "spatial/tissue_positions.csv")

    check = VisiumDownloader.verify_bundle(bundle)
    assert not check.ok
    assert "cloupe" not in check.present
    assert set(check.missing_required) == {"filtered_matrix", "tissue_image"}
    # `cloupe` gets its own dedicated warning in `post_process`, so listing it
    # in `expected_kinds` too would log the same absence twice.
    assert "cloupe" not in check.missing_expected


def test_verify_visium_bundle_detects_an_unextracted_spatial_archive(tmp_path) -> None:
    """A downloaded-but-unpacked `spatial.tar.gz` is not a usable bundle.

    `present` sees the archive, so only a filesystem check inside `spatial/`
    can tell the two apart -- which is why verification reads the disk
    rather than the download results.
    """
    bundle = tmp_path / STUDY
    bundle.mkdir()
    _touch(bundle, "filtered_feature_bc_matrix.h5", "tissue_image.btf", "spatial.tar.gz")

    check = VisiumDownloader.verify_bundle(bundle)
    assert "spatial" in check.present
    assert check.missing_required == ()
    # The archive is present as a *kind*; only the member check inside
    # `spatial/` can tell a downloaded tarball from an extracted one.
    assert check.missing_members == ("spatial/scalefactors_json.json", "spatial/tissue_positions*.csv")
    assert not check.ok


def test_verify_visium_bundle_tolerates_a_missing_directory(tmp_path) -> None:
    """A study whose every asset failed leaves no directory; that must not raise."""
    check = VisiumDownloader.verify_bundle(tmp_path / "never_downloaded")
    assert check.present == frozenset()
    assert not check.ok


def test_extract_archive_keeps_a_self_contained_archive_in_place(tmp_path) -> None:
    """`<study>_spatial.tar.gz` unpacks to `<bundle>/spatial/`, not a nested duplicate.

    That flat-beside-the-assets layout is what `spatialdata_io.visium`
    expects, so getting it right here removes the need for a staging copy.
    """
    bundle = tmp_path / STUDY
    bundle.mkdir()
    archive = bundle / f"{STUDY}_spatial.tar.gz"
    _write_tar(archive, ("spatial/scalefactors_json.json", "spatial/tissue_positions.csv"))

    downloader = VisiumDownloader(tmp_path)
    asset = next(iter(downloader.iter_assets([f"https://example.com/s/{STUDY}/{archive.name}"])))
    downloader.extract_archive(asset, archive)

    assert (bundle / "spatial" / "scalefactors_json.json").exists()
    assert not (bundle / "spatial" / "spatial").exists()


def test_extract_archive_redirects_a_flat_archive(tmp_path) -> None:
    """Two flat archives sharing a member name must not overwrite each other.

    They extract from a thread pool into one directory, so an in-place
    unpack would be both lossy and non-deterministic. Each therefore gets a
    directory named for its kind.
    """
    bundle = tmp_path / STUDY
    bundle.mkdir()
    members = ("matrix.mtx.gz", "barcodes.tsv.gz", "features.tsv.gz")

    downloader = VisiumDownloader(tmp_path)
    both = (("filtered_feature_bc_matrix", "filtered_matrix_mtx"), ("raw_feature_bc_matrix", "raw_matrix_mtx"))
    for suffix, kind in both:
        archive = bundle / f"{STUDY}_{suffix}.tar.gz"
        _write_tar(archive, members)
        asset = next(iter(downloader.iter_assets([f"https://example.com/s/{STUDY}/{archive.name}"])))
        assert asset.kind == kind
        downloader.extract_archive(asset, archive)

    for kind in ("filtered_matrix_mtx", "raw_matrix_mtx"):
        assert (bundle / kind / "matrix.mtx.gz").exists()
    assert not (bundle / "matrix.mtx.gz").exists()


def test_visium_is_registered_as_a_technology() -> None:
    """`io/__init__.py` must re-export the module, or `_ensure_builtins` never imports it."""
    spec = get_technology("visium")
    assert spec.downloader is VisiumDownloader
    assert spec.file_finder is find_visium_files
    assert spec.converter is VisiumConverter
    assert get_technology("cytassist").name == "visium"


# --------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------- #


def test_assign_cells_to_spots_leaves_cells_in_the_gaps_unassigned() -> None:
    """A cell inside a disk gets that spot's id, a cell in the gap between spots gets `<NA>`."""
    spot_xy = np.array([[0.0, 0.0], [100.0, 0.0]])
    cell_xy = np.array(
        [
            [5.0, 5.0],  # inside spot 10
            [100.0, 20.0],  # inside spot 20, off-centre
            [50.0, 0.0],  # the gap: 50 um from both centres, radius 27.5
            [27.4, 0.0],  # just inside spot 10's rim
            [27.6, 0.0],  # just outside it
        ]
    )

    assigned = assign_cells_to_spots(cell_xy, spot_xy, np.array([10, 20]), radius=27.5)

    assert str(assigned.dtype) == "Int64"  # nullable, or the NA could not live next to integer ids
    assert assigned.tolist() == [10, 20, pd.NA, 10, pd.NA]


def test_assign_cells_to_spots_reports_the_spot_id_not_its_row() -> None:
    """Ids come from `spot_ids`, so a shuffled spot order cannot relabel the cells."""
    assigned = assign_cells_to_spots(
        np.array([[0.0, 0.0]]), np.array([[500.0, 0.0], [0.0, 0.0]]), np.array([7, 3]), radius=27.5
    )
    assert assigned.tolist() == [3]


def _write_positions(path: Path, centres_px: list[tuple[float, float]], *, header: bool = True) -> Path:
    """Write a tissue-positions file (all barcodes in tissue) for spot centres given as `(x, y)` pixels."""
    rows = [(f"BC{i}-1", 1, i, i, y, x) for i, (x, y) in enumerate(centres_px)]
    frame = pd.DataFrame(
        rows, columns=["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    )
    frame.to_csv(path, index=False, header=header)
    return path


def _hex_lattice_px(pitch_px: float, n: int = 6) -> list[tuple[float, float]]:
    """An `n` x `n` Visium-style lattice: rows `pitch * sqrt(3) / 2` apart, every other row shifted half a pitch."""
    return [
        ((col + 0.5 * (row % 2)) * pitch_px, row * pitch_px * np.sqrt(3) / 2) for row in range(n) for col in range(n)
    ]


@pytest.mark.parametrize("header", [True, False], ids=["tissue_positions.csv", "tissue_positions_list.csv"])
def test_visium_he_mpp_is_the_lattice_pitch_over_its_pixel_spacing(tmp_path, header: bool) -> None:
    """100 um between neighbouring spots, 400 px apart in the image, is 0.25 um/px -- in both file formats."""
    positions = _write_positions(tmp_path / "tissue_positions.csv", _hex_lattice_px(400.0), header=header)
    assert visium_he_mpp(positions) == pytest.approx(0.25)


def test_visium_he_mpp_uses_off_tissue_spots_too(tmp_path) -> None:
    """The pitch comes from the whole lattice: one in-tissue spot alone has no neighbour to measure."""
    positions = _write_positions(tmp_path / "tissue_positions.csv", _hex_lattice_px(400.0, n=2))
    frame = pd.read_csv(positions)
    frame.loc[1:, "in_tissue"] = 0
    frame.to_csv(positions, index=False)
    assert visium_he_mpp(positions) == pytest.approx(0.25)


@pytest.mark.parametrize("pitch_px", [20.0, 5000.0])
def test_visium_he_mpp_rejects_an_implausible_pixel_size(tmp_path, pitch_px: float) -> None:
    positions = _write_positions(tmp_path / "tissue_positions.csv", _hex_lattice_px(pitch_px))
    with pytest.raises(ValueError, match="outside"):
        visium_he_mpp(positions)


#: Spot centres in H&E pixels (x, y), on a 400 px (= 100 um at 0.25 um/px) pitch; radius is 110 px.
SPOT_PX = {"AAAC-1": (150, 150), "AAAG-1": (550, 150), "AAAT-1": (150, 550), "OUTS-1": (550, 550)}
#: Space Ranger's own `spot_diameter_fullres`. On real bundles it does not follow from 55 um at the image's pixel size,
#: so the fixture gives one that would imply 0.5 um/px and the tests pin that it is ignored.
SPOT_DIAMETER_FULLRES = 110.0
GENES = ["ACTA2", "CD4", "VIM"]


def _write_10x_h5(path: Path, barcodes: list[str], counts: np.ndarray, feature_types: list[str], names: list[str]):
    """Write the minimum 10x `filtered_feature_bc_matrix.h5` that `scanpy.read_10x_h5` accepts.

    `counts` is `(n_features, n_barcodes)`; 10x stores it CSC over barcodes.
    """
    import h5py
    from scipy.sparse import csc_matrix

    mat = csc_matrix(counts)
    with h5py.File(path, "w") as f:
        grp = f.create_group("matrix")
        grp["barcodes"] = np.array(barcodes, dtype="S")
        grp["data"] = mat.data
        grp["indices"] = mat.indices
        grp["indptr"] = mat.indptr
        grp["shape"] = np.array(mat.shape)
        feats = grp.create_group("features")
        feats["id"] = np.array([f"ID{i}" for i in range(len(names))], dtype="S")
        feats["name"] = np.array(names, dtype="S")
        feats["feature_type"] = np.array(feature_types, dtype="S")
        feats["genome"] = np.array(["GRCh38"] * len(names), dtype="S")
        feats["_all_tag_keys"] = np.array([], dtype="S")


def _make_bundle(root: Path, *, counts_scale: int = 1, off_tissue_in_matrix: bool = False) -> Path:
    """A tiny extracted CytAssist bundle: 3 in-tissue spots, 1 off-tissue, 3 genes + 1 antibody."""
    import tifffile

    bundle = root / STUDY
    (bundle / "spatial").mkdir(parents=True)

    # Dark tissue block on a white slide, so Otsu has two classes to separate.
    he = np.full((700, 700, 3), 245, dtype=np.uint8)
    he[50:650, 50:650] = 90
    tifffile.imwrite(bundle / f"{STUDY}_tissue_image.tif", he, photometric="rgb")

    (bundle / "spatial" / "scalefactors_json.json").write_text(
        json.dumps(
            {
                "spot_diameter_fullres": SPOT_DIAMETER_FULLRES,
                "tissue_hires_scalef": 0.1,
                "tissue_lowres_scalef": 0.03,
            }
        )
    )
    rows = [(barcode, int(barcode != "OUTS-1"), i, i, y, x) for i, (barcode, (x, y)) in enumerate(SPOT_PX.items())]
    pd.DataFrame(
        rows,
        columns=["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"],
    ).to_csv(bundle / "spatial" / "tissue_positions.csv", index=False)

    barcodes = list(SPOT_PX)[:3]  # the filtered matrix holds in-tissue spots only
    counts = counts_scale * np.array([[1, 2, 3], [4, 0, 6], [7, 8, 9], [5, 5, 5]])  # 3 genes + 1 antibody
    if off_tissue_in_matrix:  # a matrix that, unlike Space Ranger's, also carries the off-tissue barcode
        barcodes = list(SPOT_PX)
        counts = np.c_[counts, [1, 1, 1, 1]]
    _write_10x_h5(
        bundle / f"{STUDY}_filtered_feature_bc_matrix.h5",
        barcodes,
        counts,
        ["Gene Expression"] * 3 + ["Antibody Capture"],
        [*GENES, "CD4"],
    )
    return bundle


def _write_nuclei(path: Path) -> Path:
    """Three nuclei in H&E pixels: inside spot AAAC, in the gap beside it, and inside spot AAAG."""

    def square(cx: float, cy: float, half: float = 4.0) -> dict:
        ring = [[cx - half, cy - half], [cx + half, cy - half], [cx + half, cy + half], [cx - half, cy + half]]
        return {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[*ring, ring[0]]]}}

    path.write_text(json.dumps([square(160, 150), square(350, 150), square(545, 155)]))
    return path


def _make_template(path: Path) -> Path:
    import anndata as ad

    var = pd.DataFrame({"gene_ids": ["a", "b", "c", "d", "e"]}, index=["G1", "G2", "G3", "G4", "G5"])
    ad.AnnData(X=np.zeros((2, 5), dtype=np.float32), var=var).write_h5ad(path)
    return path


def test_visium_to_spatialdata_writes_one_global_pixel_frame(tmp_path) -> None:
    """End to end on a synthetic bundle: element set, single coordinate system, tables and spot assignment."""
    import spatialdata as sd
    from spatialdata.transformations import Identity, get_transformation

    bundle = _make_bundle(tmp_path / "raw")
    zarr_path = visium_to_spatialdata(
        bundle,
        tmp_path / "out",
        nucleus_geojson=_write_nuclei(tmp_path / "cells.geojson"),
        template_adata_path=_make_template(tmp_path / "template.h5ad"),
    )
    sdata = sd.read_zarr(zarr_path)

    assert sdata.coordinate_systems == ["global"]  # not the reader's three
    assert set(sdata.images) == {"he_image"}
    assert set(sdata.shapes) == {"spots", "tissue_contours", "nucleus_boundaries"}
    assert set(sdata.tables) == {"spots_table", "table"}
    assert sdata.attrs["spatialdata_io_reader"] == "visium"
    assert sdata.attrs["source_he_mpp"] == pytest.approx(
        0.25
    )  # from the lattice, not the TIFF tag or the scale factors

    # Spots are 55 um circles (110 px at 0.25 um/px) in H&E pixels, three of them (the off-tissue barcode is not in the matrix).
    spots = sdata.shapes["spots"]
    assert len(spots) == 3
    assert spots["radius"].to_numpy() == pytest.approx(110.0)
    assert spots.geometry.x.iloc[0] == pytest.approx(150)
    # Everything shares the pixel grid, so nothing needs a transform to reach `global`.
    for element in (spots, sdata["tissue_contours"], sdata["nucleus_boundaries"], sdata["he_image"]):
        assert isinstance(get_transformation(element, "global"), Identity)

    # The ground truth lives in `spots_table`: genes on `var`, the antibody beside it.
    spots_table = sdata["spots_table"]
    assert spots_table.var_names.to_list() == GENES
    # The antibody is `CD4`, not the reader's `CD4-1`, so it still matches the gene of the same name.
    assert spots_table.obsm["protein_expression"].columns.to_list() == ["CD4"]
    assert spots_table.uns["protein_expression"]["names"] == ["CD4"]
    assert spots_table.uns["protein_expression"]["metric"] == "UMI_COUNTS"
    assert spots_table.uns["spatialdata_attrs"]["region"] == "spots"

    # The nucleus table is all zeros over the template panel and says which spot each nucleus is in.
    table = sdata["table"]
    assert table.var_names.to_list() == ["G1", "G2", "G3", "G4", "G5"]
    assert table.X.max() == 0
    assert table.uns["spatialdata_attrs"]["region"] == "nucleus_boundaries"
    spot_of_nucleus = table.obs["spot_id"].tolist()
    assert spot_of_nucleus[1] is pd.NA  # the one in the gap
    assert spot_of_nucleus[0] == spots.index[0]
    assert spot_of_nucleus[2] == spots.index[1]
    assert table.obsm["spatial"][0] == pytest.approx([160, 150])


def test_visium_to_spatialdata_without_nuclei_writes_only_the_spot_side(tmp_path) -> None:
    import spatialdata as sd

    zarr_path = visium_to_spatialdata(_make_bundle(tmp_path / "raw"), tmp_path / "out")
    sdata = sd.read_zarr(zarr_path)
    assert set(sdata.shapes) == {"spots", "tissue_contours"}
    assert set(sdata.tables) == {"spots_table"}


def test_visium_to_spatialdata_refuses_an_all_zero_spot_table(tmp_path) -> None:
    """A silent bad sample is worse than a failed run."""
    bundle = _make_bundle(tmp_path / "raw", counts_scale=0)
    with pytest.raises(ValueError, match="all zeros"):
        visium_to_spatialdata(bundle, tmp_path / "out")
    assert not (tmp_path / "out" / f"{STUDY}.zarr").exists()


def test_visium_to_spatialdata_rejects_nuclei_on_the_wrong_pixel_grid(tmp_path) -> None:
    """Nuclei segmented at a different mpp land far from every spot; that must raise, not write an empty join."""
    far = tmp_path / "far.geojson"
    far.write_text(
        json.dumps(
            [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[5000, 5000], [5008, 5000], [5008, 5008], [5000, 5000]]],
                    },
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="falls inside any spot"):
        visium_to_spatialdata(
            _make_bundle(tmp_path / "raw"),
            tmp_path / "out",
            nucleus_geojson=far,
            template_adata_path=_make_template(tmp_path / "template.h5ad"),
        )


def test_visium_to_spatialdata_names_the_missing_spatial_archive(tmp_path) -> None:
    bundle = _make_bundle(tmp_path / "raw")
    (bundle / "spatial" / "scalefactors_json.json").unlink()
    with pytest.raises(FileNotFoundError, match="scalefactors"):
        visium_to_spatialdata(bundle, tmp_path / "out")


def test_visium_converter_supports_an_extracted_bundle_only(tmp_path) -> None:
    bundle = _make_bundle(tmp_path / "raw")
    assert VisiumConverter.supports(bundle)
    assert not VisiumConverter.supports(tmp_path)


def test_visium_to_spatialdata_needs_a_template_for_nuclei(tmp_path) -> None:
    with pytest.raises(ValueError, match="template_adata_path"):
        visium_to_spatialdata(
            _make_bundle(tmp_path / "raw"),
            tmp_path / "out",
            nucleus_geojson=_write_nuclei(tmp_path / "cells.geojson"),
            template_adata_path=None,
        )


def test_visium_to_spatialdata_rejects_a_single_level_slide_too_wide_to_segment(tmp_path) -> None:
    """A raw 10x image has no coarse level, so `segment_tissue` would contour the full plane and never finish.

    The error has to name the conversion, with the scale-factor pixel size rather than the bogus TIFF tag.
    """
    import tifffile

    bundle = _make_bundle(tmp_path / "raw")
    wide = np.full((64, 9000, 3), 245, dtype=np.uint8)  # one level, wider than MAX_SEGMENTATION_WIDTH
    tifffile.imwrite(bundle / f"{STUDY}_tissue_image.tif", wide, photometric="rgb")

    with pytest.raises(ValueError, match=r"convert_to_ometiff\.py.*--mpp 0\.2500"):
        visium_to_spatialdata(bundle, tmp_path / "out")
    assert not (tmp_path / "out" / f"{STUDY}.zarr").exists()


def test_visium_to_spatialdata_rejects_spots_outside_the_image(tmp_path) -> None:
    """In-tissue spots beyond the image mean the positions are on another pixel grid; that must not be written."""
    import tifffile

    bundle = _make_bundle(tmp_path / "raw")
    small = np.full((300, 300, 3), 245, dtype=np.uint8)  # the spots reach x=550, y=550
    tifffile.imwrite(bundle / f"{STUDY}_tissue_image.tif", small, photometric="rgb")

    with pytest.raises(ValueError, match="not on this image's pixel grid"):
        visium_to_spatialdata(bundle, tmp_path / "out")
    assert not (tmp_path / "out" / f"{STUDY}.zarr").exists()


def test_visium_to_spatialdata_keeps_a_few_spots_outside_the_image(tmp_path, monkeypatch, caplog) -> None:
    """A small share of in-tissue spots past the image edge is kept, with a warning; a large share still raises."""
    import spatialdata as sd
    import tifffile

    from spatialrefinery.io import visium

    bundle = _make_bundle(tmp_path / "raw")
    narrow = np.full((700, 400, 3), 245, dtype=np.uint8)  # spot AAAG sits at x=550, beyond 400 + one 110 px radius
    narrow[50:650, 50:350] = 90
    tifffile.imwrite(bundle / f"{STUDY}_tissue_image.tif", narrow, photometric="rgb")

    with pytest.raises(ValueError, match="1 of 3 in-tissue spots"):  # 33% outside, over the default tolerance
        visium_to_spatialdata(bundle, tmp_path / "strict")

    monkeypatch.setattr(visium, "MAX_SPOTS_OUTSIDE_IMAGE", 0.5)
    zarr_path = visium_to_spatialdata(bundle, tmp_path / "relaxed")

    assert "1 of 3 in-tissue spots lie more than one spot radius outside" in caplog.text
    assert len(sd.read_zarr(zarr_path)["spots"]) == 3  # the stray spot is kept, not dropped


def test_visium_to_spatialdata_keeps_only_in_tissue_spots(tmp_path) -> None:
    """A matrix that carries an in_tissue == 0 barcode must not put that spot in the store."""
    import spatialdata as sd

    bundle = _make_bundle(tmp_path / "raw", off_tissue_in_matrix=True)
    sdata = sd.read_zarr(visium_to_spatialdata(bundle, tmp_path / "out"))

    assert len(sdata["spots"]) == 3
    assert sdata["spots_table"].n_obs == 3
    assert (sdata["spots_table"].obs["in_tissue"] == 1).all()
    assert "OUTS-1" not in sdata["spots_table"].obs_names
