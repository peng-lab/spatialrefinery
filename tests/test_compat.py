"""Tests for the InstanSeg compatibility bridges.

InstanSeg's whole-slide path carries three defects that break it in this
environment: it is written against zarr 2 (and pins `zarr<3`, which conflicts
with spatialdata), it calls `TiffSlide` without importing it, and it emits
invalid JSON. These tests pin each bridge, so a future release that changes
any of them fails here rather than halfway through a multi-minute whole-slide
run.

`instanseg` and `tiffslide` come from the optional `segmentation` extra, which
the CI test environment does not install, so the tests that need them skip
rather than fail. The zarr and GeoJSON bridges are exercised unconditionally.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import zarr

from spatialrefinery.segmentation._compat import (
    patch_instanseg,
    patch_instanseg_tiffslide,
    patch_zarr_directory_store,
    repair_geojson_trailing_comma,
)

# --------------------------------------------------------------------- #
# 1. zarr 3
# --------------------------------------------------------------------- #


def test_patch_is_idempotent():
    """Calling the patch twice must not raise, and must report no-op the second time."""
    patch_zarr_directory_store()
    assert patch_zarr_directory_store() is False
    assert hasattr(zarr, "DirectoryStore")


def test_directory_store_round_trip(tmp_path):
    """Exercise exactly what InstanSeg does: create, slice-assign, reopen, read."""
    patch_zarr_directory_store()
    path = tmp_path / "canvas.zarr"

    # 1. construct a store from a path
    store = zarr.DirectoryStore(path)

    # 2. allocate a chunked int32 canvas
    canvas = zarr.zeros((1, 600, 600), chunks=(1, 512, 512), dtype=np.int32, store=store, overwrite=True)
    assert canvas.shape == (1, 600, 600)

    # 3. slice-assign a tile, the way tiles are stitched into the canvas
    tile = np.arange(512 * 512, dtype=np.int32).reshape(512, 512)
    canvas[0, 0:512, 0:512] = tile

    # 4. reopen read-only, as the GeoJSON export does
    reopened = zarr.open(str(path), mode="r")
    assert reopened.shape == (1, 600, 600)
    np.testing.assert_array_equal(np.asarray(reopened[0, 0:512, 0:512]), tile)


def test_untouched_region_stays_zero(tmp_path):
    """Labels must not leak outside the written window."""
    patch_zarr_directory_store()
    store = zarr.DirectoryStore(tmp_path / "c.zarr")
    canvas = zarr.zeros((1, 64, 64), chunks=(1, 32, 32), dtype=np.int32, store=store, overwrite=True)
    canvas[0, 0:16, 0:16] = 7
    assert int(np.asarray(canvas[0, 32:64, 32:64]).sum()) == 0


# --------------------------------------------------------------------- #
# 2. missing TiffSlide import
# --------------------------------------------------------------------- #


def test_tiffslide_is_injected_into_instanseg():
    """`InstanSeg.read_slide` calls `TiffSlide(...)` without importing it.

    Every import of the name in `inference_class` is function-local, so the
    module global is unbound and any whole-slide call dies with `NameError`
    before reading a single tile. Guard the injection that fixes it.
    """
    inference_class = pytest.importorskip("instanseg.inference_class")
    TiffSlide = pytest.importorskip("tiffslide").TiffSlide

    patch_instanseg_tiffslide()
    assert inference_class.TiffSlide is TiffSlide

    # Idempotent: a second call reports "already bound" rather than rebinding.
    assert patch_instanseg_tiffslide() is False


def test_patch_instanseg_applies_both_bridges():
    """The convenience entry point must leave both import-time defects patched."""
    inference_class = pytest.importorskip("instanseg.inference_class")

    patch_instanseg()
    assert hasattr(zarr, "DirectoryStore")
    assert getattr(inference_class, "TiffSlide", None) is not None


# --------------------------------------------------------------------- #
# 3. invalid GeoJSON
# --------------------------------------------------------------------- #


def test_repair_geojson_trailing_comma(tmp_path):
    """InstanSeg closes its feature array as `...}},\\n]`, which is invalid JSON."""
    p = tmp_path / "cells.geojson"
    p.write_text('[\n{"a": 1},\n{"a": 2},\n]')

    with pytest.raises(json.JSONDecodeError):
        json.loads(p.read_text())

    assert repair_geojson_trailing_comma(p) is True
    assert json.loads(p.read_text()) == [{"a": 1}, {"a": 2}]


def test_repair_is_noop_on_valid_geojson(tmp_path):
    """A well-formed file must be left byte-for-byte alone."""
    p = tmp_path / "cells.geojson"
    original = '[\n{"a": 1}\n]'
    p.write_text(original)

    assert repair_geojson_trailing_comma(p) is False
    assert p.read_text() == original


def test_repair_survives_multibyte_tail(tmp_path):
    """The 4 KiB window can start mid-UTF-8, so the scan must work on bytes.

    Decoding the window would raise on a split sequence, and character offsets
    would not line up with the byte offsets used to seek back into the file.
    """
    p = tmp_path / "cells.geojson"
    padding = "µ" * 3000  # two bytes each, so the window boundary splits one
    p.write_text(f'[\n{{"name": "{padding}"}},\n]', encoding="utf-8")

    assert repair_geojson_trailing_comma(p) is True
    assert json.loads(p.read_text(encoding="utf-8")) == [{"name": padding}]
