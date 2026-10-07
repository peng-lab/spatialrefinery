"""Structural sanity check for SpatialData zarr stores produced by spatialrefinery.

Two writers feed the same downstream code and must agree on layout:

* Xenium bundles (`slurm/submit_convert_panels.sh`, `spatialrefinery.io.xenium`)
* H&E-only slides (`slurm/segment_slurm.sh`, `spatialrefinery.segmentation.to_spatialdata`)

For each store this script verifies

1. the root attrs `spatialrefinery.core.utils.sample_attrs` writes: a known `spatialdata_io_reader`, a
   `spatialdata_io_software_version`, a finite positive `source_mpp`, and `source_he_mpp`, which must be
   finite and positive when `he_image` is present and `None` otherwise,
2. the minimum element set of the profile (Xenium or H&E-only),
3. every spatial element maps into the `global` coordinate system,
4. transformation types: images may only be Identity/Affine, shapes may be
   Identity/Affine/Scale (labels and points are unrestricted),
5. `he_image`'s transformation equals `tissue_contours`' once Scale components
   are dropped (images can never carry a scale),
6. every table annotates the expected shapes element, with matching `region`
   labels and instance ids.

This is a standalone CLI, not a pytest module (the filename deliberately does
not match `test_*.py`):

    python tests/sanity_check_spatialdata.py a.zarr,b.zarr [--profile auto|xenium|he]

Exit status is 1 if any store fails a check or cannot be read.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import spatialdata as sd
from spatialdata.models import get_axes_names
from spatialdata.transformations import Affine, BaseTransformation, Identity, Scale, Sequence, get_transformation

GLOBAL_CS = "global"
READERS = frozenset({"xenium", "visium", "he"})
XY_AXES = ("x", "y")


@dataclass(frozen=True)
class Profile:
    """Minimum structure a store must satisfy.

    Attributes
    ----------
    name
        Profile label used in reports.
    required_images, required_labels, required_points, required_shapes
        Element names that must exist. Anything else present is reported as info.
    table_regions
        Required table name -> shapes elements it is allowed to annotate.
    optional_images
        Image names that may exist (they are still subject to the image rules).
    zero_table
        Whether the main `table` must be an all-zero mock template (H&E-only).
    """

    name: str
    required_images: frozenset[str]
    required_labels: frozenset[str]
    required_points: frozenset[str]
    required_shapes: frozenset[str]
    table_regions: dict[str, frozenset[str]]
    optional_images: frozenset[str] = frozenset()
    zero_table: bool = False


XENIUM = Profile(
    name="xenium",
    required_images=frozenset({"he_image"}),
    optional_images=frozenset({"morphology_mip", "morphology_focus"}),
    required_labels=frozenset({"cell_labels", "nucleus_labels"}),
    required_points=frozenset({"transcripts"}),
    required_shapes=frozenset({"cell_boundaries", "nucleus_boundaries", "spots_55um", "spots_100um", "tissue_contours"}),
    table_regions={
        "table": frozenset({"cell_boundaries", "nucleus_boundaries"}),
        "spots_55um_table": frozenset({"spots_55um"}),
        "spots_100um_table": frozenset({"spots_100um"}),
    },
)

HE_ONLY = Profile(
    name="he",
    required_images=frozenset({"he_image"}),
    required_labels=frozenset(),
    required_points=frozenset(),
    required_shapes=frozenset({"nucleus_boundaries"}),
    table_regions={"table": frozenset({"nucleus_boundaries"})},
    zero_table=True,
)

PROFILES = {"xenium": XENIUM, "he": HE_ONLY}


@dataclass
class CheckResult:
    """Outcome of one check; `info` results never fail a store."""

    name: str
    passed: bool
    detail: str = ""
    info: bool = field(default=False, repr=False)


def flatten_transformation(transformation: BaseTransformation) -> list[BaseTransformation]:
    """Return the leaf components of a (possibly nested) `Sequence`."""
    if isinstance(transformation, Sequence):
        return [leaf for t in transformation.transformations for leaf in flatten_transformation(t)]
    return [transformation]


class SpatialDataSanityChecker:
    """Run all structural checks of one `Profile` against one zarr store."""

    def __init__(self, path: str | Path, profile: Profile | None = None) -> None:
        self.path = Path(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.sdata = sd.read_zarr(self.path)
        self.profile = profile or self.detect_profile(self.sdata)

    @staticmethod
    def detect_profile(sdata: sd.SpatialData) -> Profile:
        """Xenium if the reader attribute or a `transcripts` points element is present."""
        is_xenium = sdata.attrs.get("spatialdata_io_reader") == "xenium" or "transcripts" in sdata.points
        return XENIUM if is_xenium else HE_ONLY

    def element_kinds(self) -> dict[str, str]:
        """Map every spatial element name to its kind (`images`, `labels`, `points`, `shapes`)."""
        return {name: kind for kind, name, _ in self.sdata.gen_spatial_elements()}

    def global_components(self, name: str) -> list[BaseTransformation]:
        """Leaf transformations mapping element `name` into `global`."""
        element = self.sdata[name]
        return flatten_transformation(get_transformation(element, to_coordinate_system=GLOBAL_CS))

    def run(self) -> list[CheckResult]:
        results = [
            self.check_root_attrs(),
            self.check_elements(),
            self.check_global_cs(),
            self.check_transform_types(),
            self.check_he_contour_consistency(),
            *self.check_tables(),
        ]
        extras = sorted(
            (set(self.element_kinds()) | set(self.sdata.tables))
            - self.required_names()
            - self.profile.optional_images
            - set(self.profile.table_regions)
        )
        results.append(CheckResult("extra_elements", True, ", ".join(extras) or "none", info=True))
        return results

    def required_names(self) -> set[str]:
        p = self.profile
        return set(p.required_images | p.required_labels | p.required_points | p.required_shapes)

    # ------------------------------------------------------------------ checks

    def check_root_attrs(self) -> CheckResult:
        attrs = self.sdata.attrs
        problems = []
        if attrs.get("spatialdata_io_reader") not in READERS:
            problems.append(f"spatialdata_io_reader={attrs.get('spatialdata_io_reader')!r} not in {sorted(READERS)}")
        if not attrs.get("spatialdata_io_software_version"):
            problems.append("missing spatialdata_io_software_version")
        has_he = "he_image" in self.sdata.images
        for key in ("source_mpp", "source_he_mpp"):
            if key not in attrs:
                problems.append(f"missing {key}")
            elif key == "source_he_mpp" and not has_he:
                if attrs[key] is not None:
                    problems.append(f"{key}={attrs[key]!r} but the store has no he_image (expected None)")
            else:
                try:
                    value = float(attrs[key])
                except (TypeError, ValueError):
                    problems.append(f"{key}={attrs[key]!r} (not numeric)")
                    continue
                if not (math.isfinite(value) and value > 0):
                    problems.append(f"{key}={value} (must be finite and > 0)")
        detail = "; ".join(problems) or ", ".join(
            f"{k}={attrs[k]}" for k in ("spatialdata_io_reader", "source_mpp", "source_he_mpp")
        )
        return CheckResult("root_attrs", not problems, detail)

    def check_elements(self) -> CheckResult:
        p, s = self.profile, self.sdata
        missing = [
            f"{kind}/{name}"
            for kind, names, present in (
                ("images", p.required_images, s.images),
                ("labels", p.required_labels, s.labels),
                ("points", p.required_points, s.points),
                ("shapes", p.required_shapes, s.shapes),
                ("tables", frozenset(p.table_regions), s.tables),
            )
            for name in sorted(names)
            if name not in present
        ]
        problems = [f"missing {m}" for m in missing]

        if "he_image" in s.images:
            axes = tuple(get_axes_names(s.images["he_image"]))
            n_channels = s.images["he_image"]["scale0"]["image"].sizes["c"]
            if axes != ("c", "y", "x"):
                problems.append(f"he_image axes {axes} != ('c','y','x')")
            if n_channels != 3:
                problems.append(f"he_image has {n_channels} channels, expected 3")
        for name in p.required_labels & set(s.labels):
            axes = tuple(get_axes_names(s.labels[name]))
            if axes != ("y", "x"):
                problems.append(f"{name} axes {axes} != ('y','x')")

        present_optional = sorted(p.optional_images & set(s.images))
        detail = "; ".join(problems) if problems else f"all required present (optional images: {present_optional or 'none'})"
        return CheckResult("elements", not problems, detail)

    def check_global_cs(self) -> CheckResult:
        problems = []
        if GLOBAL_CS not in self.sdata.coordinate_systems:
            problems.append(f"'{GLOBAL_CS}' not in coordinate systems {self.sdata.coordinate_systems}")
        for name in self.element_kinds():
            if GLOBAL_CS not in get_transformation(self.sdata[name], get_all=True):
                problems.append(f"{name} has no transformation to '{GLOBAL_CS}'")
        return CheckResult("global_coordinate_system", not problems, "; ".join(problems) or "all elements map to global")

    def check_transform_types(self) -> CheckResult:
        allowed = {"images": (Identity, Affine), "shapes": (Identity, Affine, Scale)}
        problems, seen = [], []
        for name, kind in self.element_kinds().items():
            if GLOBAL_CS not in get_transformation(self.sdata[name], get_all=True):
                continue  # reported by check_global_cs
            components = self.global_components(name)
            seen.append(f"{name}:{'+'.join(type(c).__name__ for c in components)}")
            if kind in allowed:
                illegal = [type(c).__name__ for c in components if not isinstance(c, allowed[kind])]
                if illegal:
                    problems.append(f"{kind[:-1]} '{name}' has illegal transformation(s) {illegal}")
        return CheckResult("transform_types", not problems, "; ".join(problems) or " ".join(seen))

    def check_he_contour_consistency(self) -> CheckResult:
        if "tissue_contours" not in self.sdata.shapes or "he_image" not in self.sdata.images:
            return CheckResult("he_vs_tissue_contours", True, "skipped (needs both he_image and tissue_contours)", info=True)
        contour_components = [c for c in self.global_components("tissue_contours") if not isinstance(c, Scale)]
        he_components = self.global_components("he_image")
        contour_xy = self.xy_matrix(contour_components)
        he_xy = self.xy_matrix(he_components)
        if np.allclose(contour_xy, he_xy, rtol=1e-6, atol=1e-6):
            return CheckResult("he_vs_tissue_contours", True, "he_image == tissue_contours minus Scale")
        return CheckResult(
            "he_vs_tissue_contours",
            False,
            f"he_image xy matrix\n{he_xy}\n!= tissue_contours (Scale dropped)\n{contour_xy}",
        )

    @staticmethod
    def xy_matrix(components: list[BaseTransformation]) -> np.ndarray:
        """Compose leaf transformations into a 3x3 matrix on the (x, y) plane."""
        sequence = components[0] if len(components) == 1 else Sequence(components)
        return sequence.to_affine_matrix(input_axes=XY_AXES, output_axes=XY_AXES)

    def check_tables(self) -> list[CheckResult]:
        results = []
        for name, table in self.sdata.tables.items():
            check = f"table[{name}]"
            attrs = table.uns.get("spatialdata_attrs")
            if not attrs:
                results.append(CheckResult(check, False, "no uns['spatialdata_attrs'] (table is not linked to any element)"))
                continue
            region, region_key, instance_key = attrs["region"], attrs["region_key"], attrs["instance_key"]
            regions = [region] if isinstance(region, str) else list(region)
            problems = []

            allowed = self.profile.table_regions.get(name)
            if allowed is not None:
                bad = [r for r in regions if r not in allowed]
                if bad:
                    problems.append(f"annotates {bad}, expected one of {sorted(allowed)}")
            for r in regions:
                if r not in self.sdata.shapes:
                    problems.append(f"annotated region '{r}' is not a shapes element")
            if len(regions) == 1 and regions[0] in self.sdata.shapes:
                shapes = self.sdata.shapes[regions[0]]
                labels = set(table.obs[region_key].astype(str))
                if labels != {regions[0]}:
                    problems.append(f"obs['{region_key}'] values {sorted(labels)} != ['{regions[0]}']")
                if table.n_obs != len(shapes):
                    problems.append(f"n_obs {table.n_obs} != {len(shapes)} shapes in '{regions[0]}'")
                elif set(table.obs[instance_key].astype(str)) != set(shapes.index.astype(str)):
                    problems.append(f"obs['{instance_key}'] does not match '{regions[0]}' index")
            if self.profile.zero_table and name == "table" and table.X is not None and table.X.max() != 0:
                problems.append("table.X is not all zeros (expected mock template)")

            results.append(CheckResult(check, not problems, "; ".join(problems) or f"-> {regions} ({table.n_obs} x {table.n_vars})"))
        return results


def parse_paths(arg: str) -> list[Path]:
    return [Path(p.strip()) for p in arg.split(",") if p.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stores", help="Comma-separated paths to SpatialData .zarr stores.")
    parser.add_argument("--profile", choices=["auto", *PROFILES], default="auto", help="Structure to enforce (default: auto-detect).")
    args = parser.parse_args(argv)

    failed = []
    for path in parse_paths(args.stores):
        print(f"\n=== {path}")
        try:
            checker = SpatialDataSanityChecker(path, None if args.profile == "auto" else PROFILES[args.profile])
            results = checker.run()
        except Exception as exc:  # noqa: BLE001 - a bad store must not abort the batch
            print(f"  [FAIL] load: {type(exc).__name__}: {exc}")
            failed.append(path)
            continue
        print(f"  profile: {checker.profile.name}")
        for r in results:
            tag = "INFO" if r.info else ("PASS" if r.passed else "FAIL")
            print(f"  [{tag}] {r.name}: {r.detail}")
        if not all(r.passed for r in results):
            failed.append(path)

    total = len(parse_paths(args.stores))
    print(f"\n{total - len(failed)}/{total} stores passed")
    for path in failed:
        print(f"  FAILED: {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
