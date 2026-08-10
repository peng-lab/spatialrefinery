"""Tests for `spatialrefinery.core.registry`.

The registry is deliberately stdlib-only (see its module docstring), so
these tests run fast and offline. Each test operates on fresh, empty
registries (via the `_isolated_registry` fixture) rather than the real
built-in registrations, so they don't depend on -- or interfere with --
whatever technologies/converters happen to be registered elsewhere.
"""

from __future__ import annotations

import pytest

from spatialrefinery.core import registry


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, "_TECHNOLOGIES", {})
    monkeypatch.setattr(registry, "_ALIASES", {})
    monkeypatch.setattr(registry, "_CONVERTERS", {})
    # Pretend built-ins are already loaded so `_ensure_builtins` (called by
    # `get_technology`/`list_technologies`/`get_converter_for`/`list_converters`)
    # is a no-op and never imports `spatialrefinery.core.converter`/`.io`.
    monkeypatch.setattr(registry, "_BUILTINS_LOADED", True)


def test_register_and_get_technology_by_alias_normalised() -> None:
    spec = registry.TechnologySpec(name="My Tech", aliases=("mt",))
    registry.register_technology(spec)

    assert registry.get_technology("my_tech") is spec
    assert registry.get_technology("MY-TECH") is spec  # case/dash/space-insensitive
    assert registry.get_technology("mt") is spec  # via alias
    assert registry.get_technology("MT") is spec
    assert registry.list_technologies() == ["my_tech"]


def test_register_technology_duplicate_requires_overwrite() -> None:
    registry.register_technology(registry.TechnologySpec(name="dup"))

    with pytest.raises(registry.RegistryError):
        registry.register_technology(registry.TechnologySpec(name="dup"))

    # Should not raise: this is the pattern `io/xenium.py` relies on so the
    # module can be re-imported under pytest's `--import-mode=importlib`.
    registry.register_technology(registry.TechnologySpec(name="dup"), overwrite=True)


def test_unregister_technology_removes_aliases_too() -> None:
    registry.register_technology(registry.TechnologySpec(name="foo", aliases=("f",)))
    registry.unregister_technology("foo")

    with pytest.raises(registry.RegistryError):
        registry.get_technology("foo")
    with pytest.raises(registry.RegistryError):
        registry.get_technology("f")


def test_unregister_unknown_technology_is_a_no_op() -> None:
    registry.unregister_technology("does-not-exist")  # must not raise


def test_get_technology_unknown_error_lists_known_names() -> None:
    registry.register_technology(registry.TechnologySpec(name="xenium"))
    with pytest.raises(registry.RegistryError, match="xenium"):
        registry.get_technology("visium")


def test_register_converter_duplicate_suffix_requires_overwrite() -> None:
    class ConverterA:
        pass

    class ConverterB:
        pass

    registry.register_converter(ConverterA, suffixes=(".xyz",))
    with pytest.raises(registry.RegistryError):
        registry.register_converter(ConverterB, suffixes=(".xyz",))

    registry.register_converter(ConverterB, suffixes=(".xyz",), overwrite=True)
    assert registry.get_converter_for("sample.xyz") is ConverterB


def test_register_converter_bare_decorator_uses_input_suffixes() -> None:
    @registry.register_converter
    class Converter:
        input_suffixes = (".foo", ".bar")

    assert registry.get_converter_for("a.foo") is Converter
    assert registry.get_converter_for("a.bar") is Converter
    assert registry.list_converters() == {".bar": "Converter", ".foo": "Converter"}


def test_get_converter_for_unknown_suffix_raises() -> None:
    with pytest.raises(registry.RegistryError, match="No converter registered"):
        registry.get_converter_for("file.totally_unknown_suffix")


def test_get_converter_for_rejects_ome_tif_as_input() -> None:
    """`.ome.tif` is our own output convention, not a vendor input format.

    Regression test: `.ome.tif`'s final suffix (`.tif`) matches
    `OpenSlideImageConverter`'s registered suffixes, so without this guard a
    second conversion run over a directory would re-ingest its own output.
    """

    class Converter:
        pass

    registry.register_converter(Converter, suffixes=(".tif", ".tiff"))

    assert registry.get_converter_for("slide.tif") is Converter
    with pytest.raises(registry.RegistryError, match="converter output"):
        registry.get_converter_for("slide.ome.tif")
    with pytest.raises(registry.RegistryError, match="converter output"):
        registry.get_converter_for("SLIDE.OME.TIFF")
