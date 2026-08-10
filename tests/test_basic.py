import spatialrefinery


def test_package_is_importable():
    assert spatialrefinery.__version__


def test_documented_functions_are_importable_from_top_level():
    """The docs (README, index.md, tutorials) import these four names directly
    off `spatialrefinery`; this guards against a future refactor silently
    breaking every code block on the docs site."""
    from spatialrefinery.core.converter import convert_to_ometiff
    from spatialrefinery.io.xenium import (
        download_xenium_study,
        xenium_to_spatialdata,
        xenium_to_spatialdata_zip,
    )

    assert spatialrefinery.convert_to_ometiff is convert_to_ometiff
    assert spatialrefinery.download_xenium_study is download_xenium_study
    assert spatialrefinery.xenium_to_spatialdata is xenium_to_spatialdata
    assert spatialrefinery.xenium_to_spatialdata_zip is xenium_to_spatialdata_zip
