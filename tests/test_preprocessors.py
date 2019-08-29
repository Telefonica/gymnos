import gymnos
import pytest


def test_load():
    preprocessor = gymnos.preprocessors.load("replace", from_val=1.0, to_val=2.0)

    assert isinstance(preprocessor, gymnos.preprocessors.preprocessor.Preprocessor)

    with pytest.raises(ValueError):
        _ = gymnos.preprocessors.load("dummy")
