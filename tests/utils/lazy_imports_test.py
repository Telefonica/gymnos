#
#
#   Lazy imports test
#
#

import os
import pytest
import subprocess

from gymnos.utils.lazy_imports import lazy_imports


def test_lazy_import_error(mocker):
    os.environ["GYMNOS_AUTOINSTALL"] = "0"

    def raise_import_error(*args, **kwargs):
        raise ImportError

    mocker.patch("importlib.import_module", raise_import_error)
    mocker.patch("subprocess.call")

    with pytest.raises(ImportError):
        _ = lazy_imports.dummy

    assert subprocess.call.call_count == 0


def test_lazy_import_autoinstall(mocker):
    os.environ["GYMNOS_AUTOINSTALL"] = "1"

    def raise_import_error_when_no_subprocess(*args, **kwargs):
        if subprocess.call.call_count == 0:
            raise ImportError
        else:
            return "dummy"

    mocker.patch("importlib.import_module", raise_import_error_when_no_subprocess)
    mocker.patch("subprocess.call")

    assert lazy_imports.dummy == "dummy"
    assert subprocess.call.call_count == 1
    main_call_args = subprocess.call.call_args_list[0][0][0]
    assert all([x in main_call_args for x in ["pip", "install", "dummy"]])
