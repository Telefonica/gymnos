#
#
#  Lazy imports for heavy dependencies
#
#

import os
import sys
import subprocess
import logging
import importlib

from .py_utils import classproperty

logger = logging.getLogger(__name__)


def _install_module_with_pip_module(module_name):
    try:
        from pip import main as pipmain
    except ImportError:
        from pip._internal import main as pipmain

    pipmain(["install", "--no-cache-dir", module_name])


def _install_module_with_subprocess(module_name):
    pip_args = ["--no-cache-dir"]
    cmd = [sys.executable, "-m", "pip", "install"] + pip_args + [module_name]
    return subprocess.call(cmd, env=os.environ.copy())


def _install_module(module_name):
    _install_module_with_subprocess(module_name)


def _try_import_and_autoinstall(module_to_import, module_to_install):
    try:
        return importlib.import_module(module_to_import)
    except ImportError:
        warning_msg = (
            'Tried importing {importing} but failed. The module you are trying to use may have additional dependencies.\n' # noqa: E501
            'Gymnos-Autoinstall is activated so we will we try to install library "{installing}" automatically.\n'
            'Note that Gymnos-Autoinstall is currently in beta and that you may encounter issues using this functionality.\n' # noqa: E501
            'If Gymnos-Autoinstall fails, please install manually "{installing}".\n'
            'To disable Gymnos-Autoinstall, set environment variable "GYMNOS_AUTOINSTALL=0".'
        )
        logger.warning(warning_msg.format(importing=module_to_import, installing=module_to_install))

        _install_module_with_subprocess(module_to_install)
    finally:
        return importlib.import_module(module_to_import)


def _try_import_and_customize_error(module_to_import, module_to_install):
    try:
        return importlib.import_module(module_to_import)
    except ImportError as e:
        error_msg = ("Tried importing {} but failed. The module you are trying to use may have additional "
                     "dependencies. Please install {} manually").format(module_to_import, module_to_install)
        raise type(e)(str(e) + "\n" + error_msg)


def _try_import(module_to_import, module_to_install=None, autoinstall=None):
    if autoinstall is None:
        autoinstall = bool(os.getenv("GYMNOS_AUTOINSTALL", "1") == "1")

    if module_to_install is None:
        module_to_install = module_to_import

    if autoinstall:
        return _try_import_and_autoinstall(module_to_import, module_to_install)
    else:
        return _try_import_and_customize_error(module_to_import, module_to_install)


class LazyImporter:

    @classproperty
    def spacy(cls):
        return _try_import("spacy")

    @classproperty
    def comet_ml(cls):
        return _try_import("comet_ml", module_to_install="comet-ml")

    @classproperty
    def statsmodels_api(cls):
        _try_import("statsmodels")
        return importlib.import_module("statsmodels.api")

    @classproperty
    def statsmodels_tsa(cls):
        _try_import("statsmodels")
        return importlib.import_module("statsmodels.tsa")

    @classproperty
    def mlflow(cls):
        return _try_import("mlflow")

    @classproperty
    def dummy(cls):
        """
        Only for testing purposes
        """
        return _try_import("dummy")


lazy_imports = LazyImporter