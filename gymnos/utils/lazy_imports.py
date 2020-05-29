#
#
#  Lazy imports for heavy dependencies
#
#

import os
import sys
import site
import GPUtil
import logging
import importlib
import subprocess


from .py_utils import classproperty

logger = logging.getLogger(__name__)


def _is_venv():
    return (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))


def _install_module_with_pip_module(module_name):
    try:
        from pip import main as pipmain
    except ImportError:
        from pip._internal import main as pipmain

    pipmain(["install", module_name])


def _install_module_with_subprocess(module_name):
    pip_args = []
    if not _is_venv() and site.ENABLE_USER_SITE:
        pip_args.append("--user")
        if not os.path.exists(site.USER_SITE):
            os.makedirs(site.USER_SITE)
            site.addsitedir(site.USER_SITE)
    cmd = [sys.executable, "-m", "pip", "install"] + pip_args + [module_name]
    return subprocess.call(cmd, env=os.environ.copy())


def _install_module(module_name):
    _install_module_with_subprocess(module_name)


def _try_import_and_autoinstall(module_to_import, module_to_install):
    try:
        return importlib.import_module(module_to_import)
    except ImportError:
        warning_msg = (
            'Tried importing {importing} but failed. The module you are trying to use may have additional dependencies.\n'  # noqa: E501
            'Gymnos-Autoinstall is activated so we will we try to install library "{installing}" automatically.\n'
            'Note that Gymnos-Autoinstall is currently in beta and that you may encounter issues using this functionality.\n'  # noqa: E501
            'If Gymnos-Autoinstall fails, please install manually "{installing}".\n'
            'To disable Gymnos-Autoinstall, set environment variable "GYMNOS_AUTOINSTALL=0".'
        )
        logger.warning(warning_msg.format(importing=module_to_import, installing=module_to_install))

        _install_module_with_subprocess(module_to_install)

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
    def tensorboard(cls):
        return _try_import("tensorboard")

    @classproperty
    def PIL(cls):
        PIL = _try_import("PIL", module_to_install="Pillow")
        return __import__("{}.Image".format(PIL.__name__))  # import common module

    @classproperty
    def requests(cls):
        return _try_import("requests")

    @classproperty
    def kaggle(cls):
        return _try_import("kaggle")

    @classproperty
    def smb(cls):
        return _try_import("smb", module_to_install="pysmb")

    @classproperty
    def scipy(cls):
        return _try_import("scipy")

    @classproperty
    def tensorflow(cls):
        has_gpu = bool(GPUtil.getAvailable())
        if has_gpu:
            return _try_import("tensorflow", module_to_install="tensorflow-gpu>=1.9.0,<2.0")
        else:
            return _try_import("tensorflow", module_to_install="tensorflow>=1.9.0,<2.0")

    @classproperty
    def sklearn(cls):
        return _try_import("sklearn", module_to_install="scikit-learn")

    @classproperty
    def torch(cls):
        return _try_import("torch")

    @classproperty
    def torchvision(cls):
        return _try_import("torchvision")

    @classproperty
    def dummy(cls):
        """
        Only for testing purposes
        """
        return _try_import("dummy")


lazy_imports = LazyImporter
