#
#
#   Command utils
#
#

import os
import ast
import sys
import glob
import json
import pkgutil
import pathlib
import warnings
import importlib
import rich.tree
import subprocess
import rich.syntax
import pkg_resources

from ..base import BasePredictor
from ..utils.py_utils import remove_suffix

from rich.text import Text
from rich.panel import Panel
from rich.markup import escape
from rich.filesize import decimal
from lazy_object_proxy import Proxy
from typing import Sequence, Optional
from omegaconf import DictConfig, OmegaConf, ListConfig


def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "dataset",
        "test",
        "mlflow",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    tree = rich.tree.Tree(":gear: CONFIG")

    for field in fields:
        config_section = config.get(field)
        if isinstance(config_section, (dict, list, DictConfig, ListConfig)):
            branch = tree.add(field)
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
            branch.add(rich.syntax.Syntax(branch_content, "yaml"))
        else:
            branch_content = json.dumps(config_section).strip('"')
            tree.add(f"{field}: {branch_content}")

    rich.print(Panel(tree))


def install_requirements(requirements):
    if not requirements:
        return

    subprocess.check_call([sys.executable, "-m", "pip", "install", *requirements])


def install_packages_with_apt(packages):
    if not packages:
        return

    try:
        import apt
    except ModuleNotFoundError:
        if sys.platform == "linux":
            warnings.warn("We couldn't install packages. Package `python-apt` is not installed. "
                          "Please install it manually")
        else:
            warnings.warn("We couldn't install packages. Packages are only for debian-based systems")
        return

    cache = apt.Cache()
    cache.update()
    cache.open()

    for package_name in packages:
        pkg = cache[package_name]

        if pkg.is_installed:
            continue

        pkg.mark_install()

    cache.commit()


def install_packages_with_cli(packages, sudo=False):
    if not packages:
        return

    command = ["apt-get", "install", "-y", *packages]

    if sudo:
        command.insert(0, "sudo")

    subprocess.check_call(command)


def print_install_requirements(lib, name):
    pip_install_command = f"pip install {lib}\[{name}]"  # noqa

    rich.print(Panel(f":floppy_disk: INSTALL PIP DEPENDENCIES\n{pip_install_command}"))


def print_install_packages(packages):
    if not packages:
        return

    command = f"apt install {' '.join(packages)}"
    rich.print(Panel(f":floppy_disk: INSTALL APT DEPENDENCIES\n{command}"))


def print_packages(packages, autocolor=True):
    tree = rich.tree.Tree(":package: APT DEPENDENCIES")
    for package in packages:
        text = package

        if autocolor:
            is_installed = is_package_installed(package)
            if is_installed is None:
                color = "bold blue"
            elif is_installed:
                color = "green"
            else:
                color = "bold red"
        else:
            color = "bold blue"

        tree.add(Text(text, color))

    rich.print(Panel(tree))


def is_package_installed(package):
    try:
        import apt
    except ModuleNotFoundError:
        return None

    cache = apt.Cache()

    pkg = cache.get(package)

    return pkg is not None and pkg.is_installed


def get_missing_packages(packages):
    try:
        import apt
    except ModuleNotFoundError:
        return None

    cache = apt.Cache()

    missing_packages = []

    for package in packages:
        pkg = cache.get(package)

        if pkg is None or not pkg.is_installed:
            missing_packages.append(package)

    return missing_packages


def print_requirements(dependencies, autocolor=True):
    tree = rich.tree.Tree(":package: PIP DEPENDENCIES")
    for dependency in dependencies:
        text = dependency

        if autocolor:
            try:
                pkg_resources.require(dependency)
                color = "green"
            except pkg_resources.DistributionNotFound:
                color = "bold red"
            except pkg_resources.VersionConflict as e:
                current_version = e.dist.version
                color = "red"
                text = f"{dependency} (got {current_version})"
        else:
            color = "bold blue"

        tree.add(Text(text, color))

    rich.print(Panel(tree))


def print_artifacts(artifacts_dir):
    """Recursively build a Tree with directory contents."""
    tree = rich.tree.Tree(":open_file_folder: ARTIFACTS")
    _walk_directory_for_rich(artifacts_dir, tree)
    rich.print(Panel(tree))


def _walk_directory_for_rich(directory, tree):
    paths = sorted(pathlib.Path(directory).iterdir(), key=lambda path: (path.is_file(), path.name.lower()))

    for path in paths:
        # Remove hidden files
        if path.name.startswith("."):
            continue
        if path.is_dir():
            style = "dim" if path.name.startswith("__") else ""
            branch = tree.add(
                f"[bold magenta]:open_file_folder: [link file://{path}]{escape(path.name)}",
                style=style,
                guide_style=style,
            )
            _walk_directory_for_rich(path, branch)
        else:
            text_filename = Text(path.name, "green")
            text_filename.highlight_regex(r"\..*$", "bold red")
            text_filename.stylize(f"link file://{path}")
            file_size = path.stat().st_size
            text_filename.append(f" ({decimal(file_size)})", "blue")
            tree.add(Text("📄 ") + text_filename)


def find_dependencies(path):
    with open(path) as fp:
        code = fp.read()

    tree = ast.parse(code)

    dependencies = None
    assigns = [x for x in tree.body if isinstance(x, ast.Assign)]

    for assign in assigns:
        if assign.targets and isinstance(assign.value, ast.List) and assign.targets[0].id == "dependencies":
            dependencies = []
            for elem in assign.value.elts:
                if isinstance(elem, ast.Str):
                    dependencies.append(elem.s)

    return dependencies


def find_model_module(trainer_target):
    lib_name, *mod_name, cls_name = trainer_target.split(".")
    lib_dir = os.path.dirname(pkgutil.get_loader(lib_name).get_filename())
    model_dir = find_file_parent_dir("__model__.py", cwd=os.path.join(lib_dir, *mod_name))

    if model_dir is None:
        raise FileNotFoundError(f"__model__.py not found for {trainer_target}")

    model_dirpath = os.path.relpath(model_dir, lib_dir)

    model_modname = model_dirpath.replace(os.path.sep, ".")

    model_module = importlib.import_module("." + model_modname, lib_name)

    return model_module


def find_dataset_module(dataset_target):
    lib_name, *mod_name, cls_name = dataset_target.split(".")
    lib_dir = os.path.dirname(pkgutil.get_loader(lib_name).get_filename())
    dataset_dir = find_file_parent_dir("__dataset__.py", cwd=os.path.join(lib_dir, *mod_name))

    if dataset_dir is None:
        raise FileNotFoundError(f"__dataset__.py not found for {dataset_target}")

    dataset_dirpath = os.path.relpath(dataset_dir, lib_dir)
    dataset_modname = dataset_dirpath.replace(os.path.sep, ".")

    dataset_module = importlib.import_module("." + dataset_modname, lib_name)

    return dataset_module


def find_env_module(env_target):
    lib_name, *mod_name, cls_name = env_target.split(".")
    lib_dir = os.path.dirname(pkgutil.get_loader(lib_name).get_filename())
    env_dir = find_file_parent_dir("__env__.py", cwd=os.path.join(lib_dir, *mod_name))

    if env_dir is None:
        raise FileNotFoundError(f"__env__.py not found for {env_target}")

    env_dirpath = os.path.relpath(env_dir, lib_dir)
    env_modname = env_dirpath.replace(os.path.sep, ".")

    env_module = importlib.import_module("." + env_modname, lib_name)

    return env_module


def find_predictors(model_module):
    predictors = []

    for var_name in dir(model_module):
        var = getattr(model_module, var_name)

        if isinstance(var, Proxy):
            var = var.__wrapped__

        if isinstance(var, type) and issubclass(var, BasePredictor):
            predictors.append(var_name)

    return predictors


def iterate_config(config: DictConfig, prefix=""):
    for key, value in config.items():
        if isinstance(value, DictConfig):
            yield from iterate_config(value, f"{prefix}{key}/")
        else:
            yield f"{prefix}{key}", value


def confirm_prompt(question: str) -> bool:
    """ Prompt the yes/no-*question* to the user. """
    from distutils.util import strtobool

    while True:
        user_input = input(question + " [y/n]: ")
        try:
            return bool(strtobool(user_input))
        except ValueError:
            print("Please use y/n or yes/no.")


def iter_modules(fname, root=None):
    if root is None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    for path in glob.iglob(os.path.join(root, "**", fname), recursive=True):
        module_path = os.path.relpath(path, root)
        module_name = remove_suffix("..." + module_path.replace(os.path.sep, "."), ".py")
        yield importlib.import_module(module_name, __name__)


def find_file_parent_dir(fname, cwd) -> Optional[str]:
    found = False
    current_dir = cwd

    while not found:
        if os.path.isfile(os.path.join(current_dir, fname)):
            found = True
        else:
            current_dir = os.path.dirname(current_dir)

        if current_dir == os.path.sep:
            break

    if not found:
        return None

    return current_dir
