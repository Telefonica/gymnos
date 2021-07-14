#
#
#   Command utils
#
#

import os
import ast
import glob
import json
import pkgutil
import inspect
import pathlib
import importlib
import rich.tree
import rich.syntax
import pkg_resources

from ..base import BasePredictor
from ..utils.py_utils import remove_suffix

from rich.text import Text
from typing import Sequence, Optional
from rich.panel import Panel
from rich.markup import escape
from rich.filesize import decimal
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

    style = "dim"
    tree = rich.tree.Tree(":gear: CONFIG", style=style, guide_style=style)

    for field in fields:
        config_section = config.get(field)
        if isinstance(config_section, (dict, list, DictConfig, ListConfig)):
            branch = tree.add(field, style=style, guide_style=style)
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
            branch.add(rich.syntax.Syntax(branch_content, "yaml"))
        else:
            branch_content = json.dumps(config_section).strip('"')
            tree.add(f"{field}: {branch_content}", style=style, guide_style=style)

    rich.print(Panel(tree))


def get_missing_dependencies(dependencies):
    missing_dependencies = []
    for dependency in dependencies:
        try:
            pkg_resources.require(dependency)
        except pkg_resources.DistributionNotFound:
            missing_dependencies.append(dependency)
    return missing_dependencies


def print_install(lib, name):
    pip_install_command = f"pip install {lib}\[{name}]"

    rich.print(Panel(f":floppy_disk: INSTALL\n{pip_install_command}"))


def print_dependencies(dependencies):
    style = "dim"
    tree = rich.tree.Tree(":package: DEPENDENCIES", style=style, guide_style=style)
    for dependency in dependencies:
        text = dependency
        try:
            pkg_resources.require(dependency)
            color = "green"
        except pkg_resources.DistributionNotFound:
            color = "bold red"
        except pkg_resources.VersionConflict as e:
            current_version = e.dist.version
            color = "red"
            text = f"{dependency} (got {current_version})"

        tree.add(Text(text, color), style=style, guide_style=style)

    rich.print(Panel(tree))


def print_artifacts(artifacts_dir):
    """Recursively build a Tree with directory contents."""
    tree = rich.tree.Tree(f":open_file_folder: ARTIFACTS", guide_style="dim")
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


def find_trainer_dependencies(trainer_config):
    package = find_trainer_package(trainer_config)
    dependencies = find_dependencies(package.get_filename())
    return dependencies


def find_trainer_package(trainer_config):
    *module, trainer = trainer_config["_target_"].split(".")
    package = pkgutil.get_loader(".".join(module))
    return package


def find_predictors(module):
    predictors = []

    for var_name in dir(module):
        if var_name.startswith("__"):
            continue

        var = getattr(module, var_name)

        if inspect.isclass(var) and issubclass(var, BasePredictor):
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


def iter_modules(fname):
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

    if not found:
        return None

    return current_dir
