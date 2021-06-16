#
#
#   Command utils
#
#

import ast
import json
import pkgutil
import rich.tree
import rich.syntax
import pkg_resources
from rich.text import Text
from typing import Sequence
from omegaconf import DictConfig, OmegaConf, ListConfig


def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "data",
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

    rich.print(tree)


def print_dependencies(dependencies):
    style = "dim"
    tree = rich.tree.Tree(":computer: DEPENDENCIES", style=style, guide_style=style)
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

    rich.print(tree)


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
    *module, trainer = trainer_config["_target_"].split(".")
    package = pkgutil.get_loader(".".join(module))
    dependencies = find_dependencies(package.get_filename())
    return dependencies


def iterate_config(config: DictConfig, prefix=""):
    for key, value in config.items():
        if isinstance(value, DictConfig):
            yield from iterate_config(value, f"{prefix}{key}/")
        else:
            yield f"{prefix}{key}", value
