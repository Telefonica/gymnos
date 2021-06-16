#
#
#   Utils
#
#

import git.exc
import logging
import warnings
import rich.tree
import rich.syntax

from git import Repo
from omegaconf import DictConfig, OmegaConf


def get_current_revision():
    logger = logging.getLogger(__name__)

    repo_not_valid_msg = "The current directory is not a valid gymnos repository. Skipping ..."

    try:
        repo = Repo()
    except git.exc.InvalidGitRepositoryError:
        logger.info(repo_not_valid_msg)
        return None

    gymnos_remote = None
    for remote in repo.remotes:
        if remote.url == "https://github.com/Telefonica/gymnos.git":
            gymnos_remote = remote

    if gymnos_remote is None:   # not a gymnos repository
        logger.info(repo_not_valid_msg)
        return None

    current_revision = repo.head.object.hexsha

    has_warnings = False

    if not repo.git.branch("-r", "--contains", current_revision):
        has_warnings = True
        logger.warning("The current revision have not been found on gymnos repository")

    if repo.is_dirty(untracked_files=True):
        has_warnings = True
        logger.warning(f"You have uncommited changes")

    if has_warnings:
        confirm = confirm_prompt("You have unpushed changes to gymnos repository. Are you sure you want to continue?: ")
        if not confirm:
            logger.info("Cancelled operation")
            raise SystemExit()

    logger.info(f"Current revision {current_revision[:8]} will be used")

    return current_revision


def confirm_prompt(question: str) -> bool:
    """ Prompt the yes/no-*question* to the user. """
    from distutils.util import strtobool

    while True:
        user_input = input(question + " [y/n]: ")
        try:
            return bool(strtobool(user_input))
        except ValueError:
            print("Please use y/n or yes/no.")


def print_launcher(config: DictConfig, resolve: bool = True):
    style = "dim"
    tree = rich.tree.Tree(":desktop_computer: SOFIA", style=style, guide_style=style)

    config = OmegaConf.to_container(config)

    branch = tree.add("hydra", style=style, guide_style=style)
    subbranch = branch.add("launcher")
    subbranch_content = OmegaConf.to_yaml(config, resolve=resolve)
    subbranch.add(rich.syntax.Syntax(subbranch_content, "yaml"))

    rich.print(tree)


def print_dependencies(dependencies):
    style = "dim"
    tree = rich.tree.Tree(":computer: DEPENDENCIES", style=style, guide_style=style)

    for dependency in dependencies:
        tree.add(dependency, style=style, guide_style=style)

    rich.print(tree)
