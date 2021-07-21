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
from rich.panel import Panel
from gymnos.cli.utils import confirm_prompt
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
        warnings.warn("The current revision have not been found on gymnos repository")

    if repo.is_dirty(untracked_files=True):
        has_warnings = True
        warnings.warn("You have uncommited changes")

    if has_warnings:
        confirm = confirm_prompt("You have unpushed changes to gymnos repository. Are you sure you want to continue?: ")
        if not confirm:
            logger.info("Cancelled operation")
            raise SystemExit()

    logger.info(f"Current revision {current_revision[:8]} will be used")

    return current_revision


def print_launcher(config: DictConfig, resolve: bool = True):
    tree = rich.tree.Tree(":desktop_computer:   SOFIA")

    config = OmegaConf.to_container(config)

    branch = tree.add("hydra")
    subbranch = branch.add("launcher")
    subbranch_content = OmegaConf.to_yaml(config, resolve=resolve)
    subbranch.add(rich.syntax.Syntax(subbranch_content, "yaml"))

    rich.print(Panel(tree))
