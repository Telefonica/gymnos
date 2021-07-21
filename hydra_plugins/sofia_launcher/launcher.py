#
#
#   Launcher
#
#

from typing import Sequence
from omegaconf import DictConfig
from dataclasses import dataclass
from hydra.plugins.launcher import Launcher
from hydra.types import TaskFunction, HydraContext
from hydra.core.utils import JobReturn, setup_globals, configure_log

from .core import launch
from .utils import get_current_revision
from .hydra_conf import SOFIALauncherHydraConf


@dataclass
class SOFIALauncher(SOFIALauncherHydraConf, Launcher):
    """
    Launch training on the SOFIA platform

    Parameters
    ------------
    project_name:
        SOFIA project name
    ref:
        Gymnos release, branch or commit to execute training. Defaults to current commit.
    device:
        Device to execute training. One of the following: ``CPU``, ``GPU``.
    """

    def setup(self, *, hydra_context: HydraContext, task_function: TaskFunction, config: DictConfig):
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function

    def launch(self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int) -> Sequence[JobReturn]:
        # Configure launcher
        setup_globals()
        configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)

        if self.ref is None:
            self.ref = get_current_revision()

        return launch(self, job_overrides, initial_job_idx)
