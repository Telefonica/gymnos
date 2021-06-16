#
#
#   Launcher
#
#

import logging
import uuid

from omegaconf import DictConfig
from omegaconf import open_dict
from typing import Sequence, Optional
from hydra.plugins.launcher import Launcher
from hydra.core.hydra_config import HydraConfig
from hydra.types import TaskFunction, HydraContext
from hydra.core.utils import JobReturn, run_job, setup_globals, configure_log

from typing import Callable
from posixpath import join as urljoin
from gymnos.services.sofia import SOFIA
from gymnos.cli.utils import print_config, find_trainer_dependencies

from .config import Device
from .core import launch
from .utils import get_current_revision


class SOFIALauncher(Launcher):

    def __init__(self, project_name: str, ref: str = None, device: Device = "CPU", show=True):
        self.project_name = project_name
        self.ref = ref
        self.device = device
        self.show = show

        self.hydra_context: Optional[HydraContext] = None
        self.task_function: Optional[TaskFunction] = None
        self.config: Optional[DictConfig] = None

    def setup(self, *, hydra_context: HydraContext, task_function: TaskFunction, config: DictConfig):
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function

    def launch(self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int) -> Sequence[JobReturn]:
        # Configure launcher
        setup_globals()
        configure_log(self.config.hydra.job_logging, self.config.hydra.verbose)

        logger = logging.getLogger(__name__)

        if self.ref is None:
            logger.info("Revision not specified. Trying to retrieve the current revision for working directory")
            self.ref = get_current_revision()

        return launch(self, job_overrides, initial_job_idx)
