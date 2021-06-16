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
from hydra.types import TaskFunction, HydraContext
from hydra.core.utils import JobReturn, run_job, setup_globals

from typing import Callable
from posixpath import join as urljoin
from gymnos.services.sofia import SOFIA
from gymnos.cli.utils import print_config, find_trainer_dependencies

from .config import Device
from .utils import get_current_revision, print_launcher, print_dependencies

logger = logging.getLogger(__name__)


class SOFIAProjectNotFound(Exception):

    def __init__(self, project_name):
        message = f"Project `{project_name}` not found"
        super().__init__(message)


def launch(args: Sequence[str], project_name: str, ref: str, device: Device) -> Callable:
    def entrypoint(config):
        response = SOFIA.create_project_job(args, project_name, ref, device.value)

        if not response.ok:
            if response.status_code == 404:
                raise SOFIAProjectNotFound(project_name)
            response.raise_for_status()

        data = response.json()

        job_url = urljoin(SOFIA.domain, data["project"]["user"]["username"], "projects", project_name,
                          "jobs", data["name"])
        logging.info(f"Project job successfully created at {job_url}")

    return entrypoint


class SOFIALauncher(Launcher):

    def __init__(self, project_name: str, ref: str = None, device: Device = "CPU", show=True):
        if ref is None:
            logger.info("Revision not specified. Trying to retrieve the current revision for working directory")
            ref = get_current_revision()

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
        job_returns = []

        setup_globals()

        if self.show:
            print_launcher(self.config.hydra.launcher)

        for idx, args in enumerate(job_overrides):
            sweep_config = self.hydra_context.config_loader.load_sweep_config(self.config, list(args))

            with open_dict(sweep_config):
                sweep_config.hydra.job.id = str(uuid.uuid4())
                sweep_config.hydra.job.num = initial_job_idx + idx

            if sweep_config.show_config:
                print_config(sweep_config, ("trainer", "data", "test"))

            if sweep_config.show_dependencies:
                dependencies = find_trainer_dependencies(sweep_config.trainer)
                print_dependencies(dependencies)

            job_return = run_job(
                launch(args, self.project_name, self.ref, self.device),
                config=sweep_config,
                hydra_context=self.hydra_context,
                job_dir_key="hydra.sweep.dir",
                job_subdir_key="hydra.sweep.subdir",
            )
            job_returns.append(job_return)

        return job_returns
