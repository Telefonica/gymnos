#
#
#   Launcher
#
#

import rich
import time
import uuid
import logging

from .hydra_conf import Device
from .utils import print_launcher
from .utils import get_current_revision
from .hydra_conf import SOFIALauncherHydraConf

from rich.markup import escape
from omegaconf import DictConfig
from omegaconf import open_dict
from dataclasses import dataclass
from posixpath import join as urljoin
from gymnos.services.sofia import SOFIA
from hydra.core.singleton import Singleton
from hydra.plugins.launcher import Launcher
from hydra.core.hydra_config import HydraConfig
from typing import Sequence, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from hydra.types import TaskFunction, HydraContext
from hydra.core.utils import JobReturn, run_job, configure_log, setup_globals


class SOFIAProjectNotFound(Exception):

    def __init__(self, project_name):
        message = f"Project `{project_name}` not found"
        super().__init__(message)


def execute_job(
    hydra_context: HydraContext,
    sweep_config: DictConfig,
    task_function: TaskFunction,
    singleton_state: Dict[Any, Any],
) -> JobReturn:
    setup_globals()
    Singleton.set_state(singleton_state)

    HydraConfig.instance().set_config(sweep_config)

    ret = run_job(
        hydra_context=hydra_context,
        task_function=task_function,
        config=sweep_config,
        job_dir_key="hydra.sweep.dir",
        job_subdir_key="hydra.sweep.subdir",
    )

    return ret


def create_job_on_sofia(idx, args: Sequence[str], project_name: str, ref: str, name: Optional[str],
                        description: Optional[str], device: Device, notify_on_completion: bool):
    def entrypoint(config):
        logger = logging.getLogger(__name__)

        response = SOFIA.create_project_job(args, project_name, ref, device.value, name, description,
                                            notify_on_completion)

        if not response.ok:
            if response.status_code == 404:
                raise SOFIAProjectNotFound(project_name)
            response.raise_for_status()

        data = response.json()

        job_name = data["name"]
        username = data["project"]["user"]["username"]

        job_url = urljoin(SOFIA.domain, username, "projects", project_name,
                          "jobs", job_name)
        logger.info(f"Project job successfully created at {job_url}")

        logger.info("Polling job statuses every 30 sec")

        job_prefix = f"[color({(idx + 1) % 255})]{job_name}  | [/]"

        current_lineno = 0
        while True:
            new_logs_response = SOFIA.get_project_job_logs(data["project"]["user"]["username"], project_name,
                                                           data["name"], current_lineno)

            if new_logs_response.ok:
                text = new_logs_response.text.strip()
                if text and text != '""':
                    lines = text.split("\n")
                    current_lineno += len(lines) + 1

                    for line in lines:
                        if not line.startswith("{'sofia': True"):
                            rich.print(job_prefix, escape(line))
            else:
                rich.print(job_prefix, "[bold red]Error retrieving new logs ...[/]")

            job_response = SOFIA.get_project_job(data["project"]["user"]["username"], project_name, data["name"])
            if job_response.ok:
                job = job_response.json()

                if job["status"] in ("SUCCESS", "FAILURE", "ABORTED", "REVOKED"):
                    return job.get("optimized_metric")
            else:
                rich.print(job_prefix, "[bold red]Error retrieving job update ...[/]")

            time.sleep(30)

    return entrypoint


def launch(launcher, job_overrides, initial_job_idx) -> Sequence[JobReturn]:
    setup_globals()

    if launcher.verbose:
        print_launcher(launcher.config.hydra.launcher)

    with ThreadPoolExecutor() as executor:
        futures = []
        singleton_state = Singleton.get_state()

        for idx, args in enumerate(job_overrides):
            sweep_config = launcher.hydra_context.config_loader.load_sweep_config(launcher.config, list(args))

            with open_dict(sweep_config):
                sweep_config.hydra.job.id = str(uuid.uuid4())
                sweep_config.hydra.job.num = initial_job_idx + idx
                sweep_config.hydra.launcher.ref = launcher.ref

            task_function = create_job_on_sofia(idx, args, launcher.project_name, launcher.ref, launcher.name,
                                                launcher.description, launcher.device, launcher.notify_on_completion)

            future = executor.submit(execute_job, launcher.hydra_context, sweep_config, task_function,
                                     singleton_state)

            futures.append(future)

        runs = []

        for future in futures:
            ret_value = future.result()
            runs.append(ret_value)

    return runs


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
        configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)

        if self.ref is None:
            self.ref = get_current_revision() or "master"

        return launch(self, job_overrides, initial_job_idx)
