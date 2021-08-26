#
#
#   Core
#
#

import uuid
import logging
import importlib

from omegaconf import open_dict
from typing import Sequence, Callable
from posixpath import join as urljoin
from gymnos.services.sofia import SOFIA
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import JobReturn, run_job, configure_log
from gymnos.cli.utils import print_config, find_model_module, print_requirements, print_packages

from .hydra_conf import Device
from .utils import print_launcher


class SOFIAProjectNotFound(Exception):

    def __init__(self, project_name):
        message = f"Project `{project_name}` not found"
        super().__init__(message)


def launch_job(args: Sequence[str], project_name: str, ref: str, device: Device, notify_on_completion: bool) -> Callable:
    def entrypoint(config):
        logger = logging.getLogger(__name__)

        response = SOFIA.create_project_job(args, project_name, ref, device.value, notify_on_completion)

        if not response.ok:
            if response.status_code == 404:
                raise SOFIAProjectNotFound(project_name)
            response.raise_for_status()

        data = response.json()

        job_url = urljoin(SOFIA.domain, data["project"]["user"]["username"], "projects", project_name,
                          "jobs", data["name"])
        logger.info(f"Project job successfully created at {job_url}")

    return entrypoint


def launch(launcher, job_overrides: Sequence[Sequence[str]], initial_job_idx: int) -> Sequence[JobReturn]:
    job_returns = []

    if launcher.verbose:
        print_launcher(launcher.config.hydra.launcher)

    for idx, args in enumerate(job_overrides):
        sweep_config = launcher.hydra_context.config_loader.load_sweep_config(launcher.config, list(args))

        with open_dict(sweep_config):
            sweep_config.hydra.job.id = str(uuid.uuid4())
            sweep_config.hydra.job.num = initial_job_idx + idx
            sweep_config.hydra.launcher.ref = launcher.ref

        HydraConfig.instance().set_config(sweep_config)

        if sweep_config.verbose:
            print_config(sweep_config, ("trainer", "data", "test"))

        model_module = find_model_module(sweep_config.trainer["_target_"])
        model_lib_name, model_mod_name = model_module.__name__.split(".", 1)
        model_meta_module = importlib.import_module("." + model_mod_name + ".__model__", model_lib_name)

        if sweep_config.verbose:
            print_requirements(getattr(model_meta_module, "pip_dependencies", []), autocolor=False)
            print_packages(getattr(model_meta_module, "apt_dependencies", []), autocolor=False)

        job_return = run_job(
            launch_job(args, launcher.project_name, launcher.ref, launcher.device, launcher.notify_on_completion),
            config=sweep_config,
            hydra_context=launcher.hydra_context,
            job_dir_key="hydra.sweep.dir",
            job_subdir_key="hydra.sweep.subdir",
        )
        job_returns.append(job_return)

        configure_log(launcher.config.hydra.hydra_logging, launcher.config.hydra.verbose)

    return job_returns
