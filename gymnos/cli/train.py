#
#
#   Train
#
#

import os
import sys
import uuid
import rich
import mlflow
import pydoc
import hydra
import logging
import importlib

from rich.panel import Panel
from collections import defaultdict
from distutils.util import strtobool
from omegaconf import DictConfig, open_dict
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate, get_original_cwd
from hydra_plugins.sofia_launcher import SOFIALauncherHydraConf

from ..dummy import DummyDataset
from ..base import BaseTrainer, BaseRLTrainer
from .utils import (print_requirements, iterate_config, print_install_requirements,
                    iter_modules, find_predictors, find_model_module, find_dataset_module, print_config,
                    print_packages, get_missing_packages, print_install_packages, install_packages_with_apt,
                    install_requirements, install_packages_with_cli, find_env_module)
from ..config import get_gymnos_home
from ..utils.py_utils import remove_prefix
from ..utils.pypi_utils import get_missing_dependencies


cs = ConfigStore.instance()

# Register SOFIA launcher
cs.store(group="hydra/launcher", name="sofia", node=SOFIALauncherHydraConf)

# Register models for Hydra
for module in iter_modules("__model__.py"):
    modname = remove_prefix(module.__package__, "gymnos.")
    cs.store(group="trainer", name=modname, node=getattr(module, "hydra_conf"))

# Register datasets for Hydra
for module in iter_modules("__dataset__.py"):
    *_, modname = module.__package__.split(".")
    cs.store(group="dataset", name=modname, node=getattr(module, "hydra_conf"))

# Register envs for Hydra
for module in iter_modules("__env__.py"):
    *_, modname = module.__package__.split(".")
    cs.store(group="env", name=modname, node=getattr(module, "hydra_conf"))


def main(config: DictConfig):
    logger = logging.getLogger(__name__)

    is_sofia_env = strtobool(os.getenv("SOFIA", "false"))

    logger.info(f"Outputs will be stored on: {os.getcwd()}")

    if config.verbose:
        print_config(config, resolve=True)

    model_module = find_model_module(config.trainer["_target_"])
    model_lib_name, model_mod_name = model_module.__name__.split(".", 1)
    model_meta_module = importlib.import_module("." + model_mod_name + ".__model__", model_lib_name)

    mod_name_to_install_by_lib = defaultdict(list)
    apt_dependencies = getattr(model_meta_module, "apt_dependencies", []).copy()
    pip_dependencies = getattr(model_meta_module, "pip_dependencies", []).copy()

    if model_mod_name != "dummy":
        mod_name_to_install_by_lib[model_lib_name].append(model_mod_name)

    if "dataset" in config:
        dataset_module = find_dataset_module(config.dataset["_target_"])
        dataset_lib_name, dataset_mod_name = dataset_module.__name__.split(".", 1)
        dataset_meta_module = importlib.import_module("." + dataset_mod_name + ".__dataset__", dataset_lib_name)

        apt_dependencies.extend(getattr(dataset_meta_module, "apt_dependencies", []))
        pip_dependencies.extend(getattr(dataset_meta_module, "pip_dependencies", []))

        if dataset_mod_name != "dummy":
            mod_name_to_install_by_lib[dataset_lib_name].append(dataset_mod_name)

    if "env" in config:
        env_module = find_env_module(config.env["_target_"])
        env_lib_name, env_mod_name = env_module.__name__.split(".", 1)
        env_meta_module = importlib.import_module("." + env_mod_name + ".__env__", env_lib_name)

        apt_dependencies.extend(getattr(env_meta_module, "apt_dependencies", []))
        pip_dependencies.extend(getattr(env_meta_module, "pip_dependencies", []))

        if env_mod_name != "dummy":
            mod_name_to_install_by_lib[env_lib_name].append(env_mod_name)

    if config.verbose:
        print_packages(apt_dependencies)
        print_requirements(pip_dependencies)

        missing_packages = get_missing_packages(apt_dependencies)
        missing_dependencies = get_missing_dependencies(pip_dependencies)

        if sys.platform == "linux":
            if missing_packages is None:
                logger.error("Apt dependency `python-apt` is not installed so we couln't retrieve installed packages")
            elif missing_packages:
                logger.warning("Some apt dependencies are missing")
                print_install_packages(missing_packages)

        if missing_dependencies:
            logger.warning("Some pip dependencies are missing")

            for lib_name, mod_names in mod_name_to_install_by_lib.items():
                print_install_requirements(lib_name, ",".join(mod_names))

    if config.install:
        if is_sofia_env:
            install_packages_with_cli(apt_dependencies, sudo=True)
        else:
            install_packages_with_apt(apt_dependencies)

        install_requirements(pip_dependencies)

    trainer_cls = pydoc.locate(config.trainer["_target_"])

    if trainer_cls is None:
        raise ImportError(f"Error importing {config.trainer['_target_']}")

    if issubclass(trainer_cls, BaseRLTrainer) and "env" not in config:
        with open_dict(config):
            config.env = "???"  # Make env mandatory
        _ = config.env   # Raise error for mandatory env key value

    if issubclass(trainer_cls, BaseTrainer) and "dataset" not in config:
        with open_dict(config):
            config.dataset = "???"  # Make dataset mandatory
        _ = config.dataset  # Raise error for mandatory dataset key value

    if "MLFLOW_TRACKING_URI" in os.environ:
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    else:
        tracking_uri = "file://" + os.path.join(get_original_cwd(), "mlruns")

    # Fix tracking uri and hydra cwd
    if tracking_uri.startswith("file://"):
        tracking_path = remove_prefix(tracking_uri, "file://")
        if not os.path.isabs(tracking_path):
            tracking_uri = "file://" + os.path.join(get_original_cwd(), tracking_path)

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run(run_name=config.mlflow.run_name) as run:
        logger.info(f"MLFlow run id: {run.info.run_id}")

        predictors = find_predictors(model_module)

        # Show usage
        usage_strs = []
        for predictor in predictors:
            import_str = f"from {model_module.__name__} import {predictor}"
            use_str = f'predictor = {predictor}.from_pretrained("{run.info.run_id}")'
            usage_strs.append(import_str + "\n" + use_str)

        if config.verbose:
            usage_str = "\nor\n".join(usage_strs)
            rich.print(Panel(f":computer: USAGE\n{usage_str}"))

        mlflow.log_artifact(".hydra")

        if is_sofia_env:
            print({
                "sofia": True,
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "module": model_module.__name__,
                "predictors": predictors
            })

        if config.mlflow.log_trainer_params:
            mlflow.log_params(dict(iterate_config(config.trainer, "trainer/")))

        if "dataset" in config and config.mlflow.log_dataset_params:
            mlflow.log_params(dict(iterate_config(config.dataset, "dataset/")))

        if "env" in config and config.mlflow.log_env_params:
            mlflow.log_params(dict(iterate_config(config.env, "env/")))

        trainer = instantiate(config.trainer)

        if "env" in config:
            import gym

            env_id = str(uuid.uuid4())[:8] + "-v0"
            # kwargs = OmegaConf.to_container(config.env)
            # entry_point = rreplace(kwargs.pop("_target_"), ".", ":")

            def entry_point():
                return instantiate(config.env)

            gym.envs.register(
                id=env_id,
                entry_point=entry_point,
                max_episode_steps=getattr(env_meta_module, "max_episode_steps", None),
                reward_threshold=getattr(env_meta_module, "reward_threshold", None),
                nondeterministic=getattr(env_meta_module, "nondeterministic", False),
            )

            trainer.prepare_env(env_id)
        elif "dataset" in config:
            dataset = instantiate(config.dataset)

            if isinstance(dataset, DummyDataset):
                data_dir = dataset.path
            else:
                *_, dataset_name = dataset_module.__name__.split(".")
                data_dir = os.path.join(get_gymnos_home(), "datasets", dataset_name)
                os.makedirs(data_dir, exist_ok=True)

            dataset.download(data_dir)
            trainer.prepare_data(data_dir)

        trainer.train()

        if config.test:
            trainer.test()


@hydra.main(config_path="../../conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
