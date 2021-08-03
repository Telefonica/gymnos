#
#
#   Train
#
#

import os
import sys
import rich
import mlflow
import hydra
import logging
import importlib

from rich.panel import Panel
from omegaconf import DictConfig
from distutils.util import strtobool
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate, get_original_cwd
from hydra_plugins.sofia_launcher import SOFIALauncherHydraConf

from ..dummy import DummyDataset
from .utils import (print_requirements, iterate_config, get_missing_requirements, print_install_requirements,
                    iter_modules, find_predictors, find_model_module, find_dataset_module, print_config,
                    print_packages, get_missing_packages, print_install_packages, install_packages_with_apt,
                    install_requirements, install_packages_with_cli)
from ..config import get_gymnos_home
from ..utils.py_utils import remove_prefix


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


def main(config: DictConfig):
    logger = logging.getLogger(__name__)

    is_sofia_env = strtobool(os.getenv("SOFIA", "false"))

    logger.info(f"Outputs will be stored on: {os.getcwd()}")

    if config.verbose:
        print_config(config, resolve=True)

    model_module = find_model_module(config.trainer["_target_"])
    model_lib_name, model_mod_name = model_module.__name__.split(".", 1)
    model_meta_module = importlib.import_module("." + model_mod_name + ".__model__", model_lib_name)

    dataset_module = find_dataset_module(config.dataset["_target_"])

    *_, dataset_name = dataset_module.__name__.split(".")

    packages = getattr(model_meta_module, "apt_dependencies", [])
    dependencies = getattr(model_meta_module, "pip_dependencies", [])

    if config.verbose:
        print_requirements(dependencies)
        print_packages(packages)

        missing_packages = get_missing_packages(packages)
        missing_dependencies = get_missing_requirements(dependencies)

        if missing_dependencies:
            logger.info("Some requirements are missing")
            print_install_requirements(model_lib_name, model_mod_name)

        if sys.platform == "linux":
            if missing_packages is None:
                logger.warning("Package `python-apt` so we couln't retrieve installed packages")
            elif missing_packages:
                logger.info("Some packages are missing")
                print_install_packages(missing_packages)

    if config.install:
        if is_sofia_env:
            install_packages_with_cli(packages, sudo=True)
        else:
            install_packages_with_apt(packages)

        install_requirements(dependencies)

    if "MLFLOW_TRACKING_URI" in os.environ:
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    else:
        tracking_uri = "file://" + os.path.join(get_original_cwd(), "mlruns")

    # # Fix tracking uri and hydra cwd
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
            mlflow.log_params(dict(iterate_config(config.trainer)))

        trainer = instantiate(config.trainer)

        if config.get("dataset") is not None:
            dataset = instantiate(config.dataset)

            if isinstance(dataset, DummyDataset):
                data_dir = dataset.path
            else:
                data_dir = os.path.join(get_gymnos_home(), "datasets", dataset_name)
                os.makedirs(data_dir, exist_ok=True)

            dataset(data_dir)
            trainer.setup(data_dir)

        trainer.train()

        if config.test:
            trainer.test()


@hydra.main(config_path="../../conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
