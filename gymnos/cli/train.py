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
import pkgutil
import importlib
import subprocess

from rich.panel import Panel
from omegaconf import DictConfig
from lazy_object_proxy import Proxy
from distutils.util import strtobool
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate, get_original_cwd

from hydra_plugins.sofia_launcher import SOFIALauncherConfig
from .utils import (print_config, print_dependencies, iterate_config, get_missing_dependencies, print_install,
                    iter_modules, find_file_parent_dir)

from ..base import BasePredictor
from ..config import get_gymnos_home
from ..utils.py_utils import remove_prefix


cs = ConfigStore.instance()

# Register SOFIA launcher
cs.store(group="hydra/launcher", name="sofia", node=SOFIALauncherConfig)

# Register models for Hydra
for module in iter_modules("__model__.py"):
    cs.store(group="trainer", name=getattr(module, "name"), node=getattr(module, "conf"))

# Register datasets for Hydra
for module in iter_modules("__dataset__.py"):
    cs.store(group="dataset", name=getattr(module, "name"), node=getattr(module, "conf"))


def main(config: DictConfig):
    logger = logging.getLogger(__name__)

    logger.info(f"Outputs will be stored on: {os.getcwd()}")

    if config.show_config:
        print_config(config, resolve=True)

    trainer_target = config.trainer["_target_"]
    lib_name, *mod_name, cls_name = trainer_target.split(".")
    lib_dir = os.path.dirname(pkgutil.get_loader(lib_name).get_filename())

    model_dir = find_file_parent_dir("__model__.py", cwd=os.path.join(lib_dir, *mod_name))

    if model_dir is None:
        raise FileNotFoundError(f"__model__.py not found for {trainer_target}")

    model_dirpath = os.path.relpath(model_dir, lib_dir)

    model_modname = model_dirpath.replace(os.path.sep, ".")

    model_module = importlib.import_module("." + model_modname, lib_name)
    model_meta_module = importlib.import_module("." + model_modname + ".__model__", lib_name)

    dataset_target = config.dataset["_target_"]
    lib_name, *mod_name, cls_name = dataset_target.split(".")
    lib_dir = os.path.dirname(pkgutil.get_loader(lib_name).get_filename())
    dataset_dir = find_file_parent_dir("__dataset__.py", cwd=os.path.join(lib_dir, *mod_name))

    if dataset_dir is None:
        raise FileNotFoundError(f"__dataset__.py not found for {dataset_target}")

    dataset_dirpath = os.path.relpath(dataset_dir, lib_dir)
    dataset_modname = dataset_dirpath.replace(os.path.sep, ".")

    dataset_meta_module = importlib.import_module("." + dataset_modname + ".__dataset__", lib_name)

    dependencies = getattr(model_meta_module, "dependencies", None)

    if dependencies is None:
        logger.warning("Error retrieving dependencies")
    else:
        if config.show_dependencies:
            print_dependencies(dependencies)

            missing_dependencies = get_missing_dependencies(dependencies)

            if missing_dependencies:
                logger.info("Some dependencies are missing. Training will probably fail")
                print_install(lib_name, getattr(model_meta_module, "name"))

        if config.dependencies.install:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *dependencies])

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

        predictors = []

        for var_name in dir(model_module):
            var = getattr(model_module, var_name)

            if isinstance(var, Proxy):
                var = var.__wrapped__

            if isinstance(var, type) and issubclass(var, BasePredictor):
                predictors.append(var_name)

        # Show usage
        usage_strs = []
        for predictor in predictors:
            import_str = f"from {model_module.__name__} import {predictor}"
            use_str = f'predictor = {predictor}.from_pretrained("{run.info.run_id}")'
            usage_strs.append(import_str + "\n" + use_str)

        usage_str = "\nor\n".join(usage_strs)
        rich.print(Panel(f":computer: USAGE\n{usage_str}"))

        mlflow.log_artifact(".hydra")

        is_sofia_env = strtobool(os.getenv("SOFIA", "false"))
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
            data_dir = os.path.join(get_gymnos_home(), "datasets", getattr(dataset_meta_module, "name"))
            os.makedirs(data_dir, exist_ok=True)

            dataset = instantiate(config.dataset)
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
