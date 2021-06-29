#
#
#   Train
#
#
import os
import sys
import hydra
import mlflow
import logging
import subprocess

from omegaconf import DictConfig
from distutils.util import strtobool
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate, get_original_cwd

from hydra_plugins.sofia_launcher import SOFIALauncherConfig
from .utils import print_config, print_dependencies, iterate_config, find_trainer_dependencies, \
    find_trainer_package, find_predictors, get_missing_dependencies, print_install

# Register SOFIA launcher
cs = ConfigStore.instance()
cs.store(group="hydra/launcher", name="sofia", node=SOFIALauncherConfig)


def main(config: DictConfig):
    logger = logging.getLogger(__name__)

    logger.info(f"Outputs will be stored on: {os.getcwd()}")

    if config.show_config:
        print_config(config, resolve=True)

    package = find_trainer_package(config.trainer)

    module = package.load_module()

    dependencies = find_trainer_dependencies(config.trainer)

    if dependencies is None:
        logger.warning("Error retrieving dependencies")
    else:
        if config.show_dependencies:
            print_dependencies(dependencies)

            missing_dependencies = get_missing_dependencies(dependencies)

            if missing_dependencies:
                logger.info("Some dependencies are missing")
                print_install(module)

        if config.dependencies.install:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *dependencies])

    if "MLFLOW_TRACKING_URI" in os.environ:
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    else:
        tracking_uri = os.path.join(get_original_cwd(), "mlruns")

    # Fix tracking uri and hydra cwd
    if tracking_uri.startswith("file://"):
        tracking_path = tracking_uri.lstrip("file://")
        if not os.path.isabs(tracking_path):
            tracking_uri = os.path.join("file://", get_original_cwd(), tracking_path)

    mlflow.set_tracking_uri(tracking_uri)

    if config.mlflow.experiment_name is not None:
        mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run(run_name=config.mlflow.run_name) as run:
        logger.info(f"MLFlow run id: {run.info.run_id}")

        mlflow.log_artifact(".hydra")

        is_sofia_env = strtobool(os.getenv("SOFIA", "false"))
        if is_sofia_env:
            package = find_trainer_package(config.trainer)
            module = package.load_module()

            print({
                "sofia": True,
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "module": module.__name__,
                "predictors": find_predictors(module)
            })

        if config.mlflow.log_trainer_params:
            mlflow.log_params(dict(iterate_config(config.trainer)))

        trainer = instantiate(config.trainer)

        if config.get("data") is not None:
            data_dir = instantiate(config.data)
            trainer.setup(data_dir)

        trainer.train()

        if config.test:
            trainer.test()


@hydra.main(config_path="../../conf", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
