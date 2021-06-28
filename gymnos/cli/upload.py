#
#
#   Upload
#
#

import os
import re
import click
import mlflow
import tempfile
import logging
import rich.tree

from rich.panel import Panel
from omegaconf import OmegaConf
from rich.console import Console
from rich import print as rprint
from rich.logging import RichHandler
from posixpath import join as urljoin

from ..services.sofia import SOFIA
from ..utils.data_utils import zipdir
from ..utils.mlflow_utils import jsonify_mlflow_run
from .utils import find_trainer_package, confirm_prompt, find_predictors, print_artifacts


def _get_mlflow_run(ctx, param, value):
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(value)
    return run


def _validate_name(ctx, param, value):
    pattern = r"^([A-Za-z0-9\-\_]+)$"
    if not re.match(pattern, value):
        raise click.BadParameter("Only letters, numbers, dashes and underscores")

    if len(value) > 100:
        raise click.BadParameter("Maximum length is 100 characters")

    return value


@click.command()
@click.argument("mlflow_run_id", callback=_get_mlflow_run)
@click.option("--name", prompt=True, callback=_validate_name)
@click.option("--description", prompt=True, default="")
@click.option("--public", prompt=True, is_flag=True)
def main(mlflow_run_id, name, description, public):
    if not description:
        description = None

    mlflow_run = mlflow_run_id

    handler = RichHandler(rich_tracebacks=True)
    logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[handler])

    logger = logging.getLogger(__name__)

    response = SOFIA.get_current_user()

    response.raise_for_status()

    user = response.json()

    client = mlflow.tracking.MlflowClient()

    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts = client.list_artifacts(mlflow_run.info.run_id)

        assert len(artifacts) > 0, "Artifacts directory empty. Must at least contain .hydra directory"

        artifacts_dir = os.path.join(tmpdir, "artifacts")

        os.makedirs(artifacts_dir)

        client.download_artifacts(mlflow_run.info.run_id, "", dst_path=artifacts_dir)

        config = OmegaConf.load(os.path.join(artifacts_dir, ".hydra", "config.yaml"))

        package = find_trainer_package(config.trainer)

        with open(package.get_filename()) as fp:
            header = fp.readline()
            has_model_header = header.startswith("#") and header.strip("# \n") == "@model"
            assert has_model_header, "Invalid model. Must start with # @model"

        module = package.load_module()

        predictors = find_predictors(module)

        lib, *path = module.__name__.split(".")

        rprint(Panel(f"{':unlocked:' if public else ':locked:'}{user['username']}/models/{name}\n{description or '[italic]No description available'}"))

        pip_install_command = f"pip install {lib}\[{'.'.join(path)}]"

        rprint(Panel(f":floppy_disk: INSTALL\n{pip_install_command}"))

        style = "dim"
        dependencies_tree = rich.tree.Tree(":package: DEPENDENCIES", style=style, guide_style=style)

        for dependency in module.dependencies:
            dependencies_tree.add(dependency, style=style, guide_style=style)

        rprint(Panel(dependencies_tree))

        print_artifacts(artifacts_dir)

        resource_url = f"{user['username']}/models/{name}"

        usage_strs = []
        for predictor in predictors:
            import_str = f"from {module.__name__} import {predictor}"
            use_str = f'model = {predictor}.from_pretrained("{resource_url}")'
            usage_strs.append(import_str + "\n" + use_str)

        usage_str = "\nor\n".join(usage_strs)
        rprint(Panel(f":computer: USAGE\n{usage_str}"))

        confirm = confirm_prompt("Is this OK?")

        if not confirm:
            logger.info("Operation cancelled")
            return

        artifacts_zip_fpath = os.path.join(tmpdir, "artifacts.zip")

        console = Console()

        with console.status("[bold green]Compressing artifacts..."):
            zipdir(artifacts_dir, artifacts_zip_fpath)

        run_info = jsonify_mlflow_run(mlflow_run)

        with console.status("[bold green]Uploading model..."):
            response = SOFIA.create_model(name, description, public, module.__name__, predictors, config, run_info,
                                          artifacts_zip_fpath)

        response.raise_for_status()

        data = response.json()

        model_url = urljoin(SOFIA.domain, data["user"]["username"], "models", data["name"])
        console.print(f":party_popper: [bold green]Successfully created. Live at {model_url}")
