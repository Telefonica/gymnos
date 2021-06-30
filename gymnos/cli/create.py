#
#
#   Create model
#
#

import os
import click
import inspect
import logging
import stringcase

from pathlib import Path
from rich.logging import RichHandler


@click.command()
@click.argument("task")
@click.argument("model")
def main(task, model):
    handler = RichHandler(rich_tracebacks=True)
    logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[handler])

    logger = logging.getLogger(__name__)

    here = os.path.abspath(os.path.dirname(__file__))
    app_dir = os.path.abspath(os.path.join(here, ".."))
    conf_dir = os.path.abspath(os.path.join(here, "..", "..", "conf"))

    task_dir = os.path.join(*task.split("/"))

    model_full_path = os.path.join(app_dir, task_dir, model)

    if os.path.isdir(model_full_path):
        logger.error(f"Model {model} for task {task} already exists")
        raise SystemExit(1)

    title_name = stringcase.titlecase(model)
    trainer_name = stringcase.pascalcase(model) + "Trainer"
    predictor_name = stringcase.pascalcase(model) + "Predictor"

    os.chdir(app_dir)

    for subtask in task.split("/"):
        os.makedirs(subtask, exist_ok=True)
        init_fpath = os.path.join(subtask, "__init__.py")
        if not os.path.isfile(init_fpath):
            Path(init_fpath).touch()
        os.chdir(subtask)

    os.chdir(here)

    init_model_filestr = inspect.cleandoc(f'''
        """
        Docstring for {title_name}
        """
   
        #  @model
        
        from .trainer import {trainer_name}
        from .predictor import {predictor_name}
        
        dependencies = [
        
        ]

    ''')

    trainer_filestr = inspect.cleandoc(f"""
        #
        #
        #   Trainer for {title_name}
        #
        #
        
        from {'.' * (len(task.split("/")) + 2)}trainer import Trainer
        
        
        class {trainer_name}(Trainer):

            def __init__(self, *args, **kwargs):
                pass

            def setup(self, data_dir):
                pass

            def train(self):
                pass  # TODO: Mandatory method

            def test(self):
                raise NotImplementedError(f"Trainer {{self.__class__.__name__}} does not support test")

    """)

    predictor_filestr = inspect.cleandoc(f"""
        #
        #
        #   Predictor for {title_name}
        #
        #
        
        from {'.' * (len(task.split("/")) + 2)}predictor import Predictor
        
        
        class {predictor_name}(Predictor):
        
            def __init__(self, *args, **kwargs):
                pass

            def load(self, artifacts_dir):
                pass

            def predict(self, *args, **kwargs):
                pass  # TODO: Mandatory method
    """)

    trainer_conf_filestr = inspect.cleandoc(f'''
        # @package trainer

        _target_: gymnos.{task.replace('/', '.')}.{model}.{trainer_name}
    ''')

    os.makedirs(model_full_path)

    init_fpath = os.path.join(model_full_path, "__init__.py")
    logger.info(f"Creating {init_fpath}")
    with open(init_fpath, "w") as fp:
        fp.write(init_model_filestr + "\n")

    trainer_fpath = os.path.join(model_full_path, "trainer.py")
    logger.info(f"Creating {trainer_fpath}")
    with open(trainer_fpath, "w") as fp:
        fp.write(trainer_filestr + "\n")

    predictor_fpath = os.path.join(model_full_path, "predictor.py")
    logger.info(f"Creating {predictor_fpath}")
    with open(predictor_fpath, "w") as fp:
        fp.write(predictor_filestr + "\n")

    conf_task_dir = os.path.join(conf_dir, "trainer", task_dir)
    os.makedirs(conf_task_dir, exist_ok=True)

    trainer_conf_fpath = os.path.join(conf_task_dir, model + ".yaml")
    logger.info(f"Creating {trainer_conf_fpath}")
    with open(trainer_conf_fpath, "w") as fp:
        fp.write(trainer_conf_filestr)
