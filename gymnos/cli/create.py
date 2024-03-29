#
#
#   Create model
#
#

import os
import re
import click
import inspect
import stringcase

from rich.tree import Tree
from rich.text import Text
from rich import print as rprint

MAIN_DOMAINS = [
    "vision",
    "generative",
    "audio",
    "nlp",
    "rl",
    "tabular",
]


def _only_letters_numbers_underscores(value):
    pattern = r"^([A-Za-z0-9\_]+)$"
    if not re.match(pattern, value):
        raise click.BadParameter("Only letters, numbers and underscores")

    return value


def _validate_name(ctx, param, value):
    return _only_letters_numbers_underscores(value)


def _validate_domain(ctx, param, value):
    subdomains = value.split("/")

    if subdomains[0] not in MAIN_DOMAINS:
        raise click.BadParameter(f"Main domain must be one of the following: {', '.join(MAIN_DOMAINS)}")

    if len(subdomains) == 1:
        subdomains.append("misc")
    elif len(subdomains) != 2:
        raise click.BadParameter("Must has the following structure: <DOMAIN>/<SUBDOMAIN>, "
                                 "e.g vision/image_classification")

    _only_letters_numbers_underscores(subdomains[1])

    return subdomains


@click.group()
def main():
    pass


@main.command()
@click.argument("name", callback=_validate_name)
@click.argument("domain", callback=_validate_domain)
def model(name, domain):
    """
    Create new model.

    The DOMAIN is the task solved by the model, it can be a main domain (e.g ``audio``, ``vision``) or
    a main domain with a subdomain (e.g ``vision/image_classification``, ``text/question_answering``).
    """
    title = stringcase.titlecase(name)
    classname = stringcase.pascalcase(name)
    hydra_conf_classname = classname + "HydraConf"
    trainer_classname = classname + "Trainer"
    predictor_classname = classname + "Predictor"

    __init__template = inspect.cleandoc(f"""
        \"""
        TODO: Docstring for {title}
        \"""

        from ....utils import lazy_import

        # Public API
        {predictor_classname} = lazy_import("gymnos.{domain[0]}.{domain[1]}.{name}.predictor.{predictor_classname}")
    """) + "\n"

    __model__template = inspect.cleandoc(f"""
        #
        #
        #   Model
        #
        #

        from .hydra_conf import {hydra_conf_classname}

        hydra_conf = {hydra_conf_classname}

        pip_dependencies = []

        apt_dependencies = []
    """) + "\n"

    trainer_template = inspect.cleandoc(f"""
        #
        #
        #   Trainer
        #
        #

        from dataclasses import dataclass

        from ....base import BaseTrainer
        from .hydra_conf import {hydra_conf_classname}


        @dataclass
        class {trainer_classname}({hydra_conf_classname}, BaseTrainer):
            \"""
            TODO: docstring for trainer
            \"""

            def prepare_data(self, root):
                pass   # OPTIONAL: do anything with your data

            def train(self):
                pass   # TODO: training code

            def test(self):
                pass   # OPTIONAL: test code
    """) + "\n"

    rl_trainer_template = inspect.cleandoc(f"""
        #
        #
        #   Trainer
        #
        #

        from dataclasses import dataclass

        from ....base import BaseRLTrainer
        from .hydra_conf import {hydra_conf_classname}


        @dataclass
        class {trainer_classname}({hydra_conf_classname}, BaseRLTrainer):
            \"""
            TODO: docstring for trainer
            \"""

            def prepare_env(self, env_id):
                pass   # OPTIONAL: prepare environment, use gym.make(env_id) to create it

            def train(self):
                pass   # TODO: training code

            def test(self):
                pass   # OPTIONAL: test code
    """) + "\n"

    predictor_template = inspect.cleandoc(f"""
        #
        #
        #   Predictor
        #
        #

        from omegaconf import DictConfig

        from ....base import BasePredictor, MLFlowRun


        class {predictor_classname}(BasePredictor):
            \"""
            TODO: docstring for predictor
            \"""

            def load(self, config: DictConfig, run: MLFlowRun, artifacts_dir: str):
                pass   # OPTIONAL: load model from MLFlow artifacts directory

            def predict(self, *args, **kwargs):
                pass   # TODO: prediction code. Define parameters
    """) + "\n"

    hydra_conf_template = inspect.cleandoc(f"""
        #
        #
        #   {title} Hydra configuration
        #
        #

        from dataclasses import dataclass, field


        @dataclass
        class {hydra_conf_classname}:

            # TODO: define trainer parameters

            _target_: str = field(init=False, repr=False, default="gymnos.{domain[0]}.{domain[1]}.{name}."
                                                                  "trainer.{trainer_classname}")
    """) + "\n"

    docs_template = inspect.cleandoc(f"""
        .. _{domain[0]}.{domain[1]}.{name}:

        {title}
        {"=" * len(title)}

        .. automodule:: gymnos.{domain[0]}.{domain[1]}.{name}

        .. prompt:: bash

            pip install gymnos[{domain[0]}.{domain[1]}.{name}]

        .. contents::
            :local:

        .. _{domain[0]}.{domain[1]}.{name}__trainer:

        Trainer
        *********

        .. prompt:: bash

            gymnos-train trainer={domain[0]}.{domain[1]}.{name}

        .. rst-class:: gymnos-hydra

            .. autoclass:: gymnos.{domain[0]}.{domain[1]}.{name}.trainer.{trainer_classname}
                :inherited-members:


        .. _{domain[0]}.{domain[1]}.{name}__predictor:

        Predictor
        ***********

        .. code-block:: py

            from gymnos.{domain[0]}.{domain[1]}.{name} import {predictor_classname}

            {predictor_classname}.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

        .. autoclass:: gymnos.{domain[0]}.{domain[1]}.{name}.predictor.{predictor_classname}
           :members:
    """) + "\n"

    if domain[1] == "misc":
        subdomain_title = "Miscellaneous"
    else:
        subdomain_title = stringcase.titlecase(domain[1])

    docs_subdomain_template = inspect.cleandoc(f"""
        .. _{domain[0]}__{domain[1]}:

        {subdomain_title}
        {"=" * len(subdomain_title)}

        .. automodule:: gymnos.{domain[0]}.{domain[1]}

        .. toctree::
            :glob:
            :maxdepth: 1

            *
    """) + "\n"

    subdomain__init__template = inspect.cleandoc(f"""
        \"""
        Models for {subdomain_title}
        \"""
    """) + "\n"

    subdomain_dir = os.path.join("gymnos", domain[0], domain[1])
    docs_subdomain_dir = os.path.join("docs", "source", domain[0], domain[1])

    if not os.path.isdir(docs_subdomain_dir):
        os.makedirs(docs_subdomain_dir)
        with open(os.path.join(docs_subdomain_dir, "index.rst"), "w") as fp:
            fp.write(docs_subdomain_template)

    if not os.path.isdir(subdomain_dir):
        os.makedirs(subdomain_dir)
        with open(os.path.join(subdomain_dir, "__init__.py"), "w") as fp:
            fp.write(subdomain__init__template)

    docs_dir = os.path.join("docs", "source", domain[0], domain[1])
    model_dir = os.path.join("gymnos", domain[0], domain[1], name)

    os.makedirs(model_dir)
    os.makedirs(docs_dir, exist_ok=True)

    with open(os.path.join(docs_dir, name + ".rst"), "w") as fp:
        fp.write(docs_template)

    with open(os.path.join(model_dir, "__init__.py"), "w") as fp:
        fp.write(__init__template)

    with open(os.path.join(model_dir, "__model__.py"), "w") as fp:
        fp.write(__model__template)

    with open(os.path.join(model_dir, "trainer.py"), "w") as fp:
        if domain[0] == "rl":
            fp.write(rl_trainer_template)
        else:
            fp.write(trainer_template)

    with open(os.path.join(model_dir, "predictor.py"), "w") as fp:
        fp.write(predictor_template)

    with open(os.path.join(model_dir, "hydra_conf.py"), "w") as fp:
        fp.write(hydra_conf_template)

    rprint("The following files have been created: ")

    tree = Tree(":open_file_folder:", guide_style="dim")
    gymnos_tree = tree.add("gymnos")
    domain_tree = gymnos_tree.add(domain[0])
    subdomain_tree = domain_tree.add(domain[1])
    model_tree = subdomain_tree.add(Text(name, "bold blue"))
    for fname in ("__init__.py", "__model__.py", "trainer.py", "predictor.py", "hydra_conf.py"):
        model_tree.add(Text(f"📄 {fname}", "bold blue"))

    docs_tree = tree.add("docs")
    source_tree = docs_tree.add("source")
    domain_tree = source_tree.add(domain[0])
    subdomain_tree = domain_tree.add(domain[1])
    subdomain_tree.add(Text(name + ".rst", "bold blue"))

    rprint(tree)


@main.command()
@click.argument("name", callback=_validate_name)
def dataset(name):
    """
    Create new dataset
    """
    title = stringcase.titlecase(name)

    __init__template = inspect.cleandoc(f"""
        \"""
        TODO: Docstring for {title}
        \"""
    """) + "\n"

    classname = stringcase.pascalcase(name)
    conf_classname = classname + "HydraConf"

    __dataset__template = inspect.cleandoc(f"""
        #
        #
        #   {title} gymnos conf
        #
        #

        from .hydra_conf import {conf_classname}

        hydra_conf = {conf_classname}

        pip_dependencies = []

        apt_dependencies = []
    """) + "\n"

    hydra_conf_template = inspect.cleandoc(f"""
        #
        #
        #   {title} Hydra conf
        #
        #

        from dataclasses import dataclass, field


        @dataclass
        class {conf_classname}:

            # TODO: add custom parameters

            _target_: str = field(init=False, repr=False, default="gymnos.datasets.{name}.dataset.{classname}")
    """) + "\n"

    dataset_template = inspect.cleandoc(f"""
        #
        #
        #   {title} dataset
        #
        #

        from ...base import BaseDataset
        from .hydra_conf import {conf_classname}

        from dataclasses import dataclass


        @dataclass
        class {classname}({conf_classname}, BaseDataset):
            \"""
            TODO: description about data structure

            Parameters
            -----------
            TODO: description of each parameter
            \"""

            def download(self, root):
                pass  # TODO: save dataset files to `root`
    """) + "\n"

    docs_template = inspect.cleandoc(f"""
        .. _datasets.{name}:

        {title}
        {"=" * len(title)}

        .. automodule:: gymnos.datasets.{name}

        .. prompt:: bash

            pip install gymnos[datasets.{name}]

        Usage
        ***********

        .. prompt:: bash

            gymnos-train dataset={name}

        .. rst-class:: gymnos-hydra

            .. autoclass:: gymnos.datasets.{name}.dataset.{classname}
    """) + "\n"

    docs_dir = os.path.join("docs", "source", "datasets")
    dataset_dir = os.path.join("gymnos", "datasets", name)

    os.makedirs(dataset_dir)
    os.makedirs(docs_dir, exist_ok=True)

    with open(os.path.join(dataset_dir, "__init__.py"), "w") as fp:
        fp.write(__init__template)

    with open(os.path.join(dataset_dir, "__dataset__.py"), "w") as fp:
        fp.write(__dataset__template)

    with open(os.path.join(dataset_dir, "dataset.py"), "w") as fp:
        fp.write(dataset_template)

    with open(os.path.join(dataset_dir, "hydra_conf.py"), "w") as fp:
        fp.write(hydra_conf_template)

    with open(os.path.join(docs_dir, f"{name}.rst"), "w") as fp:
        fp.write(docs_template)

    rprint("The following files have been created: ")

    tree = Tree(":open_file_folder:", guide_style="dim")
    gymnos_tree = tree.add("gymnos")
    datasets_tree = gymnos_tree.add("datasets")
    dataset_tree = datasets_tree.add(Text(name, "bold blue"))
    for fname in ("__init__.py", "__dataset__.py", "dataset.py", "hydra_conf.py"):
        dataset_tree.add(Text(f"📄 {fname}", "bold blue"))

    docs_tree = tree.add("docs")
    source_tree = docs_tree.add("source")
    datasets_tree = source_tree.add("datasets")
    datasets_tree.add(Text(name + ".rst", "bold blue"))

    rprint(tree)


@main.command()
@click.argument("name", callback=_validate_name)
@click.option("--rl", help="Whether or not is an RL experiment", is_flag=True)
def experiment(name, rl):
    """
    Create new experiment
    """
    title = stringcase.titlecase(name)

    template = inspect.cleandoc("""
        # @package _global_
        # TODO: description about experiment

        defaults:
            - override /trainer: <trainer_name>  # TODO: set name of trainer to use
            - override /dataset: <dataset_name>  # TODO: set name of dataset to use

        trainer:
            <param>: <value>   # TODO: override default trainer params

        dataset:
            <param>: <value>  # TODO: override default dataset params
    """) + "\n"

    rl_template = inspect.cleandoc("""
        # @package _global_
        # TODO: description about experiment

        defaults:
            - override /trainer: <trainer_name>  # TODO: set name of trainer to use
            - override /env: <env_name>  # TODO: set name of env to use

        trainer:
            <param>: <value>   # TODO: override default trainer params

        env:
            <param>: <value>  # TODO: override default env params
    """) + "\n"

    docs_template = inspect.cleandoc(f"""
        .. _{name}_experiment:

        {title}
        ==============================

        .. autoyamldoc:: conf/experiment/{name}.yaml
            :lineno-start: 1

        .. experiment-install:: conf/experiment/{name}.yaml

        Usage
        **********

        .. prompt:: bash

            gymnos-train +experiment={name}


        .. tabs::

           .. tab:: Trainer

                .. autoyaml:: conf/experiment/{name}.yaml
                    :key: trainer
                    :caption: :ref:`{{defaults[0]|override /trainer}}`

           .. tab:: Dataset

                .. autoyaml:: conf/experiment/{name}.yaml
                    :key: dataset
                    :caption: :ref:`datasets.{{defaults[1]|override /dataset}}`

    """) + "\n"

    rl_docs_template = inspect.cleandoc(f"""
        .. _{name}_experiment:

        {title}
        ==============================

        .. autoyamldoc:: conf/experiment/{name}.yaml
            :lineno-start: 1

        .. experiment-install:: conf/experiment/{name}.yaml

        Usage
        **********

        .. prompt:: bash

            gymnos-train +experiment={name}


        .. tabs::

           .. tab:: Trainer

                .. autoyaml:: conf/experiment/{name}.yaml
                    :key: trainer
                    :caption: :ref:`{{defaults[0]|override /trainer}}`

           .. tab:: Env

                .. autoyaml:: conf/experiment/{name}.yaml
                    :key: env
                    :caption: :ref:`envs.{{defaults[1]|override /env}}`

    """) + "\n"

    fpath = os.path.join("conf", "experiment", name + ".yaml")

    if os.path.isfile(fpath):
        raise FileExistsError(f"Experiment {fpath} already exists")

    with open(fpath, "w") as fp:
        if rl:
            fp.write(rl_template)
        else:
            fp.write(template)

    docs_dir = os.path.join("docs", "source", "experiments")
    docs_fpath = os.path.join(docs_dir, name + ".rst")

    os.makedirs(docs_dir, exist_ok=True)

    if os.path.isfile(docs_fpath):
        raise FileExistsError(f"Docs {docs_fpath} already exists")

    with open(docs_fpath, "w") as fp:
        if rl:
            fp.write(rl_docs_template)
        else:
            fp.write(docs_template)

    rprint("The following files have been created: ")

    tree = Tree(":open_file_folder:", guide_style="dim")
    conf_tree = tree.add("conf")
    experiment_tree = conf_tree.add("experiment")
    experiment_tree.add(Text(f"📄 {name}.yaml", "bold blue"))

    docs_tree = tree.add("docs")
    docs_source_tree = docs_tree.add("source")
    docs_experiments_tree = docs_source_tree.add("experiments")
    docs_experiments_tree.add(Text(f"📄 {name}.rst", "bold blue"))

    rprint(tree)


@main.command()
@click.argument("name", callback=_validate_name)
def env(name):
    title = stringcase.titlecase(name)

    classname = stringcase.pascalcase(name)
    conf_classname = classname + "HydraConf"

    __init__template = inspect.cleandoc(f"""
        \"""
        TODO: Docstring for {title}
        \"""

        from ...utils import lazy_import

        # Public API
        {classname} = lazy_import("gymnos.envs.{name}.env.{classname}")
    """) + "\n"

    __env__template = inspect.cleandoc(f"""
        #
        #
        #   {title} gymnos conf
        #
        #

        from .hydra_conf import {conf_classname}

        hydra_conf = {conf_classname}

        reward_threshold = None  # The reward threshold before the task is considered solved
        nondeterministic = False  # Whether this environment is non-deterministic even after seeding
        max_episode_steps = None  # The maximum number of steps that an episode can consist of

        pip_dependencies = []

        apt_dependencies = []
    """) + "\n"

    hydra_conf_template = inspect.cleandoc(f"""
        #
        #
        #   {title} Hydra conf
        #
        #

        from dataclasses import dataclass, field


        @dataclass
        class {conf_classname}:

            # TODO: define env parameters

            _target_: str = field(init=False, repr=False, default="gymnos.envs.{name}.env.{classname}")
    """) + "\n"

    env_template = inspect.cleandoc(f"""
        #
        #
        #   {title} env
        #
        #

        import gym

        from dataclasses import dataclass

        from .hydra_conf import {conf_classname}


        @dataclass
        class {classname}({conf_classname}, gym.Env):

            metadata = {{"render.modes": ["human"]}}

            @property
            def action_space(self) -> gym.spaces.Space:
                pass  # TODO: define action_space

            @property
            def observation_space(self) -> gym.spaces.Space:
                pass  # TODO: define observation_space

            def step(self, action):
                pass  # TODO: Execute one time step within the environment

            def reset(self):
                pass  # TODO: Reset the state of the environment to an initial state

            def render(self, mode="human", close=False):
                pass  # TODO: Render the environment to the screen
    """) + "\n"

    docs_template = inspect.cleandoc(f"""
        .. _envs.{name}:

        {title}
        {"=" * len(title)}

        .. automodule:: gymnos.envs.{name}

        .. prompt:: bash

            pip install gymnos[envs.{name}]

        Usage
        ***********

        .. prompt:: bash

            gymnos-train env={name}

        .. rst-class:: gymnos-hydra

            .. autoclass:: gymnos.envs.{name}.env.{classname}
    """) + "\n"

    docs_dir = os.path.join("docs", "source", "envs")
    env_dir = os.path.join("gymnos", "envs", name)

    os.makedirs(env_dir)
    os.makedirs(docs_dir, exist_ok=True)

    with open(os.path.join(env_dir, "__init__.py"), "w") as fp:
        fp.write(__init__template)

    with open(os.path.join(env_dir, "__env__.py"), "w") as fp:
        fp.write(__env__template)

    with open(os.path.join(env_dir, "env.py"), "w") as fp:
        fp.write(env_template)

    with open(os.path.join(env_dir, "hydra_conf.py"), "w") as fp:
        fp.write(hydra_conf_template)

    with open(os.path.join(docs_dir, f"{name}.rst"), "w") as fp:
        fp.write(docs_template)

    tree = Tree(":open_file_folder:", guide_style="dim")
    gymnos_tree = tree.add("gymnos")
    datasets_tree = gymnos_tree.add("envs")
    dataset_tree = datasets_tree.add(Text(name, "bold blue"))
    for fname in ("__init__.py", "__env__.py", "env.py", "hydra_conf.py"):
        dataset_tree.add(Text(f"📄 {fname}", "bold blue"))

    docs_tree = tree.add("docs")
    source_tree = docs_tree.add("source")
    datasets_tree = source_tree.add("envs")
    datasets_tree.add(Text(name + ".rst", "bold blue"))

    rprint(tree)
