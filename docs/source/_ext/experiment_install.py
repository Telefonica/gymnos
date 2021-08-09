#
#
#   Experiment install
#
#

import inspect

from omegaconf import OmegaConf
from sphinx.util.docutils import SphinxDirective

from utils import convert_rst_to_nodes


class ExperimentInstallDirective(SphinxDirective):

    required_arguments = 1
    has_content = False
    option_spec = {}

    def run(self):
        yaml_path = self.arguments[0]

        config = OmegaConf.load(yaml_path)

        trainer_mod = config["defaults"][0]["override /trainer"]
        dataset_mod = config["defaults"][1].get("override /dataset")
        env_mod = config["defaults"][1].get("override /env")

        extras = [trainer_mod]

        if dataset_mod is not None:
            extras.append("datasets." + dataset_mod)
        if env_mod is not None:
            extras.append("envs." + env_mod)

        extras_str = ",".join(extras)

        rst_content = inspect.cleandoc(f"""
        .. prompt:: bash
        
            pip install gymnos[{extras_str}]
        """)

        node = convert_rst_to_nodes(self, rst_content)

        return node


def setup(app):
    app.add_directive("experiment-install", ExperimentInstallDirective)
