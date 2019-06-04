#
#
#   Experiment
#
#

import uuid
import logging

logger = logging.getLogger(__name__)


class Experiment:
    """
    Parameters
    ----------
    name: str, optional
        Experiment name
    description: str, optional
        Experiment description
    tags: list of str, optional
        Experiment tags

    Examples
    --------

    .. code-block:: py

        Experiment(
            name="nn_beta_2"
            description= "Solving Imagenet with Neural network ",
            tags=["imagenet", "neural-network", "keras"]
        )
    """

    def __init__(self, name=None, description=None, tags=None):
        if name is None:
            logger.info("Experiment name not assigned, automatically assigning a uuid")
            name = uuid.uuid4().hex
        self.name = name
        self.description = description
        self.tags = tags
