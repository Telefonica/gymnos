#
#
#   Experiment
#
#


class Experiment:
    """
    Parameters
    ----------
    name: str, optional
        Name of the experiment
    tags: list of str, optional
        Tags of the experiment

    Examples
    --------

    .. code-block:: py

        Experiment(
            name= "Solving Imagenet with Neural network ",
            tags=["imagenet", "neural-network", "keras"]
        )
    """

    def __init__(self, name=None, tags=None):
        self.name = name
        self.tags = tags
