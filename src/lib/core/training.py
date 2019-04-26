#
#
#   Training
#
#


class Training:
    """
    Parameters
    ----------
    **parameters:
        Any parameter associated with model ``fit`` method.

    Examples
    --------
    .. code-block:: py

        Training(
            epochs=10,
            batch_size=32
        )
    """

    def __init__(self, **parameters):
        self.parameters = parameters
