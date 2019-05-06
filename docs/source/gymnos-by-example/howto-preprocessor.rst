###################################
How to create a Preprocessor
###################################

Implementing a preprocessor in Gymnos is really simple, just inherit from ``Preprocessor`` and override some methods.

.. note::
    The training configuration (:class:`lib.core.dataset.Dataset`) will read ``lib.var.preprocessors.json`` to find the preprocessor given the preprocessor's type. If you want to add a preprocessor, give it a name and add the prepocessor's location.

.. autoclass:: lib.preprocessors.preprocessor.Preprocessor
    :members:
