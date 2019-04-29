###################################
How to create a Model
###################################

Implementing a model in Gymnos is really simple, just inherit from ``Model`` and overwrite some methods.

.. note::
    The training configuration (:class:`lib.core.model.Model`) will read ``lib.var.models.json`` to find the model given the model's name. If you want to add a model, give it a name and add the model's location.

Model
-----

.. autoclass:: lib.models.model.Model
    :members:
    :inherited-members:

Mixins
------

We provide `mixins <https://www.ianlewis.org/en/mixins-and-python>`_ with default functionality for model methods.

Keras
===========

.. autoclass:: lib.models.mixins.keras.KerasMixin
    :members:

Sklearn
=============

.. autoclass:: lib.models.mixins.sklearn.SklearnMixin
    :members:


TensorFlow
================

.. autoclass:: lib.models.mixins.tensorflow.TensorFlowMixin
    :members:
