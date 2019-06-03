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

Mixins
------

These days, there's a ton of libraries that help us in our machine learning workflow providing top-performant and tested machine learning models like
TensorFlow, Keras or Sklearn. To help you develop a model in the Gymnos environment, we provide you with `mixins <https://www.ianlewis.org/en/mixins-and-python>`_ for some libraries so you don't have to overwrite some methods.

The following mixins are available:

Keras
===========

.. autoclass:: lib.models.mixins.KerasMixin
    :members:

Sklearn
=============

.. autoclass:: lib.models.mixins.SklearnMixin
    :members:


TensorFlow Saver
================

.. autoclass:: lib.models.mixins.TensorFlowSaverMixin
    :members:
