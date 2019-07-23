###################################
How to create a Model
###################################

Implementing a model in Gymnos is really simple, just inherit from ``Model`` and overwrite some methods.

.. note::
    The training configuration (:class:`gymnos.core.model.Model`) will read ``gymnos.var.models.json`` to find the model given the model's name. If you want to add a model, give it a name and add the model's location.

Model
-----

.. autoclass:: gymnos.models.model.Model
    :members:

Mixins
------

These days, there's a ton of libraries that help us in our machine learning workflow providing top-performant and tested machine learning models like
TensorFlow, Keras or Sklearn. To help you develop a model in the Gymnos environment, we provide you with `mixins <https://www.ianlewis.org/en/mixins-and-python>`_ for some libraries so you don't have to overwrite some methods.

The following mixins are available:

Keras Classifier
=================

.. autoclass:: gymnos.models.mixins.KerasClassifierMixin
    :members:

Keras Regressor
=================

.. autoclass:: gymnos.models.mixins.KerasRegressorMixin
    :members:

Sklearn
=============

.. autoclass:: gymnos.models.mixins.SklearnMixin
    :members:


TensorFlow Saver
================

.. autoclass:: gymnos.models.mixins.TensorFlowSaverMixin
    :members:
