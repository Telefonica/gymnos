###################################
How to create a Model Handler
###################################

To create a model you need to inherit from ``Model`` or its subclasses and implement some methods.
If you want to use the model in an experiment you must add the model location with an id in ``lib.var.models.json``, e.g ``mymodel: lib.models.mymodel.MyModel``.

Base model
===============

This is the parent model that all classes must inherit.
The ``Model`` class is defined in ``lib.models.model`` and you must implement the following methods:

.. code-block:: python

   def fit(self, X, y, batch_size=32, epochs=1, callbacks=None, val_data=None, verbose=1):
      # fit model to train data, it must return a dictionnary with the metrics.

   def predict(self, X, batch_size=32, verbose=0):
      # predict data, it must return a dictionnary with the metrics

   def evaluate(self, X, y, batch_size=32, verbose=0):
      # evaluate model with the input data, it must return a dictionnary with the metrics

   def restore(self, file_path):
      # restore model from checkpoint

   def save(self, directory, name="model"):
      # save model to directory with a name


Keras model
===============

If you want to implement a Keras model, you only need to inherit from ``lib.models.model.KerasModel`` and call constructor with your model.
All methods will be already implemented.

.. code-block:: python

   from .model import KerasModel

   class MyKerasModel(KerasModel):

      def __init__(self, input_shape, **hyperparameters):
         input = layers.Input(shape=input_shape)
         ...
         output = layers.Dense(10, activation="linear")(input)
         mymodel = Model(inputs=[input], outputs=[output])

         super().__init__(input_shape, mymodel)



Scikit-Learn model
====================

If you want to implement a Scikit-Learn model, you only need to inherit from ``lib.models.model.ScikitLearnModel`` and call constructor with your model.
All methods will be already implemented.

.. code-block:: python

   from .keras import ScikitLearnModel

   class MyScikitLearnModel(ScikitLearnModel):

      def __init__(self, input_shape, **hyperparameters):
         lin_reg = LinearRegression(fit_intercept=False)

         super().__init__(input_shape, lin_reg)


