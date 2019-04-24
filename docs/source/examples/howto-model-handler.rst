###################################
How to create a Model Handler
###################################

All models must follow a protocol with some methods to implement.
First, you need to inherit from ``Model`` defined in ``lib.models.model``. Then, you need to implement the following methods:

.. code-block:: python

   def fit(self, X, y, **kwargs):
      """
      Method to fit the model to training data.
      Arguments:
         - X: Data features, type: NumPy array or Pandas DataFrame/Series
         - y: Data labels, type: NumPy array or Pandas DataFrame/Series
      Returns:
         - metrics: Training metrics, type: dictionnary
      """

   def predict(self, X):
      """
      Method to return predictions for data X
      Arguments:
         - X: Data features, type: NumPy array or Pandas DataFrame/Series
      Returns:
         - predictions: Predictions from features, type: NumPy array with predictions
      """

   def evaluate(self, X, y):
      """
      Method to evaluate performance of the model
      Arguments:
         - X: Data features, type: NumPy array or Pandas DataFrame/Series
         - y: Data labels, type: NumPy array or Pandas DataFrame/Series
      Returns:
         - metrics: Testing metrics, type: dictionnary
      """

   def restore(self, directory):
      """
      Method to restore model saved by save method
      Arguments:
         - directory: directory with artifacts saved by model, type: string
      """

   def save(self, directory):
      """
      Method to save artifacts needed to restore model later.
      Arguments:
         - directory: directory with artifacts saved by model, type: string
      """

If you want to use the model in an experiment you must add the model location with an id in ``lib.var.models.json``, e.g ``mymodel: lib.models.mymodel.MyModel``.

Mixins
########

We provide `mixins <https://www.ianlewis.org/en/mixins-and-python>`_ with default functionality for model methods. The currently available mixins are defined in ``lib.models.mixins`` directory.

Keras mixin
===========

Mixin for sequential and functional keras models. It provides implementation for ``fit``, ``predict``, ``evaluate``, ``restore`` and ``save`` methods. You need to define the variable ``self.model`` with your compiled keras model.

For example:

.. code-block:: python
   :emphasize-lines: 1, 4
   
   class MyKerasModel(Model, KerasMixin):

      def __init__(self):
         self.model = keras.models.Sequential([
            ...
         ])
         self.model.compile(...)


Sklearn mixin
=============

Mixin for sklearn models. It provides implementation for ``fit``, ``predict``, ``evaluate``, ``restore`` and ``save`` methods. You need to define the variable ``self.model`` with your Sklearn model.

For example:

.. code-block:: python
   :emphasize-lines: 1, 4

   class MySklearnModel(Model, SklearnMixin):

      def __init__(self):
         self.model = sklearn.linear_model.LinearRegression(...)


TensorFlow mixin
================

Mixin for TensorFlow models. It provides implementation for ``save`` and ``restore`` methods. You need to define the variable ``self.sess`` with your TensorFlow session.

For example:

.. code-block:: python
   :emphasize-lines: 1, 5

   class MyTensorFlowModel(Model, TensorFlowMixin):

      def __init__(self):
         ...
         self.sess = tf.Session(...)

      def fit(self, X, y):
         ...

      def predict(self, X):
         ...

      def evaluate(self, X, y):
         ...
