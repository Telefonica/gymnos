###################################
How to create a Preprocessor
###################################

Preprocessors are the building blocks to transform data to prepare data for model.

All preprocessors must follow a protocol with some methods to implement.
First, you need to need to inherit from ``Preprocessor`` defined in ``lib.preprocessors.preprocessor``. Then you need to implement the following methods:

.. code-block:: python

    def fit(self, X, y=None):
        """
        Optional method to fit the preprocessor to training data. 
        Useful if you need to compute some variable from training data.
        Arguments:
            - X: Data features, type: type: NumPy array or Pandas DataFrame/Series
            - y: Data labels, type: NumPy array or Pandas DataFrame/Series
        Returns:
            - self: instance of the preprocessor
        """

    def transform(self, X):
        """
        Method to transform data.
        Arguments:
            - X: Data features, type: type: NumPy array or Pandas DataFrame/Series
        Returns:
            - X_t: Transformed data, type: NumPy array or Pandas DataFrame/Series
        """

If you want to use the preprocessor in an experiment you must add the preprocessor location with an id in ``lib.var.preprocessors.json``, e.g ``mypreprocessor: lib.models.mypreprocessor.MyPreprocessor``.
