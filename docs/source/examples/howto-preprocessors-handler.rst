###################################
How to create a Preprocessor
###################################

Preprocessors are one of the basic building blocks to preprocess data.

The preprocessor must inherit from ``lib.preprocessors.preprocessor.Preprocessor``.
If you want to use the the preprocessor in an experiment you must add the preprocessor location with and id in ``lib.var.preprocessors.json``, e.g ``mypreprocessor: lib.preprocessors.mypreprocessor.MyPreprocessor``.

.. code-block:: python

    def transform(self, X):
        # transform input data
        return X
