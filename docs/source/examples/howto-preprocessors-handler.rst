###################################
How to create a Preprocessor
###################################

Preprocessors are one of the basic building blocks to preprocess data.

The preprocessor must inherit from ``lib.preprocessors.preprocessor.Preprocessor`` and implement the following methods:

.. code-block:: python

    def transform(self, X):
        # transform input data
        return X
