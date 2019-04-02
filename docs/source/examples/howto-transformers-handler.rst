###################################
How to create a Transformer
###################################

Transformers are one of the basic building blocks to preprocess data.
The main difference with preprocessors is that transformers need training data to fit and adjust some parameters that are needed when the transformation is run.


The transformer must inherit from ``sklearn.base.TransformerMixin``. The methods you need to implement are the following:

.. code-block:: python

    def fit(self, X, y=None):
        # fit transformer to training data
        return self

    def transform(self, X):
        ...
        return X
