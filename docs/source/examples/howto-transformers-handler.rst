###################################
How to create a Transformer
###################################

Transformers are one of the basic building blocks to preprocess data.
The main difference with preprocessors is that transformers need training data to fit and adjust some parameters that are needed when the transformation is run.

The transformer is a ``Scikit-Learn`` ``Transformer`` so it must inherit from ``sklearn.base.TransformerMixin``.
If you want to use the preprocessor in an experiment you must add the transformer location with and id in ``lib.var.transformers.json``, e.g ``mytransformer: lib.transformers.mytransformer.MyTransformer``.

The methods you need to implement are the following:

.. code-block:: python

    def fit(self, X, y=None):
        # fit transformer to training data
        return self

    def transform(self, X):
        ...
        return X
