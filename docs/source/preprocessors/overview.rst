#################
Preprocessors
#################

Gymnos preprocessors are a collection of preprocessors with a common API allowing their use in a pipeline of a supervised learning system. All preprocessors inherit from :class:`gymnos.preprocessors.preprocessor.Preprocessor`.

Usage
*******
.. code-block:: python

    preprocessor = gymnos.preprocessors.load("image_resize", width=80, height=80)

    preprocessor.fit(X, y)

    X_t = preprocessor.transform(X)

If you want to use multiple preprocessors, take a look to Pipeline:

.. code-block:: python

    preprocessor_1 = gymnos.preprocessors.load("divide", factor=255)
    preprocessor_2 = gymnos.preprocessors.load("grayscale")

    pipeline = gymnos.preprocessors.preprocessor.Pipeline(
        preprocessor_1, 
        preprocessor_2
    )

    pipeline.fit(X, y)  # fit pipeline chaining preprocessors

    X_t = pipeline.transform(X)  # transform samples chaining preprocessors

All Preprocessors
******************

.. toctree::
   :maxdepth: 2

   ./standard
   ./image
   ./text
