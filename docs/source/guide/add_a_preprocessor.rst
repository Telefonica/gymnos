####################
Add a preprocessor
####################

Overview
=========

Machie Learning preprocessors are distributed in all kinds of API specifications and in all kinds of places, and they're not always implemented in a format that's ready to feed into a machine learning pipeline. Enter Gymnos preprocessors.

Gymnos preprocessors provides a way to transform all those models into a standard format to make them ready for a machine learning pipeline.

To enable this, each preprocessor implements a subclass of :class:`gymnos.preprocessors.Preprocessor`, which specifies:

* How to train preprocessor
* How transform samples

Writing ``my_preprocessor.py``
===============================

Use the default template
-------------------------
If you want to :ref:`contribute to our repo <contributing>` and add a new preprocessor, the following script will help you get started generating the required python files. To use it, clone the `Gymnos <https://github.com/Telefonica/gymnos>`_ repository and run the following command:

.. code-block:: console

  $ python3 -m scripts.create_new preprocessor --name my_preprocessor

This command will create ``gymnos/preprocessor/my_preprocessor.py`` and modify ``gymnos/var/preprocessors.json`` to reference preprocessor name with their location so we can load it using ``gymnos.load``.

Go to ``gymnos/preprocessors/my_preprocessor.py`` and then search for TODO(my_preprocessor) in the generated file to do the modifications.

Preprocessor
------------
Each preprocessor is defined as a subclass of :class:`gymnos.preprocessors.Preprocessor` implementing the following methods:

* ``fit``: fits preprocessor to training data
* ``transform``: performs preprocessing to input samples

Because of the different nature of each preprocessor, the following methods are not mandatory and you should only implement them if your preprocessor allows it:

* ``fit_generator``: fits the preprocessor on data generated batch-by-batch by a Python sequence / generator.

my_preprocessor.py
-------------------

``my_preprocessor.py`` first look like this:

.. code-block:: python

    #
    #
    #   MyPreprocessor
    #
    #

    from .preprocessor import Preprocessor

    class MyPreprocessor(Preprocessor):
        """
        TODO(my_preprocessor): Description of my preprocessor.
        """

        def __init__(self, **parameters):
            # TODO(my_preprocessor): Define and initialize model parameters

        def fit(self, X, y=None):
            # TODO(my_preprocessor): Fit preprocessor to training data.

        def fit_generator(self, generator):
            # {OPTIONAL}: Fit preprocessor to training generator. Only if your preprocessor supports incremental learning
            raise NotImplementedError()

        def transform(self, X):
            # TODO(my_preprocessor): Preprocess data


Specifying ``parameters``
===========================
Use the constructor to specify any parameters you need to build your model. These parameters may be required or optional although optional parameters are preferable.

.. code-block:: python

    class MyPreprocessor(Preprocessor):

        def __init__(self, lowercase=True, language="english"):
            self.lowercase = lowercase
            self.language = language

Training preprocessor
=======================

Fit preprocessor to training data specifying any parameters you need to train your preprocessor. Optional parameters are preferable.

It returns ``self`` for chaining purposes.

.. code-block:: python

    def fit(self, X, y=None):
        ...
        return self

Transforming samples
=======================

Preprocess input samples.

It returns the preprocessed samples.

.. code-block:: python

    def transform(self, X):
        ...
        return X_t

Adding the preprocessor to ``Telefonica/gymnos``
===================================================

If you'd like to share your work with the community, you can check in your preprocessor implementation to Telefonica/gymnos. Thanks for thinking of contributing!

Before you send your pull request, follow these last few steps (check :ref:`contributing` to see more details):

1. Test preprocessor with any Gymnos dataset
-----------------------------------------------
Check that your preprocessor is working with a Gymnos dataset.

2. Add documentation
----------------------
Add preprocessor documentation.

3. Check your code style
--------------------------
Follow the `PEP8 Python style guide <https://www.python.org/dev/peps/pep-0008/>`_, except Gymnos uses 120 characters as maximum line length.

You can lint files running ``flake8`` command:

.. code-block:: console

    $ flake8
