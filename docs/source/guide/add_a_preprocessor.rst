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

This command will create ``gymnos/preprocessors/my_preprocessor.py``, and modify ``gymnos/__init__.py`` to register preprocessor so we can load it using ``gymnos.load``.

The preprocessor registration process is done by associating the preprocessor name with their path:

.. code-block:: python
    :caption: gymnos/__init__.py

    preprocessors.register(
        name="my_preprocessor",
        entry_point="gymnos.preprocessors.my_preprocessor.MyPreprocessor"
    )

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

Summary
=============

1. Create ``MyPreprocessor`` in ``gymnos/preprocessor/my_preprocessor.py`` inheriting from :class:`gymnos.preprocessors.preprocessor.preprocessor` and implementing the following methods:

- ``fit(X, y=None)``
- ``fit_generator(generator)`` (optional)
- ``transform(X)``

2. Register the preprocessor in ``gymnos/__init__.py`` by adding:

.. code-block:: python

    preprocessors.register(
        name="my_preprocessor",
        entry_point="gymnos.preprocessors.my_preprocessor.MyPreprocessor"
    )


Adding the preprocessor to ``Telefonica/gymnos``
===================================================

If you'd like to share your work with the community, you can check in your preprocessor implementation to Telefonica/gymnos. Thanks for thinking of contributing!

Before you send your pull request, follow these last few steps (check :ref:`contributing` to see more details):

1. Test preprocessor with any Gymnos dataset
--------------------------------------------------
Check that your preprocessor is working with any Gymnos dataset.

2. Add documentation
----------------------
Add preprocessor documentation.

3. Check your code style
--------------------------
Follow the `PEP8 Python style guide <https://www.python.org/dev/peps/pep-0008/>`_, except Gymnos uses 120 characters as maximum line length.

You can lint files running ``flake8`` command:

.. code-block:: console

    $ flake8

Adding the preprocessor from other repository
=================================================

You can also add a preprocessor from other repository in a very simple way by converting your repository in a Python library.

Once you have defined your ``setup.py``, create and register your Gymnos preprocessors in the same way we have shown.

Here is a minimal example. Say we have our library named ``gymnos_my_preprocessors`` and we want to add the preprocessor ``my_preprocessor``. You have to:

1. Create ``MyPreprocessor`` in ``gymnos_my_preprocessors/my_preprocessor.py`` inheriting from :class:`gymnos.preprocessors.preprocessor.preprocessor` and implementing the abstract methods
2. Register preprocessor in your module ``__init__.py`` referencing the name and the path:

.. code-block:: python
    :caption: gymnos_my_preprocessors/__init__.py

    import gymnos

    gymnos.preprocessors.register(
        name="my_preprocessor",
        entry_point="gymnos_my_preprocessors.my_preprocessor.MyPreprocessor"
    )


That's it, when someone wants to run ``my_preprocessor`` from ``gymnos_my_preprocessors``, simply ``pip install`` the package and reference the package when you are loading the preprocessor with the following format: ``<module_name>:<preprocessor_name>``.

For example:

.. code-block:: python

    gymnos.preprocessors.load("gymnos_my_preprocessors:my_preprocessor")
