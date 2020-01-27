#####################
Lazy Imports
#####################

To make it easier for each developer to use the library with which they feel most comfortable and also keep the size of Gymnos as light as possible, we present below the concept of lazy imports.

Lazy imports are packages that are not installed when you install Gymnos but that will be installed automatically (can be disabled) when a component requires them. In this way we give each developer total flexibility in choosing which library to use and we keep the base version of Gymnos as light as possible.

We must make a distinction between the packages that will be installed as a requirement of Gymnos and the packages that can be used at this time using lazy imports.

The following packages will be installed automatically with Gymnos:

- ``numpy``
- ``pandas``
- ``sklearn``
- ``scipy``
- ``pillow``

The following packages are available via lazy imports:

- ``spacy``: :attr:`gymnos.utils.lazy_imports.lazy_imports.spacy`
- ``comet_ml``: :attr:`gymnos.utils.lazy_imports.lazy_imports.comet_ml`
- ``statsmodels.api``: :attr:`gymnos.utils.lazy_imports.lazy_imports.statsmodels_api`
- ``statsmodels.tsa``: :attr:`gymnos.utils.lazy_imports.lazy_imports.statsmodels_tsa`
- ``mlflow``: :attr:`gymnos.utils.lazy_imports.lazy_imports.mlflow`

To use a package available via lazy imports:

.. code-block:: python

    from utils.lazy_imports import lazy_imports

    spacy = lazy_imports.spacy
    nlp = spacy.load("en_core_web_sm")

.. note::

    By default, if a component needs an extra package and it is not installed, Gymnos will try to install it automatically. If you want to disable this feature, set the environment variable ``GYMNOS_AUTOINSTALL``.

Add a new lazy dependency
===========================

Simply add a property to :class:`gymnos.utils.lazy_imports.LazyImporter` specifying the module to import and the module to ``setup.py``.

For example, if we want to configure PyTorch:

.. code-block:: python

    class LazyImporter:

        @classproperty
        def torch(cls):
            return _try_import("torch")

If the import name is different than the one we will have to install:

.. code-block:: python

    class LazyImporter:

        @classproperty
        def dask_array(cls):
            return _try_import("dask.array", module_to_install="dask[array]")

Import submodules
====================

Some modules require to explicitly import submodules, for example:

If you don't import ``sklearn.linear_model``, this will fail.

.. code-block:: python

    import sklearn

    _ = sklearn.linear_model.LinearRegression()  # AttributeError: module 'sklearn' has no attribute 'linear_model'

You need to explicitly import `linear_model` submodule:

.. code-block:: python

    import sklearn.linear_model

    _ = sklearn.linear_model.LinearRegression()  # ✔️


With lazy imports you can also import submodules with python's ``__import__`` function.
Let's say we want to import ``linear_model`` from ``sklearn``:

.. code-block:: python

    sklearn = __import__(f"{lazy_imports.sklearn}.linear_model")

    _ = sklearn.linear_model.LinearRegression()  # ✔️
