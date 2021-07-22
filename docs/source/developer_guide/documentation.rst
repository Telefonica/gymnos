.. _documentation:

Documentation
==============================

Documentation is written using `Sphinx <https://www.sphinx-doc.org/en/master/>`_, `RST <https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html>`_ and `NumPy style <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_ for docstrings.

.. contents:: :local:

Build
----------
To build the documentation:

.. prompt:: bash

    pipenv run docs:build

Open ``docs/build/html/index.html`` with your browser to view the documentation.

Live-reload
-------------
To live-reload documentation:

.. prompt:: bash

    pipenv run docs:watch

Open `http://localhost:8000 <http://localhost:8000>`_ with your browser to view the documentation

Clean
----------
To clean build directory:

.. prompt:: bash

    pipenv run docs:clean

Dependencies
-------------

Documentation will be built without model dependencies, make sure you add your dependencies to ``autodock_mock_imports`` in your ``docs/source/conf.py`` file.

More information at `Sphinx documentation <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_mock_imports>`_
