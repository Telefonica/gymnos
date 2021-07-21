Installation
************

Prerequisites
===============

Gymnos requires Python 3.7+.

Stable release
================

.. code-block:: console

   $ pip install gymnos  # FIXME: package not uploaded to pypi

Bleeding-edge version
=======================

.. code-block:: console

   $ pip install git+https://github.com/Telefonica/gymnos.git@master --upgrade


Install from source
======================

.. code-block:: console

    $ git clone https://github.com/Telefonica/gymnos.git
    $ cd gymnos
    $ pip install -e .


Development
=====================

We provide a ``Pipfile`` with all dependencies resolved for the project (including dev dependencies).
First obtain `Pipenv <https://pipenv.pypa.io/en/latest>`_ and then install Gymnos with development dependencies:

.. code-block:: console

    $ git clone https://github.com/Telefonica/gymnos.git
    $ cd gymnos
    $ pipenv install --dev

Refer to `Pipenv documentation <https://pipenv.pypa.io/en/latest/install/>`_ for more information.
