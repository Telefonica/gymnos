Installation
************

Prerequisites
===============

Gymnos requires Python 3.7+.

Stable release
================

.. prompt:: bash
   :substitutions:

   pip install git+https://github.com/Telefonica/gymnos.git@|release|


Bleeding-edge version
=======================

.. prompt:: bash

   pip install git+https://github.com/Telefonica/gymnos.git@master --upgrade


Install from source
======================

.. prompt:: bash

    git clone https://github.com/Telefonica/gymnos.git
    cd gymnos
    pip install -e .


Development
=====================

We provide a ``Pipfile`` with all dependencies resolved for the project (including dev dependencies).
First obtain `Pipenv <https://pipenv.pypa.io/en/latest>`_ and then install Gymnos with development dependencies:

.. prompt:: bash

    git clone https://github.com/Telefonica/gymnos.git
    cd gymnos
    pipenv install --dev

Refer to `Pipenv documentation <https://pipenv.pypa.io/en/latest/install/>`_ for more information.
