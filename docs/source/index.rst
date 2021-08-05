.. gymnos documentation master file, created by
   sphinx-quickstart on Tue Jun 29 12:13:13 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Gymnos
==================================

.. include:: introduction.rst

.. toctree::
   :maxdepth: 1
   :name: start
   :caption: Get started

   installation
   quickstart

.. toctree::
   :name: datasets
   :caption: Datasets
   :glob:

   datasets/*

.. misctoctree::
   :maxdepth: 2
   :name: vision
   :caption: Vision
   :glob:

   vision/*/index

.. toctree::
   :maxdepth: 2
   :name: audio
   :caption: Audio
   :glob:

   audio/*/index

.. toctree::
   :maxdepth: 2
   :name: generative
   :caption: Generative
   :glob:

   generative/*/index

.. toctree::
   :maxdepth: 2
   :name: tabular
   :caption: Tabular
   :glob:

   tabular/*/index

.. toctree::
   :maxdepth: 2
   :name: nlp
   :caption: Natural Language Processing
   :glob:

   nlp/*/index

.. toctree::
   :maxdepth: 2
   :name: rl
   :caption: Reinforcement Learning
   :glob:

   rl/*/index

.. toctree::
   :maxdepth: 1
   :name: cli
   :caption: CLI

   cli/login
   cli/train
   cli/upload
   cli/create

.. toctree::
   :maxdepth: 2
   :name: launchers
   :caption: Launchers

   launchers/sofia


.. toctree::
   :name: experiments
   :caption: Experiments
   :glob:

   experiments/*


.. toctree::
   :maxdepth: 2
   :name: services
   :caption: Services

   services/sofia

.. toctree::
    :maxdepth: 2
    :name: utils
    :caption: Utils
    :glob:

    utils/*

.. toctree::
   :maxdepth: 1
   :name: developer_guide
   :caption: Developer Guide

   developer_guide/style_guide
   developer_guide/documentation
   developer_guide/new_experiment
   developer_guide/new_dataset
   developer_guide/new_model


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
