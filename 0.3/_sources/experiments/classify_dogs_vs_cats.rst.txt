.. _classify_dogs_vs_cats_experiment:

Classify dogs vs cats
==============================

.. autoyamldoc:: conf/experiment/classify_dogs_vs_cats.yaml
    :lineno-start: 1


.. experiment-install:: conf/experiment/classify_dogs_vs_cats.yaml


Usage
**********


.. prompt:: bash

    gymnos-train +experiment=classify_dogs_vs_cats


.. tabs::

   .. tab:: Trainer

        .. autoyaml:: conf/experiment/classify_dogs_vs_cats.yaml
            :key: trainer
            :caption: :ref:`{defaults[0]|override /trainer}`

   .. tab:: Dataset

        .. autoyaml:: conf/experiment/classify_dogs_vs_cats.yaml
            :key: dataset
            :caption: :ref:`datasets.{defaults[1]|override /dataset}`
