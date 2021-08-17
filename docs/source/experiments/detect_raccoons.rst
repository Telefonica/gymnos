.. _detect_raccoons_experiment:

Detect Raccoons
==============================

.. autoyamldoc:: conf/experiment/detect_raccoons.yaml
    :lineno-start: 1

.. experiment-install:: conf/experiment/detect_raccoons.yaml

Usage
**********

.. prompt:: bash

    gymnos-train +experiment=detect_raccoons


.. tabs::

   .. tab:: Trainer

        .. autoyaml:: conf/experiment/detect_raccoons.yaml
            :key: trainer
            :caption: :ref:`{defaults[0].override /trainer}`

   .. tab:: Dataset

        .. autoyaml:: conf/experiment/detect_raccoons.yaml
            :key: dataset
            :caption: :ref:`datasets.{defaults[1].override /dataset}`
