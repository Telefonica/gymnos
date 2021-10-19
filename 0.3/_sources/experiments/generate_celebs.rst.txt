.. _generate_celebs_experiment:

Generate Celebs
==============================

.. autoyamldoc:: conf/experiment/generate_celebs.yaml
    :lineno-start: 1

.. experiment-install:: conf/experiment/generate_celebs.yaml

Usage
**********

.. prompt:: bash

    gymnos-train +experiment=generate_celebs


.. tabs::

   .. tab:: Trainer

        .. autoyaml:: conf/experiment/generate_celebs.yaml
            :key: trainer
            :caption: :ref:`{defaults[0]|override /trainer}`

   .. tab:: Dataset

        .. autoyaml:: conf/experiment/generate_celebs.yaml
            :key: dataset
            :caption: :ref:`datasets.{defaults[1]|override /dataset}`
