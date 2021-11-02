.. _baby_cry_experiment_experiment:

Baby Cry Experiment
==============================

.. autoyamldoc:: conf/experiment/baby_cry_experiment.yaml
    :lineno-start: 1

.. experiment-install:: conf/experiment/baby_cry_experiment.yaml

Usage
**********

.. prompt:: bash

    gymnos-train +experiment=baby_cry_experiment


.. tabs::

   .. tab:: Trainer

        .. autoyaml:: conf/experiment/baby_cry_experiment.yaml
            :key: trainer
            :caption: :ref:`{defaults[0]|override /trainer}`

   .. tab:: Dataset

        .. autoyaml:: conf/experiment/baby_cry_experiment.yaml
            :key: dataset
            :caption: :ref:`datasets.{defaults[1]|override /dataset}`
