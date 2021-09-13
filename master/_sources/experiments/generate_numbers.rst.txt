.. _generate_numbers_experiment:

Generate Numbers
==============================

.. autoyamldoc:: conf/experiment/generate_numbers.yaml
    :lineno-start: 1

.. experiment-install:: conf/experiment/generate_numbers.yaml

Usage
**********

.. prompt:: bash

    gymnos-train +experiment=generate_numbers


.. tabs::

   .. tab:: Trainer

        .. autoyaml:: conf/experiment/generate_numbers.yaml
            :key: trainer
            :caption: :ref:`{defaults[0].override /trainer}`

   .. tab:: Dataset

        .. autoyaml:: conf/experiment/generate_numbers.yaml
            :key: dataset
            :caption: :ref:`datasets.{defaults[1].override /dataset}`
