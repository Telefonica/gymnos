.. _detect_coins_2_experiment:

Detect Coins 2
==============================

.. autoyamldoc:: conf/experiment/detect_coins_2.yaml
    :lineno-start: 1


.. prompt:: bash

    gymnos-train +experiment=detect_coins_2


.. tabs::

   .. tab:: Trainer

        .. autoyaml:: conf/experiment/detect_coins_2.yaml
            :key: trainer
            :caption: :ref:`{defaults[0].override /trainer}`

   .. tab:: Dataset

        .. autoyaml:: conf/experiment/detect_coins_2.yaml
            :key: dataset
            :caption: :ref:`{defaults[1].override /dataset}`
