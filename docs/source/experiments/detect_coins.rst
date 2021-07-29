.. _detect_coins_experiment:

Detect Coins
==============================

.. autoyamldoc:: conf/experiment/detect_coins.yaml
    :lineno-start: 1


.. prompt:: bash

    gymnos-train +experiment=detect_coins


.. tabs::

   .. tab:: Trainer

        .. autoyaml:: conf/experiment/detect_coins.yaml
            :key: trainer
            :caption: :ref:`{defaults[0].override /trainer}`

   .. tab:: Dataset

        .. autoyaml:: conf/experiment/detect_coins.yaml
            :key: dataset
            :caption: :ref:`{defaults[1].override /dataset}`
