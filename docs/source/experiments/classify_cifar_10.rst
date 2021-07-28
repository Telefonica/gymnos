.. _classify_cifar_10_experiment:

Classify Cifar 10
==============================

.. autoyamldoc:: conf/experiment/classify_cifar_10.yaml
    :lineno-start: 1


.. prompt:: bash

    gymnos-train +experiment=classify_cifar_10


.. tabs::

   .. tab:: Trainer

        .. autoyaml:: conf/experiment/classify_cifar_10.yaml
            :key: trainer
            :caption: :ref:`{defaults[0].override /trainer}`

   .. tab:: Dataset

        .. autoyaml:: conf/experiment/classify_cifar_10.yaml
            :key: dataset
            :caption: :ref:`{defaults[1].override /dataset}`
