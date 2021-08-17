.. _learn_snake_experiment:

Learn Snake
==============================

.. autoyamldoc:: conf/experiment/learn_snake.yaml
    :lineno-start: 1

.. experiment-install:: conf/experiment/learn_snake.yaml

Usage
**********

.. prompt:: bash

    gymnos-train +experiment=learn_snake


.. tabs::

   .. tab:: Trainer

        .. autoyaml:: conf/experiment/learn_snake.yaml
            :key: trainer
            :caption: :ref:`{defaults[0].override /trainer}`

   .. tab:: Env

        .. autoyaml:: conf/experiment/learn_snake.yaml
            :key: env
            :caption: :ref:`envs.{defaults[1].override /env}`
