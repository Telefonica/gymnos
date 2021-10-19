.. _learn_pong_experiment:

Learn Cartpole
==============================

.. autoyamldoc:: conf/experiment/learn_pong.yaml
    :lineno-start: 1

.. experiment-install:: conf/experiment/learn_pong.yaml

Usage
**********

.. prompt:: bash

    gymnos-train +experiment=learn_pong


.. tabs::

   .. tab:: Trainer

        .. autoyaml:: conf/experiment/learn_pong.yaml
            :key: trainer
            :caption: :ref:`{defaults[0]|override /trainer}`

   .. tab:: Env

        .. autoyaml:: conf/experiment/learn_pong.yaml
            :key: env
            :caption: :ref:`envs.{defaults[1]|override /env}`
