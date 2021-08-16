.. _learn_cartpole_experiment:

Learn Cartpole
==============================

.. autoyamldoc:: conf/experiment/learn_cartpole.yaml
    :lineno-start: 1
    
.. experiment-install:: conf/experiment/learn_cartpole.yaml
    
Usage
**********

.. prompt:: bash

    gymnos-train +experiment=learn_cartpole


.. tabs::

   .. tab:: Trainer

        .. autoyaml:: conf/experiment/learn_cartpole.yaml
            :key: trainer
            :caption: :ref:`{defaults[0].override /trainer}`

   .. tab:: Env

        .. autoyaml:: conf/experiment/learn_cartpole.yaml
            :key: env
            :caption: :ref:`envs.{defaults[1].override /env}`
