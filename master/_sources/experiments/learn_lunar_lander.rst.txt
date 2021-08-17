.. _learn_lunar_lander_experiment:

Learn Lunar Lander
==============================

.. autoyamldoc:: conf/experiment/learn_lunar_lander.yaml
    :lineno-start: 1
    
.. experiment-install:: conf/experiment/learn_lunar_lander.yaml
    
Usage
**********

.. prompt:: bash

    gymnos-train +experiment=learn_lunar_lander


.. tabs::

   .. tab:: Trainer

        .. autoyaml:: conf/experiment/learn_lunar_lander.yaml
            :key: trainer
            :caption: :ref:`{defaults[0].override /trainer}`

   .. tab:: Env

        .. autoyaml:: conf/experiment/learn_lunar_lander.yaml
            :key: env
            :caption: :ref:`envs.{defaults[1].override /env}`
