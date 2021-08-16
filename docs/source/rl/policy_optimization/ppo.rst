.. _rl.policy_optimization.ppo:

PPO
===

.. automodule:: gymnos.rl.policy_optimization.ppo

.. prompt:: bash

    pip install gymnos[rl.policy_optimization.ppo]

.. contents::
    :local:

.. _rl.policy_optimization.ppo__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=rl.policy_optimization.ppo

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.rl.policy_optimization.ppo.trainer.PPOTrainer
        :inherited-members:


.. _rl.policy_optimization.ppo__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.rl.policy_optimization.ppo import PPOPredictor

    PPOPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.rl.policy_optimization.ppo.predictor.PPOPredictor
   :members:
   :inherited-members:
