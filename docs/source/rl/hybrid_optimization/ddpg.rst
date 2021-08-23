.. _rl.policy_optimization.ddpg:

DDPG
====

.. automodule:: gymnos.rl.policy_optimization.ddpg

.. prompt:: bash

    pip install gymnos[rl.policy_optimization.ddpg]

.. contents::
    :local:

.. _rl.policy_optimization.ddpg__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=rl.policy_optimization.ddpg

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.rl.policy_optimization.ddpg.trainer.DDPGTrainer
        :inherited-members:


.. _rl.policy_optimization.ddpg__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.rl.policy_optimization.ddpg import DDPGPredictor

    DDPGPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.rl.policy_optimization.ddpg.predictor.DDPGPredictor
   :members:
   :inherited-members:
