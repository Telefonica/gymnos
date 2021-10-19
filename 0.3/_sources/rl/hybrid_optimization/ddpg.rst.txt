.. _rl.hybrid_optimization.ddpg:

DDPG
====

.. automodule:: gymnos.rl.hybrid_optimization.ddpg

.. prompt:: bash

    pip install gymnos[rl.hybrid_optimization.ddpg]

.. contents::
    :local:

.. _rl.hybrid_optimization.ddpg__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=rl.hybrid_optimization.ddpg

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.rl.hybrid_optimization.ddpg.trainer.DDPGTrainer
        :inherited-members:


.. _rl.hybrid_optimization.ddpg__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.rl.hybrid_optimization.ddpg import DDPGPredictor

    DDPGPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.rl.hybrid_optimization.ddpg.predictor.DDPGPredictor
   :members:
   :inherited-members:
