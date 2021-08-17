.. _rl.value_optimization.dqn:

DQN
===

.. automodule:: gymnos.rl.value_optimization.dqn

.. prompt:: bash

    pip install gymnos[rl.value_optimization.dqn]

.. contents::
    :local:

.. _rl.value_optimization.dqn__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=rl.value_optimization.dqn

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.rl.value_optimization.dqn.trainer.DQNTrainer
        :inherited-members:


.. _rl.value_optimization.dqn__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.rl.value_optimization.dqn import DQNPredictor

    DQNPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.rl.value_optimization.dqn.predictor.DQNPredictor
   :members:
   :inherited-members:
