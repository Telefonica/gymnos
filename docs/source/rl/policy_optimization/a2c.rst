.. _rl.policy_optimization.a2c:

A2C
===

.. automodule:: gymnos.rl.policy_optimization.a2c

.. prompt:: bash

    pip install gymnos[rl.policy_optimization.a2c]

.. contents::
    :local:

.. _rl.policy_optimization.a2c__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=rl.policy_optimization.a2c

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.rl.policy_optimization.a2c.trainer.A2CTrainer
        :inherited-members:


.. _rl.policy_optimization.a2c__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.rl.policy_optimization.a2c import A2CPredictor

    A2CPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.rl.policy_optimization.a2c.predictor.A2CPredictor
   :members:
   :inherited-members:
