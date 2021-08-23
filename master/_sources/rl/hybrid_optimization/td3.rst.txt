.. _rl.policy_optimization.td3:

TD3
===

.. automodule:: gymnos.rl.policy_optimization.td3

.. prompt:: bash

    pip install gymnos[rl.policy_optimization.td3]

.. contents::
    :local:

.. _rl.policy_optimization.td3__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=rl.policy_optimization.td3

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.rl.policy_optimization.td3.trainer.TD3Trainer
        :inherited-members:


.. _rl.policy_optimization.td3__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.rl.policy_optimization.td3 import TD3Predictor

    TD3Predictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.rl.policy_optimization.td3.predictor.TD3Predictor
   :members:
   :inherited-members:
