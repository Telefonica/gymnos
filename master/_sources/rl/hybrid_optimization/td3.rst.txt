.. _rl.hybrid_optimization.td3:

TD3
===

.. automodule:: gymnos.rl.hybrid_optimization.td3

.. prompt:: bash

    pip install gymnos[rl.hybrid_optimization.td3]

.. contents::
    :local:

.. _rl.hybrid_optimization.td3__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=rl.hybrid_optimization.td3

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.rl.hybrid_optimization.td3.trainer.TD3Trainer
        :inherited-members:


.. _rl.hybrid_optimization.td3__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.rl.hybrid_optimization.td3 import TD3Predictor

    TD3Predictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.rl.hybrid_optimization.td3.predictor.TD3Predictor
   :members:
   :inherited-members:
