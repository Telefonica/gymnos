.. _rl.hybrid_optimization.sac:

SAC
===

.. automodule:: gymnos.rl.hybrid_optimization.sac

.. prompt:: bash

    pip install gymnos[rl.hybrid_optimization.sac]

.. contents::
    :local:

.. _rl.hybrid_optimization.sac__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=rl.hybrid_optimization.sac

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.rl.hybrid_optimization.sac.trainer.SACTrainer
        :inherited-members:


.. _rl.hybrid_optimization.sac__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.rl.hybrid_optimization.sac import SACPredictor

    SACPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.rl.hybrid_optimization.sac.predictor.SACPredictor
   :members:
   :inherited-members:
