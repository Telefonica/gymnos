.. _rl.misc.random:

Random
======

.. automodule:: gymnos.rl.misc.random

.. prompt:: bash

    pip install gymnos[rl.misc.random]

.. contents::
    :local:

.. _rl.misc.random__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=rl.misc.random

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.rl.misc.random.trainer.RandomTrainer
        :inherited-members:


.. _rl.misc.random__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.rl.misc.random import RandomPredictor

    RandomPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.rl.misc.random.predictor.RandomPredictor
   :members:
