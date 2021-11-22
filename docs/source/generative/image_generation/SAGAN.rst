.. _generative.image_generation.SAGAN:

S A G A N
=========

.. automodule:: gymnos.generative.image_generation.SAGAN

.. prompt:: bash

    pip install gymnos[generative.image_generation.SAGAN]

.. contents::
    :local:

.. _generative.image_generation.SAGAN__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=generative.image_generation.SAGAN

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.generative.image_generation.SAGAN.trainer.SAGANTrainer
        :inherited-members:


.. _generative.image_generation.SAGAN__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.generative.image_generation.SAGAN import SAGANPredictor

    SAGANPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.generative.image_generation.SAGAN.predictor.SAGANPredictor
   :members:
