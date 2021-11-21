.. _generative.image_generation.vqgan:

Vqgan
=====

.. automodule:: gymnos.generative.image_generation.vqgan

.. prompt:: bash

    pip install gymnos[generative.image_generation.vqgan]

.. contents::
    :local:

.. _generative.image_generation.vqgan__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=generative.image_generation.vqgan

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.generative.image_generation.vqgan.trainer.VqganTrainer
        :inherited-members:


.. _generative.image_generation.vqgan__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.generative.image_generation.vqgan import VqganPredictor

    VqganPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.generative.image_generation.vqgan.predictor.VqganPredictor
   :members:
