.. _generative.image_generation.bigGAN:

Big G A N
=========

.. automodule:: gymnos.generative.image_generation.bigGAN

.. prompt:: bash

    pip install gymnos[generative.image_generation.bigGAN]

.. contents::
    :local:

.. _generative.image_generation.bigGAN__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=generative.image_generation.bigGAN

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.generative.image_generation.bigGAN.trainer.BigGANTrainer
        :inherited-members:


.. _generative.image_generation.bigGAN__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.generative.image_generation.bigGAN import BigGANPredictor

    BigGANPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.generative.image_generation.bigGAN.predictor.BigGANPredictor
   :members:
