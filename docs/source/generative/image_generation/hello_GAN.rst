.. _generative.image_generation.hello_GAN:

Hello  G A N
============

.. automodule:: gymnos.generative.image_generation.hello_GAN

.. prompt:: bash

    pip install gymnos[generative.image_generation.hello_GAN]

.. contents::
    :local:

.. _generative.image_generation.hello_GAN__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=generative.image_generation.hello_GAN

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.generative.image_generation.hello_GAN.trainer.Hello_GANTrainer
        :inherited-members:


.. _generative.image_generation.hello_GAN__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.generative.image_generation.hello_GAN import Hello_GANPredictor

    Hello_GANPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.generative.image_generation.hello_GAN.predictor.Hello_GANPredictor
   :members:
