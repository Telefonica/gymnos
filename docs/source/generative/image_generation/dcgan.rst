.. _generative.image_generation.dcgan:

DCGAN
=====

.. automodule:: gymnos.generative.image_generation.dcgan

.. prompt:: bash

    pip install gymnos[generative.image_generation.dcgan]

.. contents::
    :local:

.. _generative.image_generation.dcgan__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=generative.image_generation.dcgan

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.generative.image_generation.dcgan.trainer.DCGANTrainer
        :inherited-members:


.. _generative.image_generation.dcgan__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.generative.image_generation.dcgan import DCGANPredictor

    DCGANPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.generative.image_generation.dcgan.predictor.DCGANPredictor
   :members:
