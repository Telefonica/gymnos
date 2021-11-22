.. _generative.image_generation.wgan_gp:

Wgan Gp
=======

.. automodule:: gymnos.generative.image_generation.wgan_gp

.. prompt:: bash

    pip install gymnos[generative.image_generation.wgan_gp]

.. contents::
    :local:

.. _generative.image_generation.wgan_gp__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=generative.image_generation.wgan_gp

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.generative.image_generation.wgan_gp.trainer.WganGpTrainer
        :inherited-members:


.. _generative.image_generation.wgan_gp__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.generative.image_generation.wgan_gp import WganGpPredictor

    WganGpPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.generative.image_generation.wgan_gp.predictor.WganGpPredictor
   :members:
