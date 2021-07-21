.. _vision.image_classification.transfer_efficientnet:

Transfer Efficientnet
=====================

.. automodule:: gymnos.vision.image_classification.transfer_efficientnet

.. code-block:: console

    $ pip install gymnos[vision.image_classification.transfer_efficientnet]

.. contents::
    :local:

.. _vision.image_classification.transfer_efficientnet__trainer:

Trainer
*********

.. code-block:: console

    $ gymnos-train trainer=vision.image_classification.transfer_efficientnet

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.vision.image_classification.transfer_efficientnet.trainer.TransferEfficientNetTrainer
        :inherited-members:


.. _vision.image_classification.transfer_efficientnet__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.vision.image_classification.transfer_efficientnet import TransferEfficientNetPredictor

    TransferEfficientNetPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.vision.image_classification.transfer_efficientnet.predictor.TransferEfficientNetPredictor
   :members:
