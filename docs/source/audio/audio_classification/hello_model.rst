.. _audio.audio_classification.hello_model:

Hello Model
===========

.. automodule:: gymnos.audio.audio_classification.hello_model

.. prompt:: bash

    pip install gymnos[audio.audio_classification.hello_model]

.. contents::
    :local:

.. _audio.audio_classification.hello_model__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=audio.audio_classification.hello_model

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.audio.audio_classification.hello_model.trainer.HelloModelTrainer
        :inherited-members:


.. _audio.audio_classification.hello_model__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.audio.audio_classification.hello_model import HelloModelPredictor

    HelloModelPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.audio.audio_classification.hello_model.predictor.HelloModelPredictor
   :members:
