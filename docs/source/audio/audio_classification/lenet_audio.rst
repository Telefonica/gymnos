.. _audio.audio_classification.lenet_audio:

Lenet Audio
===========

.. automodule:: gymnos.audio.audio_classification.lenet_audio

.. prompt:: bash

    pip install gymnos[audio.audio_classification.lenet_audio]

.. contents::
    :local:

.. _audio.audio_classification.lenet_audio__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=audio.audio_classification.lenet_audio

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.audio.audio_classification.lenet_audio.trainer.LenetAudioTrainer
        :inherited-members:


.. _audio.audio_classification.lenet_audio__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.audio.audio_classification.lenet_audio import LenetAudioPredictor

    LenetAudioPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.audio.audio_classification.lenet_audio.predictor.LenetAudioPredictor
   :members:
