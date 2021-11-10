.. _audio.tiny_audio_classification.tiny_audio:

Tiny Audio
==========

.. automodule:: gymnos.audio.tiny_audio_classification.tiny_audio

.. prompt:: bash

    pip install gymnos[audio.tiny_audio_classification.tiny_audio]

.. contents::
    :local:

.. _audio.tiny_audio_classification.tiny_audio__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=audio.tiny_audio_classification.tiny_audio

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.audio.tiny_audio_classification.tiny_audio.trainer.TinyAudioTrainer
        :inherited-members:


.. _audio.tiny_audio_classification.tiny_audio__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.audio.tiny_audio_classification.tiny_audio import TinyAudioPredictor

    TinyAudioPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.audio.tiny_audio_classification.tiny_audio.predictor.TinyAudioPredictor
   :members:
