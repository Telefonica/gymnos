.. _audio.audio_classification.hell_model:

Hell Model
==========

.. automodule:: gymnos.audio.audio_classification.hell_model

.. prompt:: bash

    pip install gymnos[audio.audio_classification.hell_model]

.. contents::
    :local:

.. _audio.audio_classification.hell_model__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=audio.audio_classification.hell_model

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.audio.audio_classification.hell_model.trainer.HellModelTrainer
        :inherited-members:


.. _audio.audio_classification.hell_model__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.audio.audio_classification.hell_model import HellModelPredictor

    HellModelPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.audio.audio_classification.hell_model.predictor.HellModelPredictor
   :members:
