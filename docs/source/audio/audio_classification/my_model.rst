.. _audio.audio_classification.my_model:

My Model
========

.. automodule:: gymnos.audio.audio_classification.my_model

.. prompt:: bash

    pip install gymnos[audio.audio_classification.my_model]

.. contents::
    :local:

.. _audio.audio_classification.my_model__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=audio.audio_classification.my_model

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.audio.audio_classification.my_model.trainer.MyModelTrainer
        :inherited-members:


.. _audio.audio_classification.my_model__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.audio.audio_classification.my_model import MyModelPredictor

    MyModelPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.audio.audio_classification.my_model.predictor.MyModelPredictor
   :members:
