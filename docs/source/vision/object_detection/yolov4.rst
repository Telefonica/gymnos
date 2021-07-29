.. _vision.object_detection.yolov4:

Yolov4
======

.. automodule:: gymnos.vision.object_detection.yolov4

.. prompt:: bash

    pip install gymnos[vision.object_detection.yolov4]

.. contents::
    :local:

.. _vision.object_detection.yolov4__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=vision.object_detection.yolov4

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.vision.object_detection.yolov4.trainer.Yolov4Trainer
        :inherited-members:


.. _vision.object_detection.yolov4__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.vision.object_detection.yolov4 import Yolov4Predictor

    Yolov4Predictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.vision.object_detection.yolov4.predictor.Yolov4Predictor
   :members:
