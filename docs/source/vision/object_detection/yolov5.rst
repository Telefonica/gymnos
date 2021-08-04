.. _vision.object_detection.yolov5:

Yolov5
======

.. automodule:: gymnos.vision.object_detection.yolov5

.. prompt:: bash

    pip install gymnos[vision.object_detection.yolov5]

.. contents::
    :local:

.. _vision.object_detection.yolov5__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=vision.object_detection.yolov5

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.vision.object_detection.yolov5.trainer.Yolov5Trainer
        :inherited-members:


.. _vision.object_detection.yolov5__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.vision.object_detection.yolov5 import Yolov5Predictor

    Yolov5Predictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.vision.object_detection.yolov5.predictor.Yolov5Predictor
   :members:
