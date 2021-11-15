.. _tabular.classification.neural_network_classifier:

Neural Network Classifier
=========================

.. automodule:: gymnos.tabular.classification.neural_network_classifier

.. prompt:: bash

    pip install gymnos[tabular.classification.neural_network_classifier]

.. contents::
    :local:

.. _tabular.classification.neural_network_classifier__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=tabular.classification.neural_network_classifier

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.tabular.classification.neural_network_classifier.trainer.NeuralNetworkClassifierTrainer
        :inherited-members:


.. _tabular.classification.neural_network_classifier__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.tabular.classification.neural_network_classifier import NeuralNetworkClassifierPredictor

    NeuralNetworkClassifierPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.tabular.classification.neural_network_classifier.predictor.NeuralNetworkClassifierPredictor
   :members:
