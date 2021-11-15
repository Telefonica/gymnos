.. _tabular.classification.random_forest_classifier:

Random Forest Classifier
========================

.. automodule:: gymnos.tabular.classification.random_forest_classifier

.. prompt:: bash

    pip install gymnos[tabular.classification.random_forest_classifier]

.. contents::
    :local:

.. _tabular.classification.random_forest_classifier__trainer:

Trainer
*********

.. prompt:: bash

    gymnos-train trainer=tabular.classification.random_forest_classifier

.. rst-class:: gymnos-hydra

    .. autoclass:: gymnos.tabular.classification.random_forest_classifier.trainer.RandomForestClassifierTrainer
        :inherited-members:


.. _tabular.classification.random_forest_classifier__predictor:

Predictor
***********

.. code-block:: py

    from gymnos.tabular.classification.random_forest_classifier import RandomForestClassifierPredictor

    RandomForestClassifierPredictor.from_pretrained("johndoe/models/pretrained", *args, **kwargs)

.. autoclass:: gymnos.tabular.classification.random_forest_classifier.predictor.RandomForestClassifierPredictor
   :members:
