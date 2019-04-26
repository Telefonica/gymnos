###############################
Training Configuration
###############################

.. _training_configuration:

Gymnos allows you to configure and train powerful models without writing a single line of code, only using a JSON file with the configuration of the experiment.

To run an experiment, the following keys are available:

    - ``"experiment"``: defines name and description of the experiment
    - ``"dataset"``: defines the dataset with the associated preprocessing
    - ``"model"``:  defines the model
    - ``"training"``: defines training parameters
    - ``"tracking"``: defines trackers to log parameters / metrics

Each key is associated with a Python instance that builds the Python objects from the JSON value (e.g build model from name, build dataset from name, build trackers, ...).

.. image:: images/gymnos-training-config.png
   :width: 100%

Experiment
==========
.. autoclass:: lib.core.experiment.Experiment

Dataset
==========
.. autoclass:: lib.core.dataset.Dataset

Model
==========
.. autoclass:: lib.core.model.Model

Training
========
.. autoclass:: lib.core.training.Training

Tracking
========
.. autoclass:: lib.core.tracking.Tracking
