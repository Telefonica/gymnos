New experiment
==============================

In this tutorial we will create a new experiment for image classification using :ref:`vision.image_classification.transfer_efficientnet__trainer` as the trainer and :ref:`datasets.dogs_vs_cats` as the dataset.

The name of the experiment will be ``my_experiment``.

First of all, we will run the command :ref:`gymnos-create`:

.. prompt:: bash

    gymnos-create experiment my_experiment

.. tip::
    If you're creating an experiment for an RL model, you must append the ``--rl`` flag:

    .. prompt:: bash

        gymnos-create experiment --rl my_experiment


This will create the file ``conf/experiment/my_experiment.yaml``.

It would look something like this:

.. code-block:: yaml
    :linenos:
    :emphasize-lines: 2,5,6,9,12
    :caption: conf/experiment/my_experiment.yaml

    # @package _global_
    # TODO: description about experiment

    defaults:
        - override /trainer: <trainer_name>  # TODO: set name of trainer to use
        - override /dataset: <dataset_name>  # TODO: set name of dataset to use

    trainer:
        <param>: <value>   # TODO: override default trainer params

    dataset:
        <param>: <value>  # TODO: override default dataset params

First of all, we will add a description for our experiment:

.. code-block:: yaml
    :lineno-start: 2

    # Classify dogs vs cats using Transfer EfficientNet model.


Now, we will override the trainer with :ref:`vision.image_classification.transfer_efficientnet`:

.. code-block:: yaml
    :lineno-start: 5

    - override /trainer: vision.image_classification.transfer_efficientnet

Then, we will override the dataset with :ref:`datasets.dogs_vs_cats`

.. code-block:: yaml
    :lineno-start: 6

    - override /dataset: dogs_vs_cats

Now, we can override any :ref:`vision.image_classification.transfer_efficientnet__trainer` parameter:

.. code-block:: yaml
    :lineno-start: 8

    trainer:
        classes: [dog, cat]
        batch_size: 64

We can also override any :ref:`datasets.dogs_vs_cats` parameter:

.. code-block:: yaml
    :lineno-start: 11

    dataset:
        max_workers: 8
        force_download: false

Once you're done modifying the experiment, you can run it using the :ref:`gymnos-train` command:

.. prompt:: bash

    gymnos-train +experiment=my_experiment

Documentation
---------------
Remember to check the :ref:`documentation` for your new experiment
