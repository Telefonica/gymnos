##################
Command: train
##################

Train model using a JSON file.

This will create the following files in the ``execution_dir``:

- ``logging.json``: logging outputs
- ``training_specs.json``: input training JSON file
- ``results.json``: JSON file with dictionnary output from :func:`gymnos.trainer.Trainer.train`
- ``saved_trainer.zip``: saved trainer file.

Usage
-------

.. argparse::
    :ref: scripts.cli.build_parser
    :prog: gymnos
    :path: train

Examples
----------

To train `dogs_vs_cats_cnn` model with `dogs_vs_cats` dataset using 3 preprocessors (`image_resize`, `grayscale` and `divide`) and `tensorboard` and `mlflow` as tracking tools:

.. code-block:: json
    :caption: dogs_vs_cats.json

    {
        "model": {
            "model": {
                "type": "dogs_vs_cats_cnn",
                "input_shape": [80, 80, 1]
            },
            "training": {
                "batch_size": 32,
                "epochs": 5,
                "validation_split": 0.25
            }
        },
        "dataset": {
            "dataset": {
                "type": "dogs_vs_cats"
            },
            "one_hot": true,
            "samples": {
                "train": 0.8,
                "test": 0.2
            },
            "preprocessors": [
                {
                    "type": "image_resize",
                    "width": 80,
                    "height": 80
                },
                {
                    "type": "grayscale"
                },
                {
                    "type": "divide",
                    "factor": 255
                }
            ]
        },
        "tracking": {
            "trackers": [
                {
                    "type": "tensorboard"
                },
                {
                    "type": "mlflow"
                }
            ]
        }
    }

.. code-block:: console

    $ gymnos train dogs_vs_cats.json

To define and train a neural network directly into training JSON file:

.. code-block:: json
    :caption: boston_housing.json

    {
        "model": {
            "model": {
                "type": "keras_regressor",
                "sequential": [
                    {"type": "dense", "units": 512, "activation": "relu"},
                    {"type": "dense", "units": 128, "activation": "relu"},
                    {"type": "dense", "units": 1, "activation": "linear"}
                ],
                "input_shape": [13],
                "optimizer": "adam",
                "loss": "mse",
                "metrics": ["mae"]
            },
            "training": {
                "batch_size": 32,
                "epochs": 25,
                "callbacks": [
                    {
                        "type": "early_stopping"
                    }
                ],
                "validation_split": 0.25
            },
        },
        "dataset": {
            "dataset": {
                "type": "boston_housing"
            }
            "samples": {
                "train": 0.8,
                "test": 0.2
            },
            "preprocessors": [
                {
                    "type": "standard_scaler"
                }
            ],
            "seed": 0
        },
        "tracking": {
            "trackers": [
                {
                    "type": "tensorboard"
                },
                {
                    "type": "mlflow"
                }
            ]
        }
    }


.. code-block:: console

    $ gymnos train boston_housing.json
