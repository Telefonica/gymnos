######################
Execution results
######################

Gymnos currently generates a common folder structure for each training execution.
In particular, the following artifacts are provided:

.. code-block:: bash
   :emphasize-lines: 4,14

    trainings
    └── boston_housing
       └── executions
          ├── 06-16-31__08-04-2019
          │   ├── artifacts 
          │   │   ├── callbacks
          │   │   │    └── model_checkpoint
          │   │   │        ├── weights.05-27.07.h5
          │   │   │        ├── weights.10-17.55.h5
          │   │   │        └── weights.15-16.03.h5
          │   │   ├── model.h5
          │   │   └── pipeline.joblib         
          │   ├── execution.log
          │   ├── metrics.json
          │   └── training_config.json
          └── 06-39-51__10-04-2019
              ├── artifacts 
              │   ├── model.joblib
              │   └── pipeline.joblib         
              ├── execution.log
              ├── metrics.json
              └── training_config.json

***********************
Execution log
***********************
Execution logs show relevant information about training. Different levels of detail are provided.

The following block shows an example extracted from the BostonHousing training execution:

.. code-block:: bash

    2019-05-03 17:58:47,442 - gymnosd - INFO - Main - Starting gymnos environment ...
    2019-05-03 17:58:47,529 - gymnosd - INFO - Trainer - Running experiment: 17-58-47--03-05-2019__keras ...
    2019-05-03 17:58:47,529 - gymnosd - INFO - Trainer - The execution will be saved in the following directory: trainings/boston_housing/executions/17-58-47--03-05-2019__keras
    2019-05-03 17:58:47,529 - gymnosd - INFO - Trainer - Tracking information will be saved in the following directory: trainings/boston_housing/trackings
    2019-05-03 17:58:48,725 - gymnosd - DEBUG - Trainer - Python version: 3.5.6
    2019-05-03 17:58:48,725 - gymnosd - DEBUG - Trainer - Platform: Darwin-18.5.0-x86_64-i386-64bit
    2019-05-03 17:58:48,726 - gymnosd - DEBUG - Trainer - Found 0 GPUs
    2019-05-03 17:58:48,877 - gymnosd - INFO - Trainer - Loading dataset: boston_housing ...
    2019-05-03 17:58:48,877 - gymnosd - INFO - BostonHousing - Dataset exists on cache (cache/datasets/boston_housing.h5)
    2019-05-03 17:58:48,877 - gymnosd - INFO - BostonHousing - Retrieving dataset from cache
    2019-05-03 17:58:48,929 - gymnosd - DEBUG - HDFManager - Retrieved Numpy dataset from HDF5 X key (cache/datasets/boston_housing.h5)
    2019-05-03 17:58:48,931 - gymnosd - DEBUG - HDFManager - Retrieved Numpy dataset from HDF5 y key (cache/datasets/boston_housing.h5)
    2019-05-03 17:58:48,932 - gymnosd - DEBUG - Trainer - Loading data took 0.05s
    2019-05-03 17:58:48,932 - gymnosd - INFO - Trainer - Splitting dataset -> Train: 0.80 | Test: 0.20
    2019-05-03 17:58:48,932 - gymnosd - INFO - Trainer - Applying 1 preprocessors (StandardScaler)
    2019-05-03 17:58:48,934 - gymnosd - DEBUG - Trainer - Fitting preprocessors to train data took 0.00s
    2019-05-03 17:58:48,935 - gymnosd - DEBUG - Trainer - Preprocessing data took 0.00s
    2019-05-03 17:58:48,935 - gymnosd - INFO - Trainer - Fitting model with 404 samples ...
    2019-05-03 17:58:49,423 - gymnosd - DEBUG - Trainer - Fitting model took 0.49s
    2019-05-03 17:58:49,423 - gymnosd - INFO - Trainer - Results for mean_absolute_error: Min: 3.65 | Max: 21.24 | Mean: 10.60
    2019-05-03 17:58:49,423 - gymnosd - INFO - Trainer - Results for loss: Min: 24.76 | Max: 540.83 | Mean: 201.22
    2019-05-03 17:58:49,423 - gymnosd - INFO - Trainer - Results for val_mean_absolute_error: Min: 3.83 | Max: 20.19 | Mean: 9.29
    2019-05-03 17:58:49,423 - gymnosd - INFO - Trainer - Results for val_loss: Min: 37.78 | Max: 485.67 | Mean: 163.50
    2019-05-03 17:58:49,423 - gymnosd - INFO - Trainer - Logging train metrics to trackers
    2019-05-03 17:58:49,711 - gymnosd - INFO - Trainer - Evaluating model with 102 samples
    2019-05-03 17:58:49,713 - gymnosd - INFO - Trainer - Test results for mean_absolute_error: 3.56
    2019-05-03 17:58:49,713 - gymnosd - INFO - Trainer - Test results for loss: 20.34
    2019-05-03 17:58:49,713 - gymnosd - DEBUG - Trainer - Evaluating model took 0.00s
    2019-05-03 17:58:49,713 - gymnosd - INFO - Trainer - Logging test metrics to trackers
    2019-05-03 17:58:49,735 - gymnosd - INFO - Trainer - Saving model
    2019-05-03 17:58:49,750 - gymnosd - INFO - Trainer - Saving pipeline
    2019-05-03 17:58:49,751 - gymnosd - INFO - Trainer - Saving metrics to JSON file
    2019-05-03 17:58:49,752 - gymnosd - INFO - Trainer - Metrics, platform information and elapsed times saved to trainings/boston_housing/executions/17-58-47--03-05-2019__keras/metrics.json file
    2019-05-03 17:58:49,764 - gymnosd - INFO - Main - Success! Execution saved (trainings/boston_housing/executions/17-58-47--03-05-2019__keras)


***********************
Metrics
***********************
Execution metrics are key to develop benchmarking criteria. 
Gymnos currently provides a json file with different types of metrics such us: 

* Time consumption on each training stage
* HW details of the underlying execution environment
* Validation, testing, loss metrics

.. code-block:: json

    {
        "metrics": {
            "mean_absolute_error": [
                21.236964084134243,
                17.964938217263803,
                13.134718602246577,
                7.886666022511599,
                5.903721864467407,
                4.434306234416395,
                3.6490739879041616
            ],
            "val_mean_absolute_error": [
                20.190430933886237,
                15.91676406104966,
                9.766545711177411,
                6.184523365285137,
                5.162875888371231,
                4.001074493521511,
                3.8324677117980355
            ],
            "test_loss": [
                20.33574592365938
            ],
            "test_mean_absolute_error": [
                3.562303206499885
            ],
            "loss": [
                540.8273107950444,
                406.92359395546487,
                241.13668793026764,
                100.29069398181274,
                59.68981130367065,
                34.91461652497647,
                24.762916974108606
            ],
            "val_loss": [
                485.67176108785196,
                320.8614704396465,
                141.8022104395498,
                69.81834188782342,
                50.69452912736647,
                37.77724575760341,
                37.85807494361802
            ]
        },
        "platform": {
            "gpu": [],
            "python_compiler": "GCC 4.2.1 Compatible Apple LLVM 10.0.0 (clang-1000.10.44.4)",
            "node": "iMac-Pro.local",
            "cpu": {
                "cores": 16,
                "brand": "Intel(R) Xeon(R) W-2140B CPU @ 3.20GHz"
            },
            "processor": "i386",
            "architecture": "64bit",
            "python_version": "3.5.6",
            "system": "Darwin",
            "platform": "Darwin-18.5.0-x86_64-i386-64bit"
        },
        "elapsed": {
            "fit_preprocessors": 0.0017540454864501953,
            "evaluate_model": 0.0020329952239990234,
            "load_data": 0.05479598045349121,
            "transform_preprocessors": 0.0009219646453857422,
            "fit_model": 0.48743200302124023
        }
    }


*************************
Trained model
*************************
In order to reuse the model for future predictions, a copy of the model with trained parameters is saved.

******************************
Trained pipeline
******************************
In order to reuse the preprocessors pipeline for future preprocessing, a copy of the pipeline with trained parameters is saved.

***********************
Training configuration
***********************
In order to keep track of the experiment a copy of the original configuration is also provided.

.. code-block:: json

    {
        "experiment": {
            "name": "Boston Housing"
        },
        "model": {
            "name": "keras",
            "parameters": {
                "sequential": [
                    {"type": "dense", "units": 512, "activation": "relu", "input_shape": [13]},
                    {"type": "dense", "units": 128, "activation": "relu"},
                    {"type": "dense", "units": 1, "activation": "linear"}
                ],
                "compilation": {
                    "optimizer": "adam",
                    "loss": "mse",
                    "metrics": ["mae"]
                }
            }
        },
        "dataset": {
            "name": "boston_housing",
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
        "training": {
            "batch_size": 32,
            "epochs": 25,
            "callbacks": [
                {
                    "type": "early_stopping"
                },
                {
                    "type": "model_checkpoint",
                    "filepath": "weights.{epoch:02d}-{val_loss:.2f}.h5",
                    "period": 5
                }
            ],
            "validation_split": 0.25
        },
        "tracking": {
            "params": {
                "device": "cpu"
            },
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


