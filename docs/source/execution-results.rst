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
          │   ├── callbacks
          │   │   └── model_checkpoint
          │   │       ├── weights.05-27.07.h5
          │   │       ├── weights.10-17.55.h5
          │   │       └── weights.15-16.03.h5
          │   ├── execution.log
          │   ├── metrics.json
          │   ├── model.h5
          │   └── training_config.json
          └── 06-39-51__10-04-2019
              ├── callbacks
              │   └── model_checkpoint
              │       ├── weights.05-28.64.h5
              │       ├── weights.10-18.30.h5
              │       └── weights.15-16.12.h5
              ├── execution.log
              ├── metrics.json
              ├── model.h5
              └── training_config.json

***********************
Callbacks
***********************
Gymnos currently supports the following callbacks:

=======================
Keras callbacks
=======================
`Keras callbacks <https://keras.io/callbacks/>`_ are essentially a set of functions to be applied at given stages of the training procedure. 
You can use callbacks to get a view on internal states and statistics of the model during training. 
You can pass a list of callbacks (as the keyword argument callbacks) to the .fit() method of the Model class.
The relevant methods of the callbacks will then be called at each stage of the training.

| **BaseLogger**
| Callback that accumulates epoch averages of metrics. This callback is automatically applied to every Keras model.

| **TerminateOnNaN**
| Callback that terminates training when a NaN loss is encountered.

| **ProgbarLogger**
| Callback that prints metrics to stdout.

| **History**
| Callback that records events into a History object.
| This callback is automatically applied to every Keras model. 
| The History object gets returned by the fit method of models.

| **ModelCheckpoint**
| Save the model after every epoch.``filepath`` can contain named formatting options, which will be filled with the values of epoch and keys in logs (passed in on_epoch_end).
| For example: if filepath is weights.{epoch:02d}-{val_loss:.2f}.hdf5, then the model checkpoints will be saved with the epoch number and the validation loss in the filename.

| **EarlyStopping**
| Stop training when a monitored quantity has stopped improving.

| **RemoteMonitor**
| Callback used to stream events to a server.
| Requires the requests library. Events are sent to root + '/publish/epoch/end/' by default. Calls are HTTP POST, with a data argument which is a JSON-encoded dictionary of event data. If send_as_json is set to True, the content type of the request will be application/json. Otherwise the serialized JSON will be send within a form

| **LearningRateScheduler**
| Learning rate scheduler.

| **TensorBoard**
| TensorBoard basic visualizations. TensorBoard is a visualization tool provided with TensorFlow.
| This callback writes a log for TensorBoard, which allows you to visualize dynamic graphs of your training and test metrics, as well as activation histograms for the different layers in your model.

| **ReduceLROnPlateau**
| Reduce learning rate when a metric has stopped improving.
| Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.

| **CSVLogger**
| Callback that streams epoch results to a csv file.
| Supports all values that can be represented as a string, including 1D iterables such as np.ndarray.

| **LambdaCallback**
| Callback for creating simple, custom callbacks on-the-fly.


***********************
Execution log
***********************
Execution logs show relevant information about training. Different levels of detail are provided.

The following block shows an example extracted from the BostonHousing training execution:

.. code-block:: bash

    2019-04-08 06:16:29,724 - gymnosd - INFO - Main - Starting gymnos environment ...
    2019-04-08 06:16:29,725 - gymnosd - INFO - Model - Building Keras model from network specification
    2019-04-08 06:16:29,910 - gymnosd - INFO - Model - Compiling Keras model
    2019-04-08 06:16:31,749 - gymnosd - INFO - Trainer - Running experiment: 06-16-31__08-04-2019 ...
    2019-04-08 06:16:31,758 - gymnosd - INFO - Trainer - Creating directory to save training results (trainings/boston_housing/executions/06-16-31__08-04-2019)
    2019-04-08 06:16:33,376 - gymnosd - INFO - Trainer - Loading dataset: boston_housing ...
    2019-04-08 06:16:33,378 - gymnosd - INFO - BostonHousing - Downloading dataset ...
    2019-04-08 06:16:33,378 - gymnosd - INFO - BostonHousing - Retrieving dataset from library ...
    2019-04-08 06:16:33,379 - gymnosd - INFO - BostonHousing - Reading dataset ...
    2019-04-08 06:16:34,069 - gymnosd - INFO - BostonHousing - Saving dataset to cache ...
    2019-04-08 06:16:34,069 - gymnosd - DEBUG - HDFManager - Saving Numpy dataset to HDF5 X key (cache/datasets/BostonHousing.h5)
    2019-04-08 06:16:34,146 - gymnosd - DEBUG - HDFManager - Saving Numpy dataset to HDF5 y key (cache/datasets/BostonHousing.h5)
    2019-04-08 06:16:34,153 - gymnosd - DEBUG - Trainer - Loading data took 0.78s
    2019-04-08 06:16:34,154 - gymnosd - INFO - Trainer - Splitting dataset -> Fit: 0.6 | Test: 0.2 | Val: 0.2 ...
    2019-04-08 06:16:34,155 - gymnosd - INFO - Trainer - Applying 0 preprocessors ...
    2019-04-08 06:16:34,159 - gymnosd - DEBUG - Trainer - Preprocessing took 0.00s
    2019-04-08 06:16:34,159 - gymnosd - INFO - Trainer - Applying 1 transformers ...
    2019-04-08 06:16:34,165 - gymnosd - DEBUG - Trainer - Fitting transformers to train dataset took 0.01s
    2019-04-08 06:16:34,166 - gymnosd - DEBUG - Trainer - Transforming datasets took 0.00s
    2019-04-08 06:16:34,167 - gymnosd - INFO - Trainer - Fitting model with 303 samples ...
    2019-04-08 06:16:36,506 - gymnosd - DEBUG - Trainer - Fitting model took 2.34s
    2019-04-08 06:16:36,508 - gymnosd - INFO - Trainer - Logging train metrics
    2019-04-08 06:16:46,887 - gymnosd - INFO - Trainer - Evaluating model with 101 samples
    2019-04-08 06:16:46,892 - gymnosd - DEBUG - Trainer - Evaluating model took 0.00s
    2019-04-08 06:16:46,892 - gymnosd - INFO - Trainer - Logging test metrics
    2019-04-08 06:16:47,376 - gymnosd - INFO - Trainer - Saving model

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
        "elapsed": {
            "transform_preprocessors": 0.003850698471069336,
            "fit_model": 2.338494300842285,
            "transform_transformers": 0.0006225109100341797,
            "evaluate_model": 0.0040361881256103516,
            "fit_transformers": 0.0055310726165771484,
            "load_data": 0.7762744426727295
        },
        "metrics": {
            "val_loss": [
                466.7846987884824,
                270.62497643196934,
                15.611931527015006,
                15.748360737715617
            ],
            "val_mean_absolute_error": [
                19.182923175320767,
                13.976237391481305,
                2.6988475724021987,
                2.6569605841495023
            ],
            "test_loss": [
                22.415260909807564
            ],
            "loss": [
                516.9505325166306,
                351.41469722533776,
                9.36743452210631,
                9.180129478473475
            ],
            "mean_absolute_error": [
                20.776301922184405,
                16.260077253033227,
                2.2440381042241264,
                2.2286516313899076
            ],
            "test_mean_absolute_error": [
                2.991008189645144
            ]
        }
    }

**************************
Model pre-trained weights
**************************
As part of the execution outcomes, the trained model is saved in an ``.h5`` file. 
The idea behind this is to reuse pre-trained weights for future predictions.
 

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
            "input_shape": [13],
            "network": [
                {"type": "dense", "units": 512, "activation": "relu"},
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dense", "units": 1, "activation": "linear"}
            ],
            "compilation": {
                "optimizer": "adam",
                "loss": "mse",
                "metrics": ["mae"]
            }
        },
        "dataset": {
            "name": "boston_housing",
            "transformers": [
                {
                    "type": "standard_scaler"
                }
            ]
        },
        "training": {
            "samples": {
                "fit": 0.6,
                "val": 0.2,
                "test": 0.2
            },
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
            ]
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

