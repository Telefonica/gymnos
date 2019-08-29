###################
Trackers
###################

Gymnos trackers are a collection of trackers with a common API allowing their use in a pipeline of a supervised learning system. All trackers inherit from :class:`gymnos.trackers.tracker.Tracker`.

Usage
*******
.. code-block:: python

    tracker = gymnos.trackers.load("mlflow")

    tracker.start(run_name="testing", logdir="trackings")  # start tracker run, must be called before any log_* method

    tracker.log_tag("cnn")  # log tag

    tracker.log_metric("accuracy", 0.9)  # log metric

    tracker.log_metrics({  # log multiple metrics
        "precision": 0.8,
        "recall": 0.6
    })

    tracker.log_param("beta", 0.5)   # log parameter

    tracker.log_params({  # log multiple parameters
        "epochs": 5,
        "batch_size": 32
    })

    tracker.log_asset("readme.md")  # log any file

    tracker.log_image(name="confusion matrix", file_path"confusion_matrix.png")  # log image file

    tracker.log_figure(name="my figure", figure=myfigure) # log Matplotlib figure

    tracker.end()  # end tracker run

If you want to use multiple trackers, take a look to TrackerList:

.. code-block:: python

    tracker_1 = gymnos.trackers.load("mlflow")
    tracker_2 = gymnos.trackers.load("tensorboard")

    tracker_list = TrackerList(
        tracker_1,
        tracker_2
    )

    tracker_list.start(...)  # call start on every tracker

    ...  # same methods as any tracker



All Trackers
********************

.. contents:: 
    :local: 

comet_ml
========================
.. autoclass:: gymnos.trackers.comet_ml.CometML
    :noindex:

mlflow
========================
.. autoclass:: gymnos.trackers.mlflow.MLflow
    :noindex:

tensorboard
========================
.. autoclass:: gymnos.trackers.tensorboard.TensorBoard
    :noindex:

