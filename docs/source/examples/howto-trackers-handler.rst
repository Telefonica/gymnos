###################################
How to create a Tracker
###################################

Gymnos allows you to implement your own custom tracker in a simple way.
The tracker must inherit from ``lib.tracker.tracker.Tracker``.
If you want to use the tracker in an experiment you must add the tracker location with an id in ``lib.var.trackers.json``, e.g ``mytracker: lib.trackers.mytracker.MyTracker``.

The optional methods you can implement are the following:

.. code-block:: python

    def add_tag(self, tag):
        # add tag to the tracker

    def log_image(self, name, file_path):
        # log an image (file) to the tracker

    def log_asset(self, name, file_path):
        # log any asset (file) to the tracker

    def log_figure(self, name, figure):
        # log a Matplotlib figure to the tracker

    def log_metric(self, name, value, step=None):
        # log a metric to the tracker

    def log_param(self, name, value, step=None):
        # log a parameter to the tracker

    def log_other(self, name, value):
        # log any value to the tracker

    def log_model_graph(self, graph):
        # log a tensorflow graph to the tracker

    def end(self):
        # it is called when an experiment is finished
