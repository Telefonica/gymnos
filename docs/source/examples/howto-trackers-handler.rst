###################################
How to create a Tracker
###################################

Gymnos allows you to implement your own custom tracker in a simple way.

All trackers must follow a protocol with some methods to implement.
First you need to inherit from ``Tracker`` defined in ``lib.trackers.tracker.Tracker``. Then, you need to implement the following methods (all methods are optional):

.. code-block:: python

    def add_tag(self, tag):
        """
        Method to add a tag.
        Arguments:
            - tag: tag of the experiment, type: string
        """

    def log_image(self, name, file_path):
        """
        Method to log an image.
        Arguments:
            - name: name of the image, type: string
            - file_path: path of the image, type: string
        """

    def log_asset(self, name, file_path):
        """
        Method to log any asset.
        Arguments:
            - name: name of the asset, type: string
            - file_path: path of the asset, type: string
        """

    def log_figure(self, name, figure):
        """
        Method to log a Maplotlib figure.
        Arguments:
            - name: name of the figure, type: string
            - figure: Matplotlib figure, type: Matplotlib figure
        """

    def log_metric(self, name, value, step=None):
        """
        Method to log a metric.
        Arguments:
            - name: name of the metric, type: string
            - value: value of the metric, type: number
            - step: step of the metrics, type: number
        """

    def log_param(self, name, value, step=None):
        """
        Method to log a parameter.
        Arguments:
            - name: name of the parameter, type: string
            - value: value of the parameter, type: string
            - step: step of the parameter, type: string
        """

    def log_other(self, name, value):
        """
        Method to log any other value.
        Arguments:
            - name: name of the value, type: string
            - value: any value, type: any(str, int, float, ...)
        """

    def log_model_graph(self, graph):
        """
        Method to log a tensorflow graph.
        Arguments:
            - graph: tensorflow graph, type: tf.Graph
        """

    def end(self):
        """
        Called when the experiment is finished.
        """

If you want to use the tracker in an experiment you must add the tracker location with an id in ``lib.var.trackers.json``, e.g ``mytracker: lib.trackers.mytracker.MyTracker``.
