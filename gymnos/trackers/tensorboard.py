#
#
#
#   Tensorboard Tracker
#
#

import os

from .tracker import Tracker
from ..utils.lazy_imports import lazy_imports as lazy


class TensorBoard(Tracker):
    """
    Tracker for `TensorBoard <https://github.com/tensorflow/tensorboard>`_.
    """

    def start(self, run_name, logdir):
        self.writer = SummaryWriter(os.path.join(logdir, "tensorboard", run_name))

    def log_metric(self, name, value, step=None):
        self.writer.add_scalar(name, value, step)

    def end(self):
        self.writer.flush()
        self.writer.close()


class SummaryWriter:

    def __init__(self, logdir):
        _ = lazy.tensorboard  # install tensorboard if needed

        try:
            from tensorflow.summary import FileWriter as Writer
            mode = "tf"
        except ImportError:
            try:
                from torch.utils.tensorboard import SummaryWriter as Writer
                mode = "torch"
            except ImportError:
                _ = lazy.tensorflow
                from tensorflow.summary import FileWriter as Writer
                mode = "tf"

        self.mode = mode
        self.writer = Writer(logdir)

    def add_scalar(self, tag, scalar_value, step=None):
        if self.mode == "tf":
            summary = lazy.tensorflow.Summary(value=[lazy.tensorflow.Summary.Value(tag=tag, simple_value=scalar_value)])
            self.writer.add_summary(summary, step)
        elif self.mode == "torch":
            self.writer.add_scalar(tag, scalar_value, step)
        else:
            raise ValueError("Unknown mode: {}".format(self.mode))

    def close(self):
        self.writer.close()

    def flush(self):
        self.writer.flush()
