#
#
#
#   Tensorboard Tracker
#
#

import os
import tensorflow as tf

from io import BytesIO

from .tracker import Tracker
from ..utils.lazy_imports import lazy_imports


class Tensorboard(Tracker):
    """
    Tracker for `TensorBoard <https://github.com/tensorflow/tensorboard>`_.
    """

    def start(self, run_name, logdir):
        self.writer = tf.summary.FileWriter(os.path.join(logdir, "tensorboard", run_name))

    def log_tag(self, key, value):
        pass

    def log_asset(self, name, file_path):
        pass

    def log_param(self, name, value, step=None):
        pass

    def log_metric(self, name, value, step=None):
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_image(self, name, file_path):
        image = lazy_imports.Image.open(file_path)

        with BytesIO() as buffer:
            image.save(buffer, format="JPEG")
            img_summary = tf.Summary.Image(encoded_image_string=buffer.getvalue(),
                                           width=image.width,
                                           height=image.height)

        summary = tf.Summary(value=[tf.Summary.Value(tag=name, image=img_summary)])
        self.writer.add_summary(summary)

    def log_figure(self, name, figure):
        buffer = BytesIO()
        figure.savefig(buffer, format='png')
        buffer.seek(0)
        img = lazy_imports.Image.open(buffer.getbuffer())

        img_summary = tf.Summary.Image(encoded_image_string=buffer.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])

        summary = tf.Summary(value=[tf.Summary.Value(tag=name, image=img_summary)])
        self.writer.add_summary(summary)

    def end(self):
        self.writer.flush()
        self.writer.close()
