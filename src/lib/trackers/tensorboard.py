#
#
#
#   Tensorboard Tracker
#
#

import os
import cv2 as cv
import numpy as np
import tensorflow as tf

from io import BytesIO

from .tracker import Tracker
from ..utils.image_utils import imread_rgb



class Tensorboard(Tracker):
    """
    Tracker for `TensorBoard <https://github.com/tensorflow/tensorboard>`_.
    """

    def start(self, run_name, logdir):
        self.writer = tf.summary.FileWriter(os.path.join(logdir, "tensorboard", run_name))

    def add_tag(self, tag):
        pass

    def log_asset(self, name, file_path):
        pass

    def log_param(self, name, value, step=None):
        pass

    def log_metric(self, name, value, step=None):
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
        self.writer.add_summary(summary, step)


    def log_image(self, name, file_path):
        img = imread_rgb(file_path)

        is_success, img_buffer = cv.imencode(".jpg", img)
        buffer = BytesIO(img_buffer)

        img_summary = tf.Summary.Image(encoded_image_string=buffer.getvalue(),
                                       width=img.shape[1],
                                       height=img.shape[0])

        summary = tf.Summary(value=[tf.Summary.Value(tag=name, image=img_summary)])
        self.writer.add_summary(summary)


    def log_figure(self, name, figure):
        buffer = BytesIO()
        figure.savefig(buffer, format='png')
        buffer.seek(0)
        img = cv.imdecode(np.frombuffer(buffer.getbuffer(), np.uint8), -1)

        img_summary = tf.Summary.Image(encoded_image_string=buffer.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])

        summary = tf.Summary(value=[tf.Summary.Value(tag=name, image=img_summary)])
        self.writer.add_summary(summary)


    def end(self):
        self.writer.flush()
        self.writer.close()
