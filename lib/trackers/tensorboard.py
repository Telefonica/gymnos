#
#
#
#   Tensorboard Tracker
#
#

import numpy as np
import tensorflow as tf

from io import BytesIO
from PIL import Image

from .tracker import Tracker


class Tensorboard(Tracker):

    def __init__(self, logdir, tf_graph=None):
        self.writer = tf.summary.FileWriter(logdir, tf_graph)


    def log_metric(self, name, value, step=None):
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
        self.writer.add_summary(summary, step)


    def log_image(self, name, file_path):
        pil_img = Image.open(file_path)

        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        buffer = BytesIO()
        pil_img.save(buffer, format='png')

        img_summary = tf.Summary.Image(encoded_image_string=buffer.getvalue(),
                                       width=pil_img.size[0],
                                       height=pil_img.size[1])

        summary = tf.Summary(value=[tf.Summary.Value(tag=name, image=img_summary)])
        self.writer.add_summary(summary)


    def log_figure(self, name, figure):
        buffer = BytesIO()
        figure.savefig(buffer, format='png')
        buffer.seek(0)
        img = Image.open(buffer)
        img_ar = np.array(img)

        img_summary = tf.Summary.Image(encoded_image_string=buffer.getvalue(),
                                       height=img_ar.shape[0],
                                       width=img_ar.shape[1])

        summary = tf.Summary(value=[tf.Summary.Value(tag=name, image=img_summary)])
        self.writer.add_summary(summary)


    def log_model_graph(self, graph):
        self.writer.add_graph(graph)


    def end(self):
        self.writer.flush()
        self.writer.close()
