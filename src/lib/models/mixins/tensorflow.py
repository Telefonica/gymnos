#
#
#   Tensorflow mixin
#
#

import os
import tensorflow as tf


class TensorFlowMixin:
    """
    Mixin to write TensorFlow methods. It provides implementation for ``save`` and ``restore`` methods.

    Attributes
    ----------
    sess: tf.Session
        TensorFlow session.
    """

    def restore(self, save_path):
        """
        Restore session from checkpoint

        Parameters
        ----------
        save_path: str
            Path (Directory) where session is saved.
        """
        saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(save_path, "session.ckpt"))

    def save(self, save_path):
        """
        Save session.

        Parameters
        ----------
        save_path: str
            Path (Directory) to save session.
        """
        saver = tf.train.Saver()
        saver.save(self.sess, os.path.join(save_path, "session.ckpt"))

    def __del__(self):
        self.sess.close()
