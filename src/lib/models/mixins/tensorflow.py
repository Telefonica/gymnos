#
#
#   Tensorflow mixin
#
#

import os
import tensorflow as tf


class TensorFlowMixin:


    def restore(self, directory):
        saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(directory, "session.ckpt"))

    def save(self, directory):
        saver = tf.train.Saver()
        saver.save(self.sess, os.path.join(directory, "session.ckpt"))

    def __del__(self):
        self.sess.close()
