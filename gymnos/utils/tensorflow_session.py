#
#
#   TensorFlow Session
#
#

import tensorflow as tf


def build_tf_session_from_config(gpu_options=None, **kwargs):
    """
    Build tensorflow session with configuration from a dictionnary.

    Parameters
    ----------
    gpu_options: dict, optional
        Options for GPU. Available options: tf.GPUOptions
    **kwargs: any, optional
        Any configuration argument for TF session. Available options: tf.ConfigProto

    Returns
    --------
    sess: tf.Session
        Configured TensorFlow session.
    """
    gpu_options = gpu_options or {}
    gpu_options = tf.GPUOptions(**gpu_options)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, **kwargs))
