#
#
#   CNN
#
#

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf

from tqdm import trange
from collections import defaultdict

from .model import Model
from .mixins import TensorFlowSaverMixin


class DogsVsCatsCNN(TensorFlowSaverMixin, Model):
    """
    Convolutional neuronal network developed to solve Dogs vs Cats image classification
    task (:class:`gymnos.datasets.dogs_vs_cats.DogsVsCats`).

    Note
    ----
    This model can be useful to see the development of a Tensorflow model on the platform.

    Parameters
    ----------
    input_shape: list
        Data shape expected.
    classes: int, optional
        Optional number of classes to classify images into. This is useful if
        you want to train this model with another dataset.
    session: tf.Session, optional
        Tensorflow Session

    Note
    ----
    This model requires one-hot encoded labels.

    Examples
    --------
    .. code-block:: py

        DogsVsCatsCNN(
            input_shape=[120, 120, 3],
            classes=2,
            session=None
        )
    """

    def __init__(self, input_shape, classes=2, sess=None):
        self.input = tf.placeholder(tf.float32, shape=[None] + input_shape)
        self.labels = tf.placeholder(tf.float32, shape=[None, classes])
        self.is_training = tf.placeholder(tf.bool)

        L1 = tf.layers.conv2d(self.input, 32, [3, 3], activation=tf.nn.relu)
        L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
        L1 = tf.layers.dropout(L1, 0.7, self.is_training)

        L2 = tf.layers.conv2d(L1, 64, [3, 3], activation=tf.nn.relu)
        L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
        L2 = tf.layers.dropout(L2, 0.7, self.is_training)

        L3 = tf.contrib.layers.flatten(L2)
        L3 = tf.layers.dense(L3, 256, activation=tf.nn.relu)
        L3 = tf.layers.dropout(L3, 0.5, self.is_training)

        self.output = tf.layers.dense(L3, classes, activation=None)

        self.output_softmax = tf.nn.softmax(self.output)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.output))

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = optimizer.minimize(self.loss)

        self.sess = sess or tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def fit(self, X, y, epochs=10, batch_size=64, validation_split=0):
        """
        Parameters
        ----------
        X: array_like
            Features
        y: array_like
            Targets.
        epochs: int, optional
            Number of epochs to train.
        batch_size: int, optional
            Number of samples that will be propagated.
        validation_split: float, optional
            Fraction of the training data to be used as validation data. Between 0 and 1.
        Returns
        -------
        dict
            Training metrics (accuracy and loss)
        """
        metrics = defaultdict(list)

        val_data = []
        if validation_split and 0.0 < validation_split < 1.0:
            X, X_val, y, y_val = train_test_split(X, y, test_size=validation_split)
            val_data = [X_val, y_val]

        # Iterate by epoch
        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch + 1, epochs))

            losses = []
            batch_pbar = trange(0, len(X), batch_size)
            # Iterate by batch
            for i in batch_pbar:
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.input: X_batch,
                                                                               self.labels: y_batch,
                                                                               self.is_training: True})
                losses.append(np.mean(loss))
                batch_pbar.set_description("Loss: {:.2f}  Acc: {:.2f}".format(losses[-1],
                                                                              self.evaluate(X_batch, y_batch)["acc"]))

            print("Evaluating epoch")

            metrics["loss"].append(np.mean(losses))
            for metric_name, value in self.evaluate(X, y).items():
                metrics[metric_name].append(value)

            if val_data:
                val_loss = self.sess.run(self.loss, feed_dict={self.input: val_data[0],
                                                               self.labels: val_data[1]})
                metrics["val_loss"].append(np.mean(val_loss))
                print("val_loss={}".format(val_loss))

                for metric_name, value in self.evaluate(*val_data).items():
                    metrics["val_" + metric_name].append(value)
                    print("val_{}={}".format(metric_name, value))

        return metrics

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=-1)

    def predict_proba(self, X):
        proba = self.sess.run(self.output_softmax, feed_dict={self.input: X, self.is_training: False})
        return proba

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            "acc": accuracy_score(np.argmax(y, axis=1), y_pred)
        }
