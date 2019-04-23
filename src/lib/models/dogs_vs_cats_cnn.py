#
#
#   CNN
#
#

import numpy as np
import tensorflow as tf

from tqdm import trange
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .model import Model
from .mixins import TensorFlowMixin


class DogsVsCatsCNN(Model, TensorFlowMixin):

    def __init__(self, input_shape, classes=2, session=None):
        self.input = tf.placeholder(tf.float32, shape=[None] + input_shape)
        self.labels = tf.placeholder(tf.float32, shape=[None, classes])

        conv_1 = tf.layers.conv2d(
            self.input,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu
        )

        pool_1 = tf.layers.max_pooling2d(
            conv_1,
            strides=2,
            pool_size=[2, 2]
        )

        conv_2 = tf.layers.conv2d(
            pool_1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu
        )

        pool_2 = tf.layers.max_pooling2d(
            conv_2,
            strides=2,
            pool_size=[2, 2],
        )

        flat = tf.layers.flatten(pool_2)

        dense = tf.layers.dense(
            flat,
            1024,
            activation=tf.nn.relu
        )

        dropout = tf.layers.dropout(
            dense,
            rate=0.4,
        )

        self.output = tf.layers.dense(
            dropout,
            classes
        )

        self.output_softmax = tf.nn.softmax(self.output)

        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.output)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.train_op = optimizer.minimize(self.loss)

        session = session or {}
        self.sess = tf.Session(config=tf.ConfigProto(**session))
        self.sess.run(tf.global_variables_initializer())


    def fit(self, X, y, epochs=10, batch_size=32, validation_split=0):
        metrics = defaultdict(list)

        val_data = []
        if validation_split and 0.0 < validation_split < 1.0:
            X, X_val, y, y_val = train_test_split(X, y, test_size=validation_split)
            val_data = [X_val, y_val]

        # Iterate by epoch
        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch, epochs))

            losses = []
            batch_pbar = trange(0, len(X), batch_size)
            # Iterate by batch
            for i in batch_pbar:
                loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.input: X[i:i + batch_size],
                                                                               self.labels: y[i:i + batch_size]})
                losses.append(np.mean(loss))
                batch_pbar.set_description("Loss: {:.2f}".format(losses[-1]))

            metrics["loss"].append(np.mean(losses))
            for metric_name, value in self.evaluate(X, y).items():
                metrics[metric_name].append(value)

            if val_data:
                val_loss = self.sess.run(self.loss, feed_dict={self.input: val_data[0],
                                                               self.labels: val_data[1]})
                metrics["val_loss"].append(np.mean(val_loss))
                for metric_name, value in self.evaluate(*val_data).items():
                    metrics["val_" + metric_name].append(value)

        return metrics


    def predict(self, X):
        predictions = self.sess.run(self.output_softmax, feed_dict={self.input: X})
        return predictions

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            "acc": accuracy_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1))
        }
