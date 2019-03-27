#
#
#   CNN
#
#

import numpy as np
import tensorflow as tf

from tqdm import trange
from pprint import pprint
from keras.callbacks import History

from .model import Model


class DogsVsCatsCNN(Model):

    def __init__(self, input_shape):
        super().__init__(input_shape)

        self.input = tf.placeholder(tf.float32, shape=[None] + input_shape)
        self.labels = tf.placeholder(tf.float32, shape=[None, 2])

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
            2
        )

        self.output_softmax = tf.nn.softmax(self.output)

        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.output)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.train_op = optimizer.minimize(self.loss)

        self.metrics = [tf.metrics.accuracy]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def fit(self, X, y, batch_size=32, epochs=1, callbacks=None, val_data=None, verbose=1):
        callbacks = callbacks or []

        history = History()

        callbacks.append(history)

        [cb.on_train_begin() for cb in callbacks]

        # Iterate by epoch
        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch, epochs))

            [cb.on_epoch_begin(epoch) for cb in callbacks]

            batch_pbar = trange(0, len(X), batch_size)
            # Iterate by batch
            for i in batch_pbar:
                loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.input: X[i:i + batch_size],
                                                                               self.labels: y[i:i + batch_size]})
                batch_pbar.set_description("Loss: {:.2f}".format(np.mean(loss)))

            logs = {}

            losses = []
            val_losses = []
            for i in trange(0, len(X), batch_size):
                loss = self.sess.run(self.loss, feed_dict={self.input: X[i: i + batch_size],
                                                           self.labels: y[i: i + batch_size]})
                losses.append(loss)

                if val_data is not None:
                    val_loss = self.sess.run(self.loss, feed_dict={self.input: val_data[0][i: i + batch_size],
                                                                   self.labels: val_data[1][i: i + batch_size]})
                    val_losses.append(val_loss)

            logs["loss"] = np.mean(losses)
            if val_data is not None:
                logs["val_loss"] = np.mean(val_losses)

            y_pred = self.predict(X)
            logs["acc"] = tf.metrics.accuracy(y, y_pred)

            if val_data is not None:
                y_pred = self.predict(val_data[0])
                logs["val_acc"] = tf.metrics.accuracy(val_data[1], y_pred)

            pprint(logs)

            [cb.on_epoch_end(epoch, logs=logs) for cb in callbacks]

        [cb.on_train_end() for cb in callbacks]

        return history.history


    def predict(self, X, batch_size=32, verbose=0):
        predictions = np.empty(len(X))
        for i in trange(0, len(X), batch_size):
            predictions[i: i + batch_size] = self.sess.run(self.output_softmax,
                                                           feed_dict={self.input: X[i:i + batch_size]})

        return predictions

    def evaluate(self, X, y, batch_size=32, verbose=0):
        y_pred = self.predict(X, batch_size=batch_size, verbose=verbose)
        return {
            "acc": tf.metrics.accuracy(y, y_pred)
        }

    def restore(self, file_path):
        saver = tf.train.Saver()
        saver.save(self.sess, file_path)


    def save(self, file_path):
        saver = tf.train.Saver()
        saver.restore(self.sess, file_path)
