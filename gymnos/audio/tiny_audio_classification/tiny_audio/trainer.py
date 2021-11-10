#
#
#   Trainer
#
#

from dataclasses import dataclass
from ....base import BaseTrainer
from .hydra_conf import TinyAudioHydraConf
from .model import settings, create_model
import logging
from .utils import *
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from IPythson import display
from model import create_model, export_to_lite

@dataclass
class TinyAudioTrainer(TinyAudioHydraConf, BaseTrainer):
    """
    TODO: docstring for trainer
    """

    def prepare_data(self, root):
        print(root)

        speech_commands_dataset_basepath = root + "/audios"
        list_classes = self.sounds_to_detect
        classes = list_classes.split(",")
        sample_rate = self.sample_rate
        channels = self.channels
        count = 0
        count_train = 0

        fullpaths,targets,folds = create_file_lists()
        #Creting tf dataset
        fullpaths_ds = tf.data.Dataset.from_tensor_slices(
            (fullpaths, targets, folds))

        #Loading wav files
        wav_ds = fullpaths_ds.map(load_wav_map)
        #Splitting samples longer than 1sg
        split_wav_ds = wav_ds.flat_map(split_wav_map)
        split_wav_ds = split_wav_ds.filter(lambda x, y, z: wav_not_empty(x))

        #Creating spectograms for model
        spectrograms_ds = split_wav_ds.map(create_spectrogram_map)
        random_seed = self.random_seed

        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)

        #Catching the wav files
        cached_ds = spectrograms_ds.cache().take(300)

        #Splitting into train and validation datasets
        self.train_ds = cached_ds.filter(lambda spectrogram, label, fold: fold == 0)
        self.val_ds = cached_ds.filter(lambda spectrogram, label, fold: fold == 1)

        remove_fold_column = lambda spectrogram, label, fold: (tf.expand_dims(spectrogram, axis=-1), label)
        self.train_ds = self.train_ds.map(remove_fold_column)
        self.val_ds = self.val_ds.map(remove_fold_column)

        self.train_ds = self.train_ds.cache().shuffle(10, seed=random_seed).batch(32).prefetch(tf.data.AUTOTUNE)
        self.val_ds = self.val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)

        for spectrogram, _, _ in cached_ds.take(1):
            self.input_shape = tf.expand_dims(spectrogram, axis=-1).shape






    def train(self):

        input_shape = self.input_shape
        model = create_model()
        model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["Accuracy"],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(verbose=1, patience=25),
            tf.keras.callbacks.LearningRateScheduler(scheduler)
            ]

        model.summary()
        EPOCHS = 250
        history = model.fit(
        self.train_ds,
        validation_data=self.val_ds,
        epochs=self.epochs,
        callbacks=callbacks,
        )

        #Saving the trained model
        model.save('sounds_model')

        #Exporting model to tf lite
        export_to_lite(model)




    def test(self):
        pass   # OPTIONAL: test code
