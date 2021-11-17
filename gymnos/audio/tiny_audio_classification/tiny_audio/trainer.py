#
#
#   Trainer
#
#

from dataclasses import dataclass
from ....base import BaseTrainer
from .hydra_conf import TinyAudioHydraConf
import logging
from .utils import *
import tensorflow as tf
from .model_tiny import *
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

@dataclass
class TinyAudioTrainer(TinyAudioHydraConf, BaseTrainer):


    def prepare_data(self, root):

        logger = logging.getLogger(__name__)
        logger.info("Preparing wavs files ...")
        speech_commands_dataset_basepath = root
        list_classes = self.sounds_to_detect
        classes = list_classes.split(",")

        count = 0
        count_train = 0

        fullpaths,targets,folds = create_file_lists(classes,speech_commands_dataset_basepath)

        fullpaths_ds = tf.data.Dataset.from_tensor_slices(
            (fullpaths, targets, folds))

        wav_ds = fullpaths_ds.map(load_wav_map)
        wav_ds = wav_ds.filter(lambda x, y, z: wav_not_empty(x))
        logger.info("Preparing spectrograms ...")
        spectogram = wav_ds.map(create_spectrogram_map)
        spectogram = spectogram.map(create_mel_map)

        split_wav_ds = wav_ds.filter(lambda x, y, z: wav_not_empty(x))
        # #Creating spectograms for model
        spectrograms_ds = spectogram.cache()


        # #Catching the spectrograms files
        self.cached_ds = spectrograms_ds.cache().take(len(targets))
        inputs_x = []
        inputs_y = []
        inputs_z = []
        for x,y,z in self.cached_ds:
            inputs_x.append(x.numpy().flatten())
            inputs_y.append(y)
            inputs_z.append(z)


        self.cached_ds = tf.data.Dataset.from_tensor_slices(
            (inputs_x, inputs_y, inputs_z))

        # #Splitting into train and validation datasets

        self.train_ds = self.cached_ds.filter(lambda spectrogram, label, fold: fold == 1)
        self.val_ds = self.cached_ds.filter(lambda spectrogram, label, fold: fold == 0)

        remove_fold_column = lambda spectrogram, label, fold: (tf.expand_dims(spectrogram, axis=-1), label)
        self.train_ds = self.train_ds.map(remove_fold_column)
        self.val_ds = self.val_ds.map(remove_fold_column)
        #
        self.train_ds = self.train_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
        self.val_ds = self.val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
        for spectrogram, _, _ in self.cached_ds.take(1):
            self.input_shape = tf.expand_dims(spectrogram, axis=-1).shape

        self.root = root

    def train(self):

        input_shape = self.input_shape
        logger = logging.getLogger(__name__)
        logger.info("Start training! ")

        model = create_model(self)
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        log_dir = self.root+"/"+"logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        early = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=0, patience=0, verbose=0,
                mode='auto', baseline=None, restore_best_weights=False
                )

        callbacks = [
            early,
            tf.keras.callbacks.LearningRateScheduler(scheduler),
            tensorboard_callback
            ]

        model.summary()
        EPOCHS = self.epochs
        history = model.fit(
        self.train_ds,
        validation_data=self.val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        )

        #Saving the trained model
        model.save(self.root+"/"+'sounds_model')

        #Exporting model to tf lite
        export_to_lite(model,self)




    def test(self):
        pass   # OPTIONAL: test code
