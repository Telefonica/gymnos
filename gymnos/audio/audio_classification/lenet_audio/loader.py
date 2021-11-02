import random
import json
import os

import torch
import torchaudio
import numpy as np
import pandas as pd

def load_train_partitions(path, window_size=24000, fs=16000):

    # Augments
    augments = ['white_noise']
    augments = {key: True for key in augments}

    # Read class index
    with open(path + '/metadata/classes_index.json') as f:
        class_index = json.load(f)
    
    # Load train set
    train_df = pd.read_csv(path + '/metadata/train.tsv', sep='\t')
    train_ID_list = list(train_df['Sample_ID'])

    # Load validation set
    validation_df = pd.read_csv(path + '/metadata/dev.tsv', sep='\t')
    validation_ID_list = list(validation_df['Sample_ID'])

    # Generate Datasets
    train_dataset = AudioDataset(
        path,
        train_ID_list,
        train_df,
        class_index,
        window_size=window_size,
        fs=fs,
        augments=augments
        )
    validation_dataset = AudioDataset(
        path,
        validation_ID_list,
        validation_df,
        class_index,
        window_size=window_size,
        fs=fs,
        augments=augments
        )

    return train_dataset, validation_dataset


def load_test_partition(path, window_size=24000, fs=16000, augments=None):

    # Augments
    augments = ['white_noise']
    augments = {key: False for key in augments}

    # Read class index
    with open(path + '/metadata/classes_index.json') as f:
        class_index = json.load(f)
    
    # Load train set
    test_df = pd.read_csv(path + '/metadata/test.tsv', sep='\t')
    test_ID_list = list(test_df['Sample_ID'])

    # Generate Datasets
    test_dataset = AudioDataset(
        path,
        test_ID_list,
        test_df,
        class_index,
        window_size=window_size,
        fs=fs,
        augments=augments
        )

    return test_dataset


class AudioDataset(torch.utils.data.Dataset):
    """
    Torch dataset for lazy load.
    """
    def __init__(self, path, list_IDs, dataframe, class_index, window_size=24000, fs=16000, augments=None):

        # data path
        self.path = path
        
        self.window_size = window_size
        self.fs = fs # Hz

        # Data information
        self.list_IDs = list_IDs
        self.dataframe = dataframe
        self.n_samples = len(list_IDs)
        self.class_index = class_index

        # Data augments
        self.augments = [k for k,v in augments.items() if v == True]
        self.white_noise = True if augments['white_noise'] else None

        # Number of classes
        self.n_classes = len(self.get_unique_classes())
        self.classes = self.get_unique_classes()

    def __len__(self):
        """
        Denote dataset sample.
        """
        return len(self.list_IDs)

    def get_unique_classes(self):
        """
        Get unique classes from multilabel dataset.
        """
        return list(self.dataframe['Label'].unique())
    
    def get_class_weigths(self):
        """
        Get class weigths.
        """
        # Calculate the real value counts for each class
        unique_classes = self.get_unique_classes()
        real_counts = dict.fromkeys(unique_classes, 0)
        value_counts = self.dataframe['Label'].value_counts().to_dict()

        for unique_class in unique_classes:
            for value_count in value_counts:
                if(unique_class in value_count):
                    real_counts[unique_class] += value_counts[value_count]

        # Calculate class weigths
        class_weigths = dict.fromkeys(unique_classes, 1.0)
        for unique_class in unique_classes:
            class_weigths[unique_class] /= real_counts[unique_class]

        return class_weigths

    def get_sample_weigths(self):
        class_weigths = self.get_class_weigths()
        samples_labels = self.dataframe['Label_Index']

        # Get a dict with the classes index and weigths
        index_weigths = {}
        for index in self.class_index:
            index_weigths[index] = class_weigths[self.class_index[index]]

        samples_weight = np.zeros(len(samples_labels))

        iterator = 0
        for sample_label in samples_labels:
            samples_weight[iterator] += index_weigths[str(sample_label)]
            iterator += 1

        return samples_weight

    def get_number_of_classes(self):
        return self.n_classes

    def __repr__(self):
        """
        Data infromation
        """
        repr_str = (
            "Number of samples: " + str(self.n_samples) + "\n"
            "Window size: " + str(self.window_size) + "\n"
            "Augments: " + str(self.augments) + "\n"

        )
        return repr_str
        
    def __getitem__(self, index):
        """
        Get a single sample
        Args:
            index: index to recover a single sample
        Returns:
            x,y: features extracted and label
        """
        # Select sample
        ID = self.list_IDs[index]

        # Read audio
        start = 0.0
        end = self.dataframe.set_index('Sample_ID').at[ID, 'Audio_Length']
        audio_path = self.dataframe.set_index('Sample_ID').at[ID, 'Sample_Path']
        audio = self.__read_wav(audio_path, start, end)
        
        # Prepare audio
        audio = self.__prepare_audio(audio)
        
        # Get label
        label = self.dataframe.set_index('Sample_ID').at[ID, 'Label_Index']

        # One-hot encoding
        target = self.__one_hot_encoding(label)

        return ID, audio, target

    def __one_hot_encoding(self, label):
        target = torch.eye(len(self.class_index))[int(label)]
        return target.float()

    def __read_wav(self, filepath, start, end):
        """
        Read audio wave file applying normalization with respecto of the maximum of the signal
        Args:
            filepath: audio file path
        Returns:
            audio_signal: numpy array containing audio signal
        """
        audio_signal, _ = torchaudio.load(
            os.path.join(self.path, filepath),
            frame_offset=int(start*self.fs),
            num_frames=int((end-start)*self.fs)
            )
        audio_signal = audio_signal[0].numpy()
        if(np.max(np.abs(audio_signal)) == 0.0):
            print('Problem with audio: {} start at {} and end at {}'.format(filepath, start, end))

        audio_signal = self.__normalize_audio(audio_signal)
        return audio_signal
    
    def __normalize_audio(self, audio, eps=0.001):
        """
        Peak normalization.
        """
        return (audio.astype(np.float32) / float(np.amax(np.abs(audio)))) + eps

    def __prepare_audio(self, audio_signal):
        """
        Adapt audio clip to  window size. Crops if larger and pads if shorter
        """

        # Adapt sample to windows size
        audio_length = audio_signal.shape[0]
        if(audio_length >= self.window_size):
            
            # If audio is bigger than window size use random crop: random shift
            left_bound = random.randint(0, audio_length - self.window_size)
            right_bound = left_bound + self.window_size
            audio_signal = audio_signal[left_bound:right_bound]
            
        else:
            # If the audio is smaller than the window size: pad original signal with 0z
            padding = self.window_size - audio_length
            bounds_sizes = np.random.multinomial(padding, np.ones(2)/2, size=1)[0]
            audio_signal = np.pad(
                audio_signal,
                (bounds_sizes[0], bounds_sizes[1]),
                'constant',
                constant_values=(0, 0)
                )

        # Add white noise
        if(self.white_noise):
            noise = (np.random.normal(0,1.3,len(audio_signal))*32).astype('int16')
            noise = noise.astype(np.float32) / float(np.iinfo(noise.dtype).max)
            audio_signal += noise

        return audio_signal

