
#
#
#   SyntheticOkAura
#
#

import numpy as np
import os
import csv

from glob import glob
from ..utils.audio_utils import load_wav_file
from .dataset import Dataset, Array, ClassLabel

class SyntheticOkAura(Dataset):
    """
    A 5000-samples dataset for key word detection based on 'Ok Aura' utterance.

    Parameters
    ==========
    size: int
        Dataset size. 
            - Default size is 250 audio samples
            - Maximum size is 5000 audio samples
    """

    def __init__(self, size=250):
        self.size = size

    @property
    def features_info(self):
        return Array(shape=[5511, 101], dtype=np.float32)

    @property
    def labels_info(self):
        return Array(shape=[], dtype=np.float32)

    def download_and_prepare(self, dl_manager):
        self.dl_path = dl_manager["sofia"].download({"soka": "sofia://datasets/23"})
        self.ext_path = dl_manager.extract(self.dl_path)
        wav_files_paths = sorted(glob(os.path.join(self.ext_path["soka"],"synthetic_ok_aura","*.wav")))
        csv_files_paths = sorted(glob(os.path.join(self.ext_path["soka"],"synthetic_ok_aura","*.csv")))
        self.audio_paths_ = np.array(wav_files_paths)
        self.label_paths_ = np.array(csv_files_paths)

    def __getitem__(self, index):
        labels = []
        rate, audio = load_wav_file(self.audio_paths_[index])
        with open(self.label_paths_[index], 'r') as file:
            reader = csv.reader(file, delimiter='|')
            for row in reader:
                labels.append(row)

        y = np.array(labels, np.float16)
        y = np.swapaxes(y, 0, 1)

        return audio, y
        
    def __len__(self):
        return self.size

