#
#
#   KDDCup99
#
#

import os
import pandas as pd

from keras.utils import to_categorical
from .dataset import PublicDataset
from ..utils.io_utils import read_from_text


class KDDCup99(PublicDataset):
    """
    Kind: Classification
    Shape:
        features: []
        labels: []
    Description: >
        The task is to build a network intrusion detector, a predictive model capable of distinguishing
        between "bad" connections,called intrusions or attacks, and "good" normal connections.
    """

    public_dataset_files = [
        "http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz",
        "http://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.names"
    ]

    def read(self, download_dir):
        data_file_path = os.path.join(download_dir, "corrected")
        feature_names_file_path = os.path.join(download_dir, "kddcup.names")
        columns = self.__read_features_names(feature_names_file_path)
        data = pd.read_csv(data_file_path, names=columns, header=None)
        features, labels = self.__features_labels_split(data)
        return features, to_categorical(labels, 2)

    def __read_features_names(self, file_path):
        text = read_from_text(file_path)
        lines = text.splitlines()[1:]  # first row = target names
        feature_names = [line.split(":")[0] for line in lines]
        feature_names.append("label")
        return feature_names

    def __features_labels_split(self, data):
        features = data.drop(columns="label")
        labels = data["label"]
        return features, labels
