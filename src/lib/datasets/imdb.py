#
#
#   IMDB
#
#

import os
import pandas as pd

from keras.utils import to_categorical

from .dataset import KaggleDataset


class IMDB(KaggleDataset):
    """
    Kind: Classification
    Shape:
        features: [1]
        labels: [2]
    Description: >
        Dataset with movie reviews for binary sentiment classification.
    """

    kaggle_dataset_name = "oumaimahourrane/imdb-reviews"
    kaggle_dataset_files = ["dataset.csv"]

    def read(self, download_dir):
        file_path = os.path.join(download_dir, self.kaggle_dataset_files[0])
        data = pd.read_csv(file_path, encoding="latin-1")
        features, labels = self.__features_labels_split(data)
        return features, to_categorical(labels, 2)


    def __features_labels_split(self, data):
        features = data["SentimentText"]
        labels = data["Sentiment"]
        return features, labels
