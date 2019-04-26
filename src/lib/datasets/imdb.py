#
#
#   IMDB
#
#

import os
import pandas as pd

from .dataset import KaggleDataset


class IMDB(KaggleDataset):
    """
    Dataset with movie reviews for binary sentiment classification.

    The class labels are:

    +----------+--------------+
    | Label    | Description  |
    +==========+==============+
    | 0        | Negative     |
    +----------+--------------+
    | 1        | Positive     |
    +----------+--------------+

    Characteristics
        - **Classes**: 2
        - **Samples total**: 25 000
        - **Features**: texts
    """

    kaggle_dataset_name = "oumaimahourrane/imdb-reviews"
    kaggle_dataset_files = ["dataset.csv"]

    def read(self, download_dir):
        file_path = os.path.join(download_dir, self.kaggle_dataset_files[0])
        data = pd.read_csv(file_path, encoding="latin-1")
        features, labels = self.__features_labels_split(data)
        return features, labels


    def __features_labels_split(self, data):
        features = data["SentimentText"]
        labels = data["Sentiment"]
        return features, labels
