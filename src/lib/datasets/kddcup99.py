#
#
#   KDDCup99
#
#

import pandas as pd

from ..utils.io_utils import read_from_text
from .dataset import Dataset, DatasetInfo, Tensor, ClassLabel


class KDDCup99(Dataset):
    """
    The task is to build a network intrusion detector, a predictive model capable of distinguishing
    between "bad" connections,called intrusions or attacks, and "good" normal connections.

    Characteristics
        - **Classes**: 38
        - **Samples total**: xxxx
        - **Dimensionality**: [122]
        - **Features**: continuous and discrete

    References
    ----------
    `Kdd Cup '99 <http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html>`_
    """

    def _info(self):
        return DatasetInfo(
            features=Tensor(shape=[122]),
            labels=ClassLabel(num_classes=38)
        )

    def _download_and_prepare(self, dl_manager):
        paths = dl_manager.download_and_extract({
            "csv_data": "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz",
            "feature_names": "http://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.names"
        })

        self.csv_data_path_ = paths["csv_data"]
        self.feature_names_path_ = paths["feature_names"]


    def _load(self):
        columns = self.__read_features_names(self.feature_names_path_)
        data = pd.read_csv(self.csv_data_path_, names=columns, header=None)
        features, labels = self.__features_labels_split(data)
        features = pd.get_dummies(features)
        return features, pd.factorize(labels)[0]

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
