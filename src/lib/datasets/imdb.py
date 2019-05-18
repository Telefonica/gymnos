#
#
#   IMDB
#
#

import pandas as pd

from .dataset import Dataset, DatasetInfo, ClassLabel


class IMDB(Dataset):
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

    def _info(self):
        return DatasetInfo(
            features=str,
            labels=ClassLabel(names=["negative", "positive"])
        )

    def _download_and_prepare(self, dl_manager):
        self.csv_path_ = dl_manager.download_kaggle(dataset_name="oumaimahourrane/imdb-reviews",
                                                    file_or_files="dataset.csv")


    def _load(self):
        data = pd.read_csv(self.csv_path_, encoding="latin-1")
        features = data["SentimentText"]
        labels = data["Sentiment"]
        return features, labels
