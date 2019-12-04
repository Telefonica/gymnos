#
#
#   IMDB
#
#

import logging
import pandas as pd

from .dataset import Dataset, ClassLabel, Array

logger = logging.getLogger(__name__)

KAGGLE_DATASET_NAME = "oumaimahourrane/imdb-reviews"
KAGGLE_DATASET_FILENAME = "dataset.csv"


class IMDB(Dataset):
    """
    Services: :class:`~gymnos.services.kaggle.Kaggle`

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

    @property
    def features_info(self):
        return Array(shape=[], dtype=str)

    @property
    def labels_info(self):
        return ClassLabel(names=["negative", "positive"])

    def download_and_prepare(self, dl_manager):
        csv_path = dl_manager["kaggle"].download(dataset_name=KAGGLE_DATASET_NAME,
                                                 file_or_files=KAGGLE_DATASET_FILENAME)
        logger.info("Reading CSV")
        self.data_ = pd.read_csv(csv_path, encoding="latin-1")

    def __getitem__(self, index):
        row = self.data_.iloc[index]
        return row["SentimentText"], row["Sentiment"]

    def __len__(self):
        return len(self.data_)
