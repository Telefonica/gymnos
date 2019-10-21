#
#
#   Reddit self-post classification
#
#

import os
import pandas as pd

from .dataset import IterableDataset, Array, ClassLabel


class RedditSelfPostClassification(IterableDataset):

    def __init__(self, blocksize=512):
        self.blocksize = blocksize

    def download_and_prepare(self, dl_manager):
        self.rspct_path_ = dl_manager["kaggle"].download(dataset_name="mswarbrickjones/reddit-selfposts",
                                                         file_or_files="rspct.tsv")

        self.num_rows_ = sum(1 for _ in open(self.rspct_path_)) - 1  # substract header

    @property
    def features_info(self):
        return Array([], str)

    @property
    def labels_info(self):
        return ClassLabel(names_file=os.path.join(os.path.dirname(__file__),
                                                  "reddit_self_post_classification_labels.txt"))

    def __iter__(self):
        labels = self.labels_info

        for chunk in pd.read_csv(self.rspct_path_, sep="\t", iterator=True, chunksize=self.blocksize):
            for index, row in chunk.iterrows():
                text = row.title + " " + row.selftext
                yield text, labels.str2int(row.subreddit)

    def __len__(self):
        return self.num_rows_
