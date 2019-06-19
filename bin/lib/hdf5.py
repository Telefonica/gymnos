#
#
#   HDF5
#
#

import h5py
import pickle
import gymnos.datasets


class HDF5Dataset(gymnos.datasets.Dataset):
    """
    Create dataset from HDF5 file.

    Parameters
    ----------
    file_path: str
        HDF5 dataset file path.
    features_key: str
        Key to load features.
    labels_key: str
        Key to load labels
    info_key: str
        Key to load info
    """

    def __init__(self, file_path, features_key="features", labels_key="labels", info_key="info"):
        self.features_key = features_key
        self.labels_key = labels_key
        self.info_key = info_key

        self.data = h5py.File(file_path, mode="r")


    def info(self):
        features_info = pickle.loads(self.data[self.features_key].attrs[self.info_key])
        labels_info = pickle.loads(self.data[self.labels_key].attrs[self.info_key])
        return gymnos.datasets.DatasetInfo(
            features=features_info,
            labels=labels_info
        )

    def download_and_prepare(self, dl_manager):
        """
        It does nothing.
        """
        pass

    def __getitem__(self, index):
        return (
            self.data[self.features_key][index],
            self.data[self.labels_key][index]
        )

    def __len__(self):
        return len(self.data[self.features_key])  # len(features) == len(labels)
