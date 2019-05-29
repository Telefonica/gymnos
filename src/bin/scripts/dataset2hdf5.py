#
#
#   Optimize dataset
#
#

import os
import sys
import h5py
import pickle
import argparse
import numpy as np

from tqdm import tqdm

from lib.core.dataset import Dataset
from lib.services import DownloadManager
from lib.utils.data import DataLoader

FEATURES_KEY = "features"
LABELS_KEY = "labels"
INFO_KEY = "info"
DEFAULT_CACHE_DIR = "hdf5_datasets"
FILENAME_FORMAT = "{dataset_name}.h5"
DOWNLOAD_DIR = "./downloads"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", help="Dataset name", type=str)
    parser.add_argument("-o", "--output_dir", help="Output directory", default=DEFAULT_CACHE_DIR, type=str)
    parser.add_argument("-cs", "--chunk_size", help="Chunk size to load dataset", type=int, default=1024)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    filename = FILENAME_FORMAT.format(dataset_name=args.dataset_name)
    file_path = os.path.join(args.output_dir, filename)

    if os.path.isfile(file_path):
        confirmation = input(("HDF5 file for dataset {} already exists. " +
                              "Do you want to overwrite it? (y, n): ").format(args.dataset_name))

        if confirmation in ["y", "yes"]:
            os.remove(file_path)
        else:
            print("Aborting ...")
            sys.exit()

    dataset = Dataset(name=args.dataset_name)

    dataset_info = dataset.dataset.info()

    dl_manager = DownloadManager(download_dir=DOWNLOAD_DIR)

    print("Downloading and preparing dataset")

    dataset.dataset.download_and_prepare(dl_manager)

    data_loader = DataLoader(dataset.dataset, batch_size=args.chunk_size)

    with h5py.File(file_path) as h5f:
        features_shape = [len(dataset.dataset)] + dataset_info.features.shape
        labels_shape = [len(dataset.dataset)] + dataset_info.labels.shape

        features_dtype = dataset_info.features.dtype
        if features_dtype == str:
            features_dtype = h5py.special_dtype(vlen=str)

        features = h5f.create_dataset(FEATURES_KEY, shape=features_shape, compression="gzip",
                                      dtype=features_dtype)
        labels = h5f.create_dataset(LABELS_KEY, shape=labels_shape, compression="gzip",
                                    dtype=dataset_info.labels.dtype)

        features.attrs[INFO_KEY] = np.string_(pickle.dumps(dataset_info.features))
        labels.attrs[INFO_KEY] = np.string_(pickle.dumps(dataset_info.labels))

        for index, (X, y) in enumerate(tqdm(data_loader)):
            start = index * data_loader.batch_size
            end = start + data_loader.batch_size
            features[start:end] = X
            labels[start:end] = y

    print("Operation completed")
