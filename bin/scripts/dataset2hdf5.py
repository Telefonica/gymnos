#
#
#   Convert dataset to HDF5
#
#

import os
import sys
import h5py
import pickle
import argparse
import numpy as np

from tqdm import tqdm

from gymnos.core.dataset import Dataset
from gymnos.services import DownloadManager
from gymnos.utils.data import DataLoader
from bin.scripts.gymnosd import read_preferences

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", help="Dataset name", type=str)
    parser.add_argument("-cs", "--chunk_size", help="Chunk size to load dataset", type=int)
    args = parser.parse_args()

    config = read_preferences()

    hdf5_filename = args.dataset_name + config["hdf5_file_extension"]
    hdf5_file_path = os.path.join(config["hdf5_datasets_dir"], hdf5_filename)

    os.makedirs(config["hdf5_datasets_dir"], exist_ok=True)

    if os.path.isfile(hdf5_file_path):
        confirmation = input(("HDF5 file for dataset {} already exists. " +
                              "Do you want to overwrite it? (y, n): ").format(args.dataset_name))

        if confirmation in ["y", "yes"]:
            os.remove(hdf5_file_path)
        else:
            print("Aborting ...")
            sys.exit()

    dataset = Dataset(name=args.dataset_name)

    dataset_info = dataset.dataset.info()

    dl_manager = DownloadManager(download_dir=config["download_dir"], extract_dir=config["extract_dir"],
                                 force_download=config["force_download"], force_extraction=config["force_extraction"])

    print("Downloading and preparing dataset")

    dataset.dataset.download_and_prepare(dl_manager)

    chunk_size = args.chunk_size
    if chunk_size is None:
        chunk_size = len(dataset.dataset)

    data_loader = DataLoader(dataset.dataset, batch_size=chunk_size)

    with h5py.File(hdf5_file_path) as h5f:
        features_shape = [len(dataset.dataset)] + dataset_info.features.shape
        labels_shape = [len(dataset.dataset)] + dataset_info.labels.shape

        labels_dtype = dataset_info.labels.dtype
        features_dtype = dataset_info.features.dtype

        # h5py can't handle str types, we need to convert them to variable length strings
        if features_dtype == str:
            features_dtype = h5py.special_dtype(vlen=str)

        features = h5f.create_dataset(config["hdf5_features_key"], compression=config["hdf5_compression"],
                                      shape=features_shape, dtype=features_dtype,
                                      compression_opts=config["hdf5_compression_opts"])
        labels = h5f.create_dataset(config["hdf5_labels_key"], compression=config["hdf5_compression"],
                                    shape=labels_shape, dtype=labels_dtype,
                                    compression_opts=config["hdf5_compression_opts"])

        features.attrs[config["hdf5_info_key"]] = np.string_(pickle.dumps(dataset_info.features))
        labels.attrs[config["hdf5_info_key"]] = np.string_(pickle.dumps(dataset_info.labels))

        for index, (X, y) in enumerate(tqdm(data_loader)):
            start = index * data_loader.batch_size
            end = start + data_loader.batch_size
            features[start:end] = X
            labels[start:end] = y

    print("Operation completed")
