#
#
#   ML Utils
#
#

import numpy as np


def train_val_test_split(X, y, train_size, val_size, test_size, seed=None):
    len_data = len(X)

    train_num_samples = int(len_data * train_size)
    val_num_samples = int(len_data * val_size)
    test_num_samples = int(len_data * test_size)

    shuffled_indices = np.random.RandomState(seed=seed).permutation(len_data)

    train_samples_index = train_num_samples
    val_samples_index = train_samples_index + val_num_samples
    test_samples_index = val_samples_index + test_num_samples

    X_train = X[shuffled_indices[:train_samples_index]]
    y_train = y[shuffled_indices[:train_samples_index]]

    X_val = X[shuffled_indices[train_samples_index:val_samples_index]]
    y_val = y[shuffled_indices[train_samples_index:val_samples_index]]

    X_test = X[shuffled_indices[val_samples_index:test_samples_index]]
    y_test = y[shuffled_indices[val_samples_index:test_samples_index]]

    return (X_train, X_val, X_test), (y_train, y_val, y_test)
