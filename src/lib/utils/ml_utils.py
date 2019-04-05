#
#
#   ML Utils
#
#

import numpy as np
import pandas as pd


def train_val_test_split(X, y, train_size=0.6, val_size=0.2, test_size=0.2, seed=None, shuffle=True):
    len_data = len(X)

    train_num_samples = int(len_data * train_size)
    val_num_samples = int(len_data * val_size)
    test_num_samples = int(len_data * test_size)

    indices = np.arange(len_data)
    if shuffle:
        indices = np.random.RandomState(seed=seed).permutation(len_data)

    train_samples_index = train_num_samples
    val_samples_index = train_samples_index + val_num_samples
    test_samples_index = val_samples_index + test_num_samples

    if isinstance(X, (pd.Series, pd.DataFrame)):
        X = X.iloc
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.iloc

    X_train = X[indices[:train_samples_index]]
    X_val = X[indices[train_samples_index:val_samples_index]]
    X_test = X[indices[val_samples_index:test_samples_index]]

    y_train = y[indices[:train_samples_index]]
    y_val = y[indices[train_samples_index:val_samples_index]]
    y_test = y[indices[val_samples_index:test_samples_index]]

    return (X_train, X_val, X_test), (y_train, y_val, y_test)
