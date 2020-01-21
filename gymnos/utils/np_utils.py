#
#
#   Numpy utils
#
#

import numpy as np


# Adapted from tensorflow/keras
def to_categorical(y, num_classes=None, dtype='int'):
    """
    Converts a class vector (integers) to binary class matrix.

    Parameters
    ------------
    y: array-like
        Class vector to be converted into a matrix (integers from 0 to num_classes).
    num_classes: int
        Total number of num_classes.
    dtype: str or type
        The data type expected by the output. Default: `'int'`.
    Returns:
        A binary matrix representation of the input. The num_classes axis is placed last.
    """
    y = np.array(y, dtype='int')

    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1

    n = y.shape[0]

    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1

    output_shape = input_shape + (num_classes,)

    categorical = np.reshape(categorical, output_shape)
    return categorical


def to_categorical_multilabel(y, num_classes, dtype="int"):
    """
    Converts a class array (list of lists of integers) to binary class matrix

    Parameters
    --------------
    y: array-like
        2D class matrix to be converted into a matrix (list of lists of integers from 0 to num_classes)
    num_classes: int
        Total number of num_classes
    dtype: str or type
        The data type expected by the output. Default: `'int'`.
    Returns
    ---------
        A binary matrix representation of the input.
    """
    nrows = np.shape(y)[0]

    categorical = np.zeros((nrows, num_classes), dtype=dtype)

    rows_to_activate = []
    cols_to_activate = []

    for row_index in range(len(y)):
        rows_to_activate.extend([row_index] * len(y[row_index]))
        cols_to_activate.extend(y[row_index])
        categorical[row_index, y[row_index]] = 1

    categorical[rows_to_activate, cols_to_activate] = 1

    return categorical


def label_binarize(y, num_classes=None, multilabel=False):
    """
    Converts array to one-hot encoded array with support for multilabel categories.
    Parameters
    -----------
    y: array-like
        Class vector
    num_classes: int
        Number of classes. If not provided, they will be computed.
    multilabel: bool
        Whether or not class vector is multilabel
    """
    if multilabel:
        return to_categorical_multilabel(y, num_classes=num_classes)
    else:
        return to_categorical(y, num_classes=num_classes)
