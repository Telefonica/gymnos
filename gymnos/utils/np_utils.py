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
        The data type expected by the input. Default: `'float32'`.
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
        from sklearn.preprocessing import MultiLabelBinarizer

        binarizer = MultiLabelBinarizer(range(num_classes))
        return binarizer.fit_transform(y)
    else:
        return to_categorical(y, num_classes=num_classes)
