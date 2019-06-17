#
#
#   Keras metrics
#
#

import keras.backend as K


def accuracy_multilabel(y_true, y_pred):
    """
    Compute accuracy for multilabel data. A row is considered right if all labels match.

    Parameters
    ----------
    y_true: tensor
        True labels.
    y_pred: tensor
        Predicted labels

    Returns
    -------
    metric: tensor
        Multilabel accuracy
    """
    return K.mean(K.all(K.equal(y_true, K.round(y_pred)), axis=-1))


def precision(y_true, y_pred):
    """
    Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.

    Parameters
    ----------
    y_true: tensor
        True labels.
    y_pred: tensor
        Predicted labels

    Returns
    -------
    metric: tensor
        Precision
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_score = true_positives / (predicted_positives + K.epsilon())
    return precision_score
