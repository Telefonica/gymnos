#
#
#   Accuracy Multilabel metric
#
#

import keras.backend as K


def accuracy_multilabel(y_true, y_pred):
    return K.mean(K.all(K.equal(y_true, K.round(y_pred)), axis=-1))
