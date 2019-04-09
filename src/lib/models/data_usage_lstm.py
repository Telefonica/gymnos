#
#
#   Data usage LSTM
#
#

from keras import models, layers
from keras.preprocessing.sequence import TimeseriesGenerator

from .model import KerasModel
from ..utils.temporal_series_utils import mad_mean_error, nrmsd_error_norm, residual_analysis, rmse_train


class DataUsageLSTM(KerasModel):

    def __init__(self, input_shape, **hyperparameters):
        model = models.Sequential([
            layers.LSTM(100, activation='relu', input_shape=input_shape),
            layers.Dense(2)
        ])

        super().__init__(input_shape, model, **hyperparameters)

        self.n_preds = hyperparameters.get("n_preds", 2)

    def fit(self, X, y, batch_size=32, epochs=1, callbacks=None, val_data=None, verbose=1):
        """
        LSTM with data preprocessed using window method
        """
        y = y.flatten()
        y = y.reshape((int(-self.input_shape[0] / self.input_shape[1]), self.input_shape[1]))

        generator = TimeseriesGenerator(y, y, length=self.input_shape[0], batch_size=1)
        history = self.model.fit_generator(generator, steps_per_epoch=1, epochs=400, verbose=0)

        return history.history

    def evaluate(self, X, y, batch_size=32, verbose=0):
        """
        Evaluates with the following metrics.

        mean_error: (float) mean Absolute Deviation/Mean ratio
        normal_error: (float) root mean square deviation
        residual_error: (bool) True/False according to independence test result
        emc_error: (float) quadratic mean error for execution date

        """
        y = y.flatten()
        y = y[-self.input_shape[0] * self.input_shape[1]:].reshape((1, self.input_shape[0], self.input_shape[1]))

        y_pred = self.predict(y, batch_size=1, verbose=verbose)

        y = y.flatten().tolist()
        y_pred = y_pred.flatten().tolist()

        return {
            "mean_error": mad_mean_error(y[-self.n_preds:], y_pred[-self.n_preds:]),
            "normal_error": nrmsd_error_norm(y[-self.n_preds:], y_pred[-self.n_preds:]),
            "residual_error": residual_analysis(y[-self.n_preds:], y_pred[-self.n_preds:]),
            "emc_error": rmse_train(y[-self.n_preds:], y_pred[-self.n_preds:])
        }
