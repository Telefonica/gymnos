#
#
#  Unusual Data Usage Weighted Thresholds
#
#

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from .model import Model


class UnusualDataUsageWeightedThresholds(Model):

    def __init__(self, input_shape, **hyperparameters):
        super().__init__(input_shape)

        self.input = self.input_shape
        self.sigma = hyperparameters.pop("sigma")
        self.pred_last_day_api_name = hyperparameters.pop("pred_last_day_api_name")

    def fit(self, X, y, batch_size=32, epochs=1, callbacks=None, val_data=None, verbose=1):
        pass

    def predict(self, X, batch_size=32, verbose=0):
        """
        This function labels consumptions associated as anomalous using an algorithm. Return two new fields:
        - deviation: variable that in tha main script will be rename as
                     deviation_qt used to calculate anomaly threshold and scoring the anomaly in case that exists
        - anomaly_ind: binary variable that in the main script will be rename as anomaly_ind
                      (1 if it is anomaly or 0 in another case)

        The anomaly detection algorithm consist of the next steps in the function:

        1. If there are at least 2 months of historic we continue if another case we return standard values
          (anomaly_id=0, deviation_qt None) in the two new fields of the output.
           If continue in the process, we average the most recent consumption and the rest of the consumptions
           including the most recent. This way we can empower last consumption in this mean and we continue
           with the step 3.
        2. We calculate standard deviation bases on the the mean resulted of step 2.
           In the end of the step, we multiply by sigma value (it recommendable to define it as 2.0 in config.ini
           as parameter and divide it by 2.
        3. In the last part of the function, we set the threshold for being anomalous. The requirements are:
           4.1 The prediction at the end of cycle is upper than sum of mean of step 2 and standard deviation
               of step 3.
           4.2 The prediction value is upper than the most recent consumption.


        Returns:
            the two new values that will be rename in the main script:
                - deviation: deviation resulted of step 3 in cases that has sense.
                - anomaly_ind: binary field with 1 if it is anomaly or 0 in another case.

            """

        # Step 1

        if len([val for val in X if val > 0]) >= 2:

            history_moving_avg = np.array(X).astype(float).mean()
            monthly_moving_avg = X[-1]

            avg = (float(history_moving_avg) + float(monthly_moving_avg)) / 2

            # Step 2
            history_moving_desv = np.sqrt(np.sum([(x - avg) ** 2 for x in list(X)]) / len(X) - 1)

            desv = float(history_moving_desv) * float(self.sigma)

            # Step 3
            if self.pred_last_day_api_name is not None:
                if ((float(self.pred_last_day_api_name) > avg + float(desv)) & (
                        float(self.pred_last_day_api_name) > float(monthly_moving_avg))):  # anomaly

                    return str(1)

                else:  # not anomaly

                    return str(0)
        else:

            return str(0)

    def evaluate(self, X, y, batch_size=32, verbose=0):
        """
        Evaluates in several anomaly stages using the following metrics:

        f1_score
        precision_score
        recall_score

        """
        y_pred = self.predict(X, batch_size=batch_size, verbose=verbose)

        pred_last_day = y["pred_last_day"]
        real_cum_last_day = y["real_cum_last_day"]

        # Execute to several stages
        stages = [0.05, 0.1, 0.15, 0.2, 0.25, 0.8, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 9, 10]

        result = []
        for i in range(len(stages)):
            scen = stages[i]
            if (pred_last_day > (1 + scen) * real_cum_last_day) or (pred_last_day < (1 - scen) * real_cum_last_day):
                result.append(['0', y_pred])
            else:
                result.append(['1', y_pred])

        return {
            "f1_score": dict(
                zip(stages, [f1_score([val[0]], [val[1]], average='micro') for val in result])),
            "precision_score": dict(
                zip(stages, [precision_score([val[0]], [val[1]], average='micro') for val in result])),
            "recall_score": dict(
                zip(stages, [recall_score([val[0]], [val[1]], average='micro') for val in result]))
        }

    def restore(self, file_path):
        pass

    def save(self, file_path):
        pass
