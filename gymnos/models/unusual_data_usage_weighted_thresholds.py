#
#
#  Unusual Data Usage Weighted Thresholds
#
#

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from .model import Model


class UnusualDataUsageWT(Model):
    """
    This model labels consumptions associated as anomalous using an algorithm. It was developed to solve Unusual
    Data Usage dataset.

    The anomaly detection algorithm consist of the next steps in the function:

    1. If there are at least 2 months of historic we continue if another case we return
    standard values (anomaly_id=0, deviation_qt None) in the two new fields of the output.
    If continue in the process, we average the most recent consumption and the rest of the consumptions including
    the most recent. This way we can empower last consumption in this mean and we continue with the step 3.

    2. We calculate standard deviation bases on the the mean resulted of step 2. In the end of the step, we multiply
    by sigma value (it recommendable to define it as 2.0 in config.ini as parameter and divide it by 2).

    3. In the last part of the function, we set the threshold for being anomalous. The requirements are:

       3.1 The prediction at the end of cycle is upper than sum of mean of step 2 and standard deviation of step 3.

       3.2 The prediction value is upper than the most recent consumption.

    Parameters
    ----------
    sigma: float, optional
        TODO
    pred_last_day_api_name: float, optional
        TODO
    pred_last_day: int, float
        TODO
    real_cum_last_day: int, float
        TODO
    """

    def __init__(self, sigma=2.0, pred_last_day_api_name=20.3, pred_last_day=23, real_cum_last_day=33):
        self.sigma = sigma
        self.pred_last_day_api_name = pred_last_day_api_name
        self.pred_last_day = pred_last_day
        self.real_cum_last_day = real_cum_last_day

    def fit(self, X, y):
        """
        This model doesn't need to fit to training data.
        """
        return {}

    def predict(self, X):
        """
        Returns
        -------
            deviation:
                Deviation resulted of step 3 in cases that has sense.
            anomaly_ind:
                Binary field with 1 if it is anomaly or 0 in another case.
        """

        # Step 1

        if len([val for val in list(X) if val > 0]) >= 2:

            history_moving_avg = np.array(list(X)).astype(float).mean()
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

    def evaluate(self, X, y):
        """
        Returns
        -------
        f1_score
        precision_score
        recall_score
        """
        y_pred = self.predict(X)

        # Execute to several stages
        stages = [0.05, 0.1, 0.15, 0.2, 0.25, 0.8, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 9, 10]

        result = []
        for i in range(len(stages)):
            scen = stages[i]
            if (self.pred_last_day > (1 + scen) * self.real_cum_last_day) or (self.pred_last_day < (1 - scen) *
                                                                              self.real_cum_last_day):
                result.append(['0', y_pred])
            else:
                result.append(['1', y_pred])

        return {
            "stages": stages,
            "stages_f1_score": [f1_score([val[0]], [val[1]], average='micro') for val in result],
            "stages_precision_score": [precision_score([val[0]], [val[1]], average='micro') for val in result],
            "stages_recall_score": [recall_score([val[0]], [val[1]], average='micro') for val in result]
        }

    def restore(self, save_path):
        pass

    def save(self, save_path):
        pass
