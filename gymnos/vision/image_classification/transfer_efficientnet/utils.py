#
#
#   Utils
#
#

import mlflow
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.loggers

from itertools import accumulate


def split_indices(num_samples, frac_list=None, shuffle=True, random_state=None):
    if frac_list is None:
        frac_list = [0.8, 0.1, 0.1]
    frac_list = np.asarray(frac_list)
    assert np.allclose(np.sum(frac_list), 1.), \
        'Expect frac_list sum to 1, got {:.4f}'.format(np.sum(frac_list))
    lengths = (num_samples * frac_list).astype(int)
    lengths[-1] = num_samples - np.sum(lengths[:-1])
    if shuffle:
        indices = np.random.RandomState(
            seed=random_state).permutation(num_samples)
    else:
        indices = np.arange(num_samples)

    return [indices[offset - length:offset] for offset, length in zip(accumulate(lengths), lengths)]


class MLFlowLogger(pl.loggers.mlflow.MLFlowLogger):

    @property
    def experiment(self):
        self._run_id = mlflow.active_run().info.run_id
        self._experiment_id = mlflow.active_run().info.experiment_id

        return super().experiment
