import logging
import os

from lib.core.dataset import Dataset
from lib.datasets.data_usage_test import DataUsageTest
from lib.datasets.unusual_data_usage_test import UnusualDataUsageTest
from lib.models.data_usage_holt_winters import DataUsageHoltWinters
from lib.models.unusual_data_usage_weighted_thresholds import UnusualDataUsageWT

# Parameters

# general parameters
path_logs = ""

# data usage model paremeters
n_preds = 17
min_historic = 5
flag_optimize_hiperparams = True
slen = 1
alpha = 0.716
beta = 0.029
gamma = 0.993

# unusual data usage model paremeters
execution_path = ""
sigma = 2.0
real_cum_last_day = 23.3

# logs
logging.basicConfig(filename=path_logs,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level="DEBUG")
logging.info("Start Data Usage Use Case Test")
logger = logging.getLogger('DataUsageUseCase')

logger.info("Loading Data Usage Test")

dut = DataUsageTest()
dut.download_and_prepare(dl_manager=None)
X_data_usage = dut.labels_
logger.debug("First label of X_data_usage:{0}".format(X_data_usage[0]))

logger.info("Loading Unusual Data Usage Test")

udut = UnusualDataUsageTest()
udut.download_and_prepare(dl_manager=None)
X_unusual_data_usage = udut.labels_
logger.debug("First label of X_unusual_data_usage:{0}".format(X_unusual_data_usage[0]))

logger.info("Predicting with Data Usage Holt Winters model")

duhw = DataUsageHoltWinters(n_preds=n_preds, min_historic=min_historic,
                            flag_optimize_hiperparams=flag_optimize_hiperparams, slen=slen, alpha=alpha, beta=beta,
                            gamma=gamma)
prediction = duhw.predict(X_data_usage)
logger.debug("Last element label of prediction:{0}".format(prediction[-1]))

logger.info("Restoring preprocessors of Unusual Data Usage WT")

dataset = Dataset("unusual_data_usage_test")
dataset.preprocessors.restore(os.path.join(execution_path, "artifacts/preprocessors.pkl"))

logger.info("Applying preprocessors of Unusual Data Usage WT")

samples = dataset.preprocessors.transform(X_unusual_data_usage)
logger.debug("First label of X_unusual_data_usage after apply preprocessors:{0}".format(samples[0]))

logger.info("Predicting with Unusual Data Usage WT model")

uduwt = UnusualDataUsageWT(sigma=sigma, pred_last_day=prediction[-1], pred_last_day_api_name=prediction[-1])
uduwt.restore(os.path.join(execution_path, "artifacts"))
result = uduwt.predict(samples)

logger.info("Result of anomaly in Data Usage Use Case:{0}".format(result))

logging.info("End Data Usage Use Case Test")
