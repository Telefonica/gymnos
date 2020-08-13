#
#
#   Fourth Platform
#
#

import json
import time
import base64
import logging

from .. import config

from ..utils.cli_colors import color
from ..utils.json_utils import NumpyEncoder, default
from ..utils.lazy_imports import lazy_imports as lazy
from .execution_environment import ExecutionEnvironment

from dateutil.parser import parse
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class FourthPlatformAPI:
    """
    API client to interact with 4th platform

    Parameters
    ------------
    client_id: str
        4th Platform client ID.
    password: str
        4th Platform client password
    autologin: bool
        Whether or not automatically re-login if JSON token has expired
    """

    BAIKAL_DOMAIN = "global-int-next"
    BAIKAL_API_HOST = "https://api.{BAIKAL_DOMAIN}.baikalplatform.com"
    BAIKAL_AUTH_HOST = "https://auth.{BAIKAL_DOMAIN}.baikalplatform.com"

    LOGIN_PATH = "/token"
    ABOUT_ALGORITHM_PATH = "/algorithms/v1/unstable/algorithms/{algorithm_id}/"
    RUN_ALGORITHM_PATH = "/algorithms/v1/algorithms/{algorithm_id}/run"
    SCHEDULE_ALGORITHM_PATH = "/algorithms/v1/algorithms/{algorithm_id}/schedules"
    STATUS_ALGORITHM_EXECUTION_PATH = "/algorithms/v1/executions/{algorithm_id}"
    LOGS_ALGORITHM_EXECUTION_PATH = "/algorithms/v1/executions/{algorithm_id}/logs"
    METRICS_ALGORITHM_EXECUTION_PATH = "/algorithms/v1/executions/{algorithm_id}/metrics"
    STOP_ALGORITHM_EXECUTION_PATH = "/algorithms/v1/executions/{algorithm_id}"

    def __init__(self, client_id, password, autologin=True):
        self.client_id = client_id
        self.password = password
        self.autologin = autologin

        self._current_token = None
        self._current_token_expires = None

    def _save_login_token(self):
        data = self.login()

        self._current_token = data["access_token"]
        self._current_token_expires = datetime.now() + timedelta(seconds=data["expires_in"])

    def _login_if_needed(self):
        if self._current_token is not None and self._current_token_expires > datetime.now():
            return  # we have a token and the token has not expired so we don't need to login again

        logger.debug("Logging in user to retrieve access token")

        data = self.login()

        self._current_token = data["access_token"]
        self._current_token_expires = datetime.now() + timedelta(seconds=data["expires_in"])

    def _api_request(self, path, method, **kwargs):
        if self.autologin:
            self._login_if_needed()

        baikal_api_host = self.BAIKAL_API_HOST.format(BAIKAL_DOMAIN=self.BAIKAL_DOMAIN)

        auth_headers = {
            "Authorization": "Bearer " + self._current_token
        }

        res = lazy.requests.request(method, baikal_api_host + path, headers=auth_headers, **kwargs)

        res.raise_for_status()

        return res

    def login(self, grant_type="client_credentials", scope=None, purpose=None):
        data = dict(grant_type=grant_type)

        if scope is not None:
            data["scope"] = scope
        if purpose is not None:
            data["purpose"] = purpose

        baikal_auth_host = self.BAIKAL_AUTH_HOST.format(BAIKAL_DOMAIN=self.BAIKAL_DOMAIN)

        url = baikal_auth_host + self.LOGIN_PATH

        res = lazy.requests.post(url, auth=(self.client_id, self.password), data=data)
        res.raise_for_status()

        return res.json()

    def about_algorithm(self, algorithm_id):
        path = self.ABOUT_ALGORITHM_PATH.format(algorithm_id=algorithm_id)

        res = self._api_request(path, "GET")

        return res.json()

    def run_algorithm(self, algorithm_id, args=None, environment=None):
        if args is None:
            args = []
        if environment is None:
            environment = {}

        path = self.RUN_ALGORITHM_PATH.format(algorithm_id=algorithm_id)

        res = self._api_request(path, "POST", json=dict(args=args, environment=environment))

        return res.json()

    def schedule_algorithm(self, algorithm_id, cron, args=None, environment=None):
        if args is None:
            args = []
        if environment is None:
            environment = {}

        path = self.SCHEDULE_ALGORITHM_PATH.format(algorithm_id=algorithm_id)

        res = self._api_request(path, "POST", data=dict(args=args, environment=environment, cron=cron))

        return res.json()

    def status_algorithm_execution(self, algorithm_id):
        path = self.STATUS_ALGORITHM_EXECUTION_PATH.format(algorithm_id=algorithm_id)
        res = self._api_request(path, "GET")
        return res.json()

    def logs_algorithm_execution(self, algorithm_id):
        path = self.LOGS_ALGORITHM_EXECUTION_PATH.format(algorithm_id=algorithm_id)
        res = self._api_request(path, "GET")
        return res.json()

    def metrics_algorithm_execution(self, algorithm_id):
        path = self.METRICS_ALGORITHM_EXECUTION_PATH.format(algorithm_id=algorithm_id)
        res = self._api_request(path, "GET")
        return res.json()

    def stop_algorithm_execution(self, algorithm_id):
        path = self.STOP_ALGORITHM_EXECUTION_PATH.format(algorithm_id=algorithm_id)
        res = self._api_request(path, "DELETE")
        return res.json()


def _difference_streamer():
    """
    Wrapper to compute differences between streamed lists.
    Original data is asummed to be in the same order as previous data.

    Examples
    ---------
    >>> diff_streamer = _difference_streamer()
    >>> new_data = diff_streamer([0, 1, 2])  # new_data=[0, 1, 2]
    >>> new_data = diff_streamer([0, 1, 2, 3, 4])  # new_data=[3, 4]
    >>> new_data = diff_streamer([0, 1, 2, 3, 4])  # new_data=[]
    """
    last_length = 0

    def difference_stream(data):
        nonlocal last_length

        if len(data) > last_length:
            new_data = data[last_length:]
            last_length = len(data)
        else:
            new_data = []

        return new_data

    return difference_stream


class FourthPlatform(ExecutionEnvironment):
    """
    Execution environment to run experiments in 4th platform.

    Parameters
    ------------
    config_files: list of str, optional
        List of JSON paths to look for configuration values.
    """

    GYMNOS_ALGORITHM_ID = "gymnos"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--monitor", help="Whether or not monitor training from execution environment",
                            action="store_true", default=False)
        parser.add_argument("--monitor_refresh", help="Number of seconds to retrieve new logs and status",
                            default=30, type=int)

    class Config(config.Config):
        """
        You need a 4th Platform account to train with this service.

        Attributes
        ------------
        FOURTH_PLATFORM_CLIENT_ID: str
            Fourth Platform Client ID
        FOURTH_PLATFORM_PASSWORD: str
            Fourth Platform Client Password
        """  # noqa: E501
        FOURTH_PLATFORM_PASSWORD = config.Value(required=True, help="Fourth Platform Client ID")
        FOURTH_PLATFORM_CLIENT_ID = config.Value(required=True, help="Fourth Platform Client Password")

    def __init__(self, config_files=None):
        super().__init__(config_files)

        self._api = FourthPlatformAPI(self.config.FOURTH_PLATFORM_CLIENT_ID,
                                      self.config.FOURTH_PLATFORM_PASSWORD)

    def train(self, trainer, **kwargs):
        """
        Train experiment

        Parameters
        -----------
        trainer: gymnos.trainer.Trainer
            Trainer instance

        Returns
        ---------
        dict
            Dictionnary with "execution_id"
        """
        experiment_str_json = json.dumps(trainer.to_dict(), cls=NumpyEncoder,    # we only need the specifications
                                         default=default)

        experiment_b64_bytes = base64.b64encode(experiment_str_json.encode("utf8"))
        experiment_b64_str = str(experiment_b64_bytes, "utf8")

        res = self._api.run_algorithm(self.GYMNOS_ALGORITHM_ID, args=[experiment_b64_str])

        logger.info("Experiment executed successfully. Execution ID: {}".format(res["execution_id"]))

        if kwargs["monitor"]:
            self._start_monitoring(res["execution_id"], kwargs["monitor_refresh"])

    def _start_monitoring(self, execution_id, refresh_seconds=30):
        """
        Monitor execution fetching logs and status.
        It will finish monitoring when the status is either completed or failed.

        Parameters
        -----------
        execution_id: str
            Execution ID to monitor.
        refresh_seconds: int
            Refresh seconds
        """
        diff_streamer = _difference_streamer()

        logger.info("New logs and status will be fetched every {}s".format(refresh_seconds))

        while True:
            logs = self._api.logs_algorithm_execution(execution_id)

            status = self._api.status_algorithm_execution(execution_id)

            new_logs = diff_streamer(logs)

            lvl_to_color = {
                'WARN': "yellow",
                'INFO': "white",
                'DEBUG': "blue",
                'FATAL': "yellow",
                'ERROR': "red"
            }

            for new_log in new_logs:
                date = parse(new_log["time"])
                new_log["time"] = color(date.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3], fg="grey")

                if new_log["lvl"] in lvl_to_color:
                    new_log["msg"] = color(new_log["msg"], fg=lvl_to_color[new_log["lvl"]])
                    new_log["lvl"] = color(new_log["lvl"], fg=lvl_to_color[new_log["lvl"]])

                message = "{} - {} - {}".format(new_log["time"], new_log["lvl"], new_log["msg"])

                print(message)

            if status["status"] == "failed":
                logger.error("Training failed. Check logs to get more info.")
                break

            if status["status"] == "completed":
                logger.info("Training successfully completed")
                break

            time.sleep(refresh_seconds)
