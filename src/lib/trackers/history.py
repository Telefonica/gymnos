#
#
#   History Tracker
#
#

from .tracker import Tracker
from collections import defaultdict


class History(Tracker):

    def start(self, run_name=None, logdir=None):
        self.metrics = defaultdict(list)
        self.params = {}


    def log_metric(self, name, value, step=None):
        self.metrics[name].append(value)


    def log_param(self, name, value, step=None):
        self.params[name] = value
