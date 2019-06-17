#
#
#   History Tracker
#
#

from .tracker import Tracker

from datetime import datetime
from collections import defaultdict


class History(Tracker):

    def start(self, run_name=None, logdir=None):
        self.run_name = run_name
        self.logdir = logdir

        self.start_datetime = datetime.now()

        self.tags = {}
        self.params = {}
        self.assets = {}
        self.images = {}
        self.figures = {}
        self.metrics = defaultdict(list)

    def log_tag(self, key, value):
        self.tags[key] = value

    def log_asset(self, name, file_path):
        self.assets[name] = file_path

    def log_image(self, name, file_path):
        self.images[name] = file_path

    def log_figure(self, name, figure):
        self.figures[name] = figure

    def log_metric(self, name, value, step=None):
        self.metrics[name].append(value)

    def log_param(self, name, value, step=None):
        self.params[name] = value

    def end(self):
        self.end_datetime = datetime.now()
