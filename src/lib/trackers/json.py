#
#
#   JSON Tracker
#
#

import json

from .tracker import Tracker
from collections import defaultdict


class JSON(Tracker):

    def __init__(self, file_path):
        self.file_path = file_path
        self.contents = defaultdict(list)

    def add_tag(self, tag):
        self.contents["tags"].append(tag)


    def log_metric(self, name, value, step=None):
        self.contents["metrics"].append({
            name: value,
            "step": step
        })


    def log_param(self, name, value, step=None):
        self.contents["params"].append({
            name: value,
            "step": step
        })


    def end(self):
        with open(self.file_path, 'w') as outfile:
            json.dump(self.contents, outfile)
