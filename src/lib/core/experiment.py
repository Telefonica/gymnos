#
#
#   Experiment
#
#

from datetime import datetime


class Experiment:

    def __init__(self, name=None, tags=None):
        self.name = name
        self.tags = tags

        self.creation_date = datetime.now()
