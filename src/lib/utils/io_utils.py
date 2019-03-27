#
#
#   I/O Utils
#
#

import json


def read_from_text(file_path):
    with open(file_path) as f:
        return f.read()


def read_from_json(file_path):
    with open(file_path) as f:
        return json.load(f)
