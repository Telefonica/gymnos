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


def save_to_json(path, obj, indent=4):
    with open(path, "w") as outfile:
        json.dump(obj, outfile, indent=indent)
