#
#
#   IO Utils
#
#

import json
import collections

from gymnos.utils.io_utils import read_file_text, import_from_json


def test_read_file_text(tmp_path):
    text_to_write = "Hello world"
    file_path = tmp_path / "test.txt"
    file_path.write_text(text_to_write, encoding="utf8")

    read_text = read_file_text(str(file_path))

    assert read_text == text_to_write


def test_import_from_json(tmp_path):
    data = {
        "ordereddict": "collections.OrderedDict"
    }

    json_path = tmp_path / "test.json"

    with open(str(json_path), "w") as fp:
        json.dump(data, fp)

    OrderedDict = import_from_json(str(json_path), "ordereddict")

    assert OrderedDict == collections.OrderedDict
