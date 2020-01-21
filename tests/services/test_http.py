#
#
#   Downloader test
#
#

import os
import json

from gymnos.services.http import download_file_from_url


def test_download_file_from_url(requests_mock, tmp_path):
    url = "https://fake-api.com/users"
    requests_mock.get(url, json={"data": [0, 1, 2]})
    filename = "users.json"
    file_path = str(tmp_path / filename)
    response = download_file_from_url(url, file_path)

    assert os.path.isfile(file_path)

    with open(file_path) as fp:
        users = json.load(fp)

    assert "data" in users

    response = download_file_from_url(url, file_path, force=False)

    assert response is None

    assert os.path.isfile(file_path)

    response = download_file_from_url(url, file_path, force=True)

    assert response is not None

    assert os.path.isfile(file_path)
