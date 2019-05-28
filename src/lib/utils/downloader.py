#
#
#   Download
#
#

import os
import requests

from tqdm import tqdm


def download_file_from_url(url, file_path, force=False, verbose=True):
    """
    Download url to local file.

    Parameters
    ---------
    url: str
        Url to download
    file_path: str
        Path to download file
    force: bool, optional
        Whether or not force download. By default downloads are cached
    verbose: bool, optional
        Whether or not show progress bar

    Returns
    -------
    requests.Response
        Response object
    """
    if os.path.isfile(file_path) and not force:
        if verbose:
            print("File for url {} already exists in {}. Skipping".format(url, file_path))
        return

    response = requests.get(url, stream=True)
    file_size = int(response.headers["Content-Length"])
    chunk_size = 1024
    num_bars = file_size // chunk_size

    iterator = response.iter_content(chunk_size=chunk_size)
    if verbose:
        iterator = tqdm(iterator, total=num_bars, unit="KB", desc=url[:50], leave=True)

    with open(file_path, "wb") as fp:
        for chunk in iterator:
            fp.write(chunk)

    return response
