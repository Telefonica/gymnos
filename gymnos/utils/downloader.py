#
#
#   Download
#
#

import os
import socket
import pathlib
import requests
import logging

from tqdm import tqdm
from urllib.error import URLError
from urllib.parse import urlparse
from contextlib import contextmanager
from smb.SMBConnection import SMBConnection

logger = logging.getLogger(__name__)


def urljoin(*args):
    """
    Joins given arguments into an url. Trailing but not leading slashes are
    stripped for each argument.
    """

    return "/".join(s.strip("/") for s in args)


def download_file_from_url(url, file_path, force=False, verbose=True, headers=None, raise_for_status=True,
                           chunk_size=1024):
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
    raise_for_status: bool, optional
        Whether or not raise error if status code is not success
    chunk_size: int, optional
        Chunk size to read stream.

    Returns
    -------
    requests.Response
        Response object
    """
    if os.path.isfile(file_path) and not force:
        return

    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()

    iterator = response.iter_content(chunk_size=chunk_size)

    content_length = response.headers.get("Content-Length")

    if verbose and content_length is not None:
        file_size = int(content_length)
        num_bars = file_size // chunk_size
        iterator = tqdm(iterator, total=num_bars, unit="KB", desc=url[:50], leave=True)

    with open(file_path, "wb") as fp:
        for chunk in iterator:
            fp.write(chunk)

    return response


def parse_smb_uri(smb_uri):
    uri_parsed = urlparse(smb_uri)

    smb_server_port = uri_parsed.port or 445
    smb_server_ip = uri_parsed.hostname

    smb_file_path = pathlib.Path(uri_parsed.path)

    shared_drive, *smb_file_path = smb_file_path.parts[1:]
    smb_file_path = os.path.join(*smb_file_path)

    return dict(
        ip=smb_server_ip,
        port=smb_server_port,
        drive=shared_drive,
        path=smb_file_path
    )


@contextmanager
def smb_connection(ip, port=445, username=None, password=None):
    server_name = socket.gethostbyaddr(ip)
    if server_name:
        server_name = server_name[0]
    else:
        raise URLError('Hostname with ip {} does not reply back with its machine name'.format(ip))

    client_name = socket.gethostname()
    if client_name:
        client_name = client_name.split('.')[0]
    else:
        client_name = 'SMB%d' % os.getpid()

    with SMBConnection(username, password, client_name, server_name, use_ntlm_v2=True, is_direct_tcp=True) as conn:
        success = conn.connect(ip, port)
        if not success:
            raise ValueError("Authentication to {} failed. Check your credentials".format(ip))

        yield conn
