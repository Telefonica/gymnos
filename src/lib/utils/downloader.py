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
from smb.SMBConnection import SMBConnection

logger = logging.getLogger(__name__)


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
            logger.debug("File for url {} already exists in {}. Skipping".format(url, file_path))
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


def download_file_from_smb(smb_uri, file_path, username, password, force=False, verbose=False):
    if os.path.isfile(file_path) and not force:
        if verbose:
            logger.debug("File for SMB uri {} already exists in {}. Skipping".format(smb_uri, file_path))
        return

    uri_parsed = urlparse(smb_uri)

    smb_server_port = uri_parsed.port or 445
    smb_server_ip = uri_parsed.hostname

    smb_file_path = pathlib.Path(uri_parsed.path)

    shared_drive, *smb_file_path = smb_file_path.parts[1:]
    smb_file_path = os.path.join(*smb_file_path)

    smb_server_name = socket.gethostbyaddr(smb_server_ip)
    if smb_server_name:
        smb_server_name = smb_server_name[0]
    else:
        raise URLError('SMB error: Hostname with ip {} does not reply back with its machine name'.format(smb_server_ip))

    client_name = socket.gethostname()
    if client_name:
        client_name = client_name.split('.')[0]
    else:
        client_name = 'SMB%d' % os.getpid()

    with SMBConnection(username, password, client_name, smb_server_name, use_ntlm_v2=True, is_direct_tcp=True) as conn:
        success = conn.connect(smb_server_ip, smb_server_port)
        if not success:
            raise ValueError("Authentication to {} failed. Check your credentials".format(smb_server_ip))

        with open(file_path, "wb") as fp:
            conn.retrieveFile(shared_drive, smb_file_path, fp)
