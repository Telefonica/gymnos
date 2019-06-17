#
#
#   Hashing
#
#

import hashlib


def sha1_text(text):
    """
    Hash using SHA-1 text.

    Parameters
    ----------
    text: str
        Text to hash.

    Returns
    --------
    str
        Hashed text
    """
    sha1 = hashlib.sha1(text.encode("utf-8"))
    return sha1.hexdigest()
