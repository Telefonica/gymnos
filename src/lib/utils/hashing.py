#
#
#   Hashing
#
#

import hashlib


def sha1_text(text):
    sha1 = hashlib.sha1(text.encode("utf-8"))
    return sha1.hexdigest()
