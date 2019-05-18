#
#
#   Text utils
#
#

import re
import unicodedata


def ensure_unicode(s):
    """
    Ensure string is a unicode string. If it isn't it assumed it is
    utf-8 and decodes it to a unicode string.
    """
    if isinstance(s, bytes):
        s = s.decode('utf-8', 'replace')
    return s


def humanize_url(url):
    re_start = re.compile(r"^https?://")
    re_end = re.compile(r"/$")
    url = re_end.sub("", re_start.sub("", url))
    url = url.replace("www.", "")
    url = url.replace("ww2.", "")
    return url


def filenamify_url(url, replace_char="!"):
    """
    Convert a string to something suitable as a file name. E.g.
     Matlagning del 1 av 10 - Räksmörgås | SVT Play
       ->  matlagning.del.1.av.10.-.raksmorgas.svt.play
    """
    # ensure it is unicode
    url = humanize_url(url)
    url = ensure_unicode(url)

    # NFD decomposes chars into base char and diacritical mark, which
    # means that we will get base char when we strip out non-ascii.
    url = unicodedata.normalize('NFD', url)

    # Convert to lowercase
    # Drop any non ascii letters/digits
    # Drop any leading/trailing whitespace that may have appeared
    url = re.sub(r'[^a-z0-9 .-]', replace_char, url.lower().strip())

    # Replace whitespace with dot
    url = re.sub(r'\s+', '.', url)
    url = re.sub(r'\.-\.', '-', url)

    return url
