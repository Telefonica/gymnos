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

    Parameters
    ----------
    s: str
        String to decode.

    Returns
    -------
    str
        Unicode string.
    """
    if isinstance(s, bytes):
        s = s.decode('utf-8', 'replace')
    return s


def humanize_url(url):
    """
    Humanize URL by removing http, www prefixes, etc ...

    Parameters
    ----------
    url: str
        URL to humanize
    Returns
    -------
    str
        Humanized url
    """
    re_start = re.compile(r"^https?://")
    re_end = re.compile(r"/$")
    url = re_end.sub("", re_start.sub("", url))
    url = url.replace("www.", "")
    url = url.replace("ww2.", "")
    return url


def filenamify_url(url, replace_char="_"):
    """
    Convert a string to something suitable as a file name.

    Parameters
    ----------
    url: str
        URL to filenamify
    replace_char: str
        Character to replace for non-ascii letters/digits

    Returns
    -------
    str
        Clean URL to save as filename
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


def humanize_bytes(num_bytes, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num_bytes) < 1024.0:
            return "%3.1f%s%s" % (num_bytes, unit, suffix)
        num_bytes /= 1024.0
    return "%.1f%s%s" % (num_bytes, 'Yi', suffix)
