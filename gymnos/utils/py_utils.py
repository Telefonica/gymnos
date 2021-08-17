#
#
#   Python utils
#
#


def remove_prefix(text: str, prefix: str):
    """
    Remove preffix from text

    Parameters
    ----------
    text
    prefix

    Returns
    -------
    str
    """
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def remove_suffix(text: str, suffix: str):
    """
    Remove suffix from text

    Parameters
    ----------
    text
    suffix

    Returns
    -------
    str
    """
    return text[:-len(suffix)] if text.endswith(suffix) and len(suffix) != 0 else text


def rreplace(s, old, new):
    """
    Replace 1 ocurrence right to left

    Parameters
    ----------
    s
    old
    new

    Returns
    -------

    """
    try:
        place = s.rindex(old)
        return ''.join((s[:place], new, s[place + len(old):]))
    except ValueError:
        return s
