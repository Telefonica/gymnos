#
#
#   Python utils
#
#


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def remove_suffix(text, suffix):
    return text[:-len(suffix)] if text.endswith(suffix) and len(suffix) != 0 else text
