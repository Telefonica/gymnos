#
#
#   Python utils
#
#

def strip_underscores(obj):
    return {k[1:]: v for (k, v) in obj.__dict__.items()}
