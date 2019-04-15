#
#
#   Session
#
#


class SessionOptions:

    def __init__(self, allow_memory_growth=False, num_cores=4):
        self.allow_memory_growth = allow_memory_growth
        self.num_cores = num_cores


class Session:

    def __init__(self, device=None, options=None):
        options = options or {}

        self.device = device
        self.options = SessionOptions(**options)
