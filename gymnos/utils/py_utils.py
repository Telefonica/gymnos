#
#
#   Python utils
#
#


class classproperty:

    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner):
        return self.getter(owner)


def chain(*funcs):
    assert len(funcs) > 0

    def inner_chain(*args, **kwargs):
        result = funcs[0](*args, **kwargs)
        for func in funcs[1:]:
            result = func(result)
        return result

    return inner_chain
