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
    """
    Chain functions. Output will be the input for the next function.
    Parameters
    ------------
    funcs: variadic callables
        Functions to chain. Order will be respected. Note that non callable objects will be ignored
    Returns
    --------
    callable
        Callable expecting arguments for the first function
    """
    assert len(funcs) > 0

    funcs = list(
        filter(callable, funcs))

    def inner_chain(*args, **kwargs):
        result = funcs[0](
            *args, **kwargs)
        for func in funcs[1:]:
            if callable(func):
                result = func(result)
        return result

    return inner_chain


_missing = object()

# Adapted from https://github.com/pallets/werkzeug


class cached_property():
    """
    A decorator that converts a function into a lazy property.  The
    function wrapped is called the first time to retrieve the result
    and then that calculated result is used the next time you access
    the value::

        class Foo(object):

            @cached_property
            def foo(self):
                # calculate something important here
                return 42

    The class has to have a `__dict__` in order for this property to
    work.
    """

    # implementation detail: this property is implemented as non-data
    # descriptor.  non-data descriptors are only invoked if there is
    # no entry with the same name in the instance's __dict__.
    # this allows us to completely get rid of the access function call
    # overhead.  If one choses to invoke __get__ by hand the property
    # will still work as expected because the lookup logic is replicated
    # in __get__ for manual invocation.

    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = obj.__dict__.get(
            self.__name__, _missing)
        if value is _missing:
            value = self.func(obj)
            obj.__dict__[
                self.__name__] = value
        return value


def drop(dic, key):
    return {k: v for k, v in dic.items() if k != key}
