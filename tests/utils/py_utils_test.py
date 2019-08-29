#
#
#   Python utils
#
#

from gymnos.utils.py_utils import classproperty


class Example:

    a = "hello"

    @classproperty
    def property_1(cls):
        return 1 + 2

    @classproperty
    def property_2(cls):
        return cls.a


def test_classproperty():
    assert Example.property_1 == 3

    assert Example.property_2 == "hello"
