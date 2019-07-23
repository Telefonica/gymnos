#
#
#   Test alphanumeric
#
#

from gymnos.preprocessors import Alphanumeric


def test_transform():
    alphanumeric = Alphanumeric()
    text = "h.ell-+o_"
    new_text = alphanumeric.transform([text])[0]
    assert new_text == "h ell o "

    text = "he99llo"
    new_text = alphanumeric.transform([text])[0]
    assert new_text == text

    text = "hello"
    new_text = alphanumeric.transform([text])[0]
    assert new_text == text
