#
#
#   Hashing test
#
#

from gymnos.utils.hashing import sha1_text


def test_hashing():
    text = "hello world"
    expected_hashed_text = "2aae6c35c94fcfb415dbe95f408b9ce91ee846ed"

    assert sha1_text(text) == expected_hashed_text

    text = "GyMnOs"
    expected_hashed_text = "db7f56efbf91ec97b4005a82736371848fcb0a60"

    assert sha1_text(text) == expected_hashed_text
