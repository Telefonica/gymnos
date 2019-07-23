#
#
#   Text utils
#
#

from gymnos.utils.text_utils import ensure_unicode, humanize_url, filenamify_url, humanize_bytes


def test_ensure_unicode():
    text = "Hello World!"
    encoded_text = str.encode(text)

    assert text == ensure_unicode(encoded_text)

    assert text == ensure_unicode(text)


def test_humanize_url():
    assert humanize_url("http://www.google.com") == "google.com"

    assert humanize_url("google.com") == "google.com"

    assert humanize_url("www.google.com") == "google.com"

    assert humanize_url("www2.google.com") == "google.com"

    assert humanize_url("www3.google.com") == "google.com"


def test_filenamify_url():
    assert filenamify_url("www.google.com/maps", "_") == "google.com_maps"

    assert filenamify_url("www2.google.com/maps.madrid") == "google.com_maps.madrid"

    assert filenamify_url("www2.google.com/maps madrid") == "google.com_maps.madrid"


def test_humanize_bytes():
    assert humanize_bytes(100) == "100.0B"

    assert humanize_bytes(4096) == "4.0KiB"

    assert humanize_bytes(4194304) == "4.0MiB"

    assert humanize_bytes(4294967296) == "4.0GiB"

    assert humanize_bytes(4398046511104) == "4.0TiB"
