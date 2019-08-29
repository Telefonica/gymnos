#
#
#   IO Utils
#
#

from gymnos.utils.io_utils import read_file_text


def test_read_file_text(tmp_path):
    text_to_write = "Hello world"
    file_path = tmp_path / "test.txt"
    file_path.write_text(text_to_write, encoding="utf8")

    read_text = read_file_text(str(file_path))

    assert read_text == text_to_write
