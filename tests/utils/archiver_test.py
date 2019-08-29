#
#
#   Extractor test
#
#

import os
import gzip
import shutil
from pathlib import Path

from gymnos.utils.archiver import extract_zip, extract_gz, extract_tar, zipdir


def test_zipdir(tmp_path):
    to_zip_dir = tmp_path / "compress"
    os.makedirs(str(to_zip_dir))

    file_path_1 = to_zip_dir / "text1.txt"
    file_path_2 = to_zip_dir / "text2.txt"

    file_path_1.write_text("text1")
    file_path_2.write_text("text2")

    texts_dir = to_zip_dir / "texts"
    os.makedirs(str(texts_dir))
    inside_dir_file_path = texts_dir / "inside.txt"
    inside_dir_file_path.write_text("inside.txt")

    assert os.path.isfile(str(file_path_1))
    assert os.path.isfile(str(file_path_2))
    assert os.path.isfile(str(inside_dir_file_path))

    zipped_file_path = str(tmp_path / "zipped.zip")

    zipdir(str(to_zip_dir), zipped_file_path)

    extract_dir = tmp_path / "extracted"

    shutil.unpack_archive(zipped_file_path, extract_dir=str(extract_dir))

    assert os.path.isfile(str(extract_dir / "text1.txt"))
    assert os.path.isfile(str(extract_dir / "text2.txt"))
    assert os.path.isfile(str(extract_dir / "texts" / "inside.txt"))


def _test_extract_archive(tmp_path, compression_format, extract_func):
    filename_1 = "text1.txt"
    filename_2 = "text2.txt"

    text_1 = tmp_path / filename_1
    text_2 = tmp_path / filename_2

    text_1.write_text("Hello", encoding="utf8")
    text_2.write_text("World", encoding="utf8")

    compressed_filename = tmp_path / "test"

    compressed_file_path = shutil.make_archive(str(compressed_filename), compression_format, str(tmp_path))

    assert os.path.isfile(compressed_file_path)

    extract_dir = str(tmp_path / "extracted")

    extracted_dir = extract_func(compressed_file_path, extract_dir=extract_dir, verbose=False)

    assert extract_dir == extracted_dir

    extracted_file_path_1 = tmp_path / "extracted" / filename_1
    extracted_file_path_2 = tmp_path / "extracted" / filename_2

    assert os.path.isfile(str(extracted_file_path_1))
    assert os.path.isfile(str(extracted_file_path_2))

    assert extracted_file_path_1.read_text() == "Hello"
    assert extracted_file_path_2.read_text() == "World"

    extract_func(compressed_file_path, extract_dir=str(tmp_path / "extracted"), verbose=False)

    assert os.path.isfile(str(extracted_file_path_1))
    assert os.path.isfile(str(extracted_file_path_2))

    assert extracted_file_path_1.read_text() == "Hello"
    assert extracted_file_path_2.read_text() == "World"

    extract_func(compressed_file_path, extract_dir=str(tmp_path / "extracted"), verbose=False, force=True)

    assert os.path.isfile(str(extracted_file_path_1))
    assert os.path.isfile(str(extracted_file_path_2))

    assert extracted_file_path_1.read_text() == "Hello"
    assert extracted_file_path_2.read_text() == "World"


def test_extract_tar(tmp_path):
    _test_extract_archive(tmp_path, "tar", extract_tar)


def test_extract_zip(tmp_path):
    _test_extract_archive(tmp_path, "zip", extract_zip)


def test_extract_gz(tmp_path):
    filename = "text.txt"
    file_path = tmp_path / filename
    file_path.write_text("Hello", encoding="utf8")

    compressed_filepath = tmp_path / (filename + ".gz")

    with open(str(file_path), 'rb') as f_in, gzip.open(str(compressed_filepath), 'wb') as f_out:
        f_out.writelines(f_in)

    assert os.path.isfile(str(compressed_filepath))

    extract_dir = tmp_path / "extracted"

    os.makedirs(str(extract_dir))

    extracted_file_path = extract_gz(str(compressed_filepath), extract_dir=str(tmp_path / "extracted"))

    assert os.path.isfile(extracted_file_path)

    assert Path(extracted_file_path).read_text() == "Hello"

    extracted_file_path = extract_gz(str(compressed_filepath), extract_dir=str(tmp_path / "extracted"), force=False)

    assert os.path.isfile(extracted_file_path)

    assert Path(extracted_file_path).read_text() == "Hello"

    extracted_file_path = extract_gz(str(compressed_filepath), extract_dir=str(tmp_path / "extracted"), force=True)

    assert os.path.isfile(extracted_file_path)

    assert Path(extracted_file_path).read_text() == "Hello"
