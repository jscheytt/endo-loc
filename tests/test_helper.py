from .context import sample

import tests.conftest as cft

import helper.helper as hlp


def test_strip_file_extension():
    filename = "abcdef.jpg"
    assert "abcdef" == hlp.strip_file_extension(filename)


def test_file_length():
    assert hlp.file_length(cft.example_img)
