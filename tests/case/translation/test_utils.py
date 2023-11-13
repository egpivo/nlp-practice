import os

import pytest

from nlp_practice.case.translation.utils import read_file


@pytest.fixture
def test_base_path():
    return "."


def test_read_file(test_base_path, capsys):
    file_contents = ["Go.\tVa !", "Hello.\tSalut !", "How are you?\tComment ça va ?"]
    test_file_path = os.path.join(test_base_path, "eng-fra.txt")
    with open(test_file_path, "w", encoding="utf-8") as test_file:
        test_file.write("\n".join(file_contents))

    result = read_file(test_base_path)

    assert isinstance(result, list)
    assert all(isinstance(line, str) for line in result)
    assert result == file_contents

    os.remove(test_file_path)


def test_read_file_file_not_found(test_base_path, capsys):
    # Test the read_file function with a non-existent file
    with pytest.raises(FileNotFoundError) as e:
        read_file(test_base_path, language1="nonexistent", language2="file")

    # Check if the error message contains the correct information
    expected_error_message = (
        f"The file `./nonexistent-file.txt` was not found in the current directory"
    )
    assert expected_error_message in str(e.value)
