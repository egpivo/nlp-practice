import os

import pytest
import torch

from nlp_practice.case.translation.utils import (
    causal_mask,
    filter_pairs,
    is_valid_pair,
    normalize_string,
    read_file,
    remove_accents,
)

ENGLISH_PREFIXES = ("I", "You", "He", "She", "It", "We", "They")
MAX_LENGTH = 10


@pytest.mark.parametrize(
    "input_str, expected_output",
    [
        ("A\u2019A", "A’A"),
        # Add more test cases as needed
    ],
)
def test_remove_accents(input_str, expected_output):
    result = remove_accents(input_str)
    assert result == expected_output


@pytest.mark.parametrize(
    "input_str, expected_output",
    [
        ("I\u2019m    happy.Really,,, ~..!", "i m happy really !"),
        # Add more test cases as needed
    ],
)
def test_normalize_string(input_str, expected_output):
    result = normalize_string(input_str)
    assert result == expected_output


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


@pytest.mark.parametrize(
    "pair, prefixes, expected",
    [
        (["Hello", "Bonjour"], ENGLISH_PREFIXES, False),
        (
            ["This is a long sentence", "C'est une longue phrase"],
            ENGLISH_PREFIXES,
            False,
        ),
        (["Invalid Pair", "Not French"], ENGLISH_PREFIXES, False),
    ],
)
def test_is_valid_pair(pair, prefixes, expected):
    result = is_valid_pair(pair, prefixes)
    assert result == expected


@pytest.mark.parametrize(
    "pairs, prefixes, expected",
    [
        (
            [["Hello", "Bonjour"], ["Short", "Court"], ["Too long", "Trop long"]],
            ENGLISH_PREFIXES,
            [],
        ),
        (
            [["Invalid Pair", "Not French"], ["Another invalid", "Encore invalide"]],
            ENGLISH_PREFIXES,
            [],
        ),
    ],
)
def test_filter_pairs(pairs, prefixes, expected):
    result = filter_pairs(pairs, prefixes)
    assert result == expected


def test_causal_mask():
    # Test with size 1
    size_1 = 1
    expected_mask_1 = torch.tensor([[[True]]])
    assert torch.equal(causal_mask(size_1), expected_mask_1)

    # Test with size 2
    size_2 = 2
    expected_mask_2 = torch.tensor([[[True, False], [True, True]]])
    assert torch.equal(causal_mask(size_2), expected_mask_2)

    # Test with size 3
    size_3 = 3
    expected_mask_3 = torch.tensor(
        [[[True, False, False], [True, True, False], [True, True, True]]]
    )
    assert torch.equal(causal_mask(size_3), expected_mask_3)
