import pytest
import torch

from nlp_practice.case.translation import EOS_TOKEN
from nlp_practice.case.translation.data.data_handler import (
    LanguageData,
    index_from_sentence,
    index_tensor_from_sentence,
)

# Set a seed for reproducibility
torch.manual_seed(42)


@pytest.fixture
def language_data():
    return LanguageData(name="eng")


def test_language_data_add_sentence(language_data):
    sentence = "This is a test sentence."
    language_data.add_sentence(sentence)

    assert language_data.word_to_index == {
        "This": 2,
        "is": 3,
        "a": 4,
        "test": 5,
        "sentence.": 6,
    }
    assert language_data.index_to_word == {
        0: "SOS",
        1: "EOS",
        2: "This",
        3: "is",
        4: "a",
        5: "test",
        6: "sentence.",
    }
    assert language_data.num_words == 7


def test_language_data_add_word(language_data):
    word = "example"
    language_data.add_word(word)

    assert language_data.word_to_index == {"example": 2}
    assert language_data.index_to_word == {0: "SOS", 1: "EOS", 2: "example"}
    assert language_data.num_words == 3


@pytest.fixture
def test_language_data():
    language_data = LanguageData(name="test_language")
    language_data.add_word("This")
    language_data.add_word("is")
    language_data.add_word("a")
    language_data.add_word("test")
    language_data.add_word("sentence.")
    return language_data


def test_index_from_sentence(test_language_data):
    sentence = "This is a test sentence."
    result = index_from_sentence(test_language_data, sentence)

    # Check if the result is a list of integers
    assert isinstance(result, list)
    assert all(isinstance(index, int) for index in result)
    expected_result = [2, 3, 4, 5, 6]
    assert result == expected_result


def test_index_tensor_from_sentence(test_language_data):
    sentence = "This is a test sentence."
    result = index_tensor_from_sentence(test_language_data, sentence)

    assert isinstance(result, torch.Tensor)
    expected_result = torch.tensor([2, 3, 4, 5, 6, EOS_TOKEN], dtype=torch.long).view(
        1, -1
    )
    assert torch.equal(result, expected_result)
