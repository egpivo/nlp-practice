import pytest
import torch

from nlp_practice.model.word2vec import SkipGram


@pytest.fixture
def skipgram_model():
    input_size = 10
    embedding_size = 5
    return SkipGram(input_size, embedding_size)


def test_forward(skipgram_model):
    input_tensor = torch.tensor([1, 2, 3, 4, 5])
    output_tensor = skipgram_model(input_tensor)
    assert output_tensor.shape == torch.Size([5, skipgram_model.input_size])
