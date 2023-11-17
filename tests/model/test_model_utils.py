import pytest
import torch

from nlp_practice.model.utils import PositionalEncoder, TokenEmbedding


@pytest.fixture
def positional_encoder():
    embedding_size = 512
    max_length = 100
    wave_factor = 10000
    return PositionalEncoder(embedding_size, max_length, wave_factor)


def test_forward(positional_encoder):
    token_embedding = torch.rand((10, positional_encoder.pos_embedding.size(2)))
    output = positional_encoder(token_embedding)

    assert output.squeeze(0).shape == token_embedding.shape
    assert not torch.allclose(output, token_embedding)


@pytest.fixture
def token_embedding():
    input_size = 100
    embedding_size = 256
    return TokenEmbedding(input_size, embedding_size)


def test_forward(token_embedding):
    tokens = torch.tensor([1, 2, 3, 4, 5])

    output = token_embedding(tokens)

    assert output.shape == torch.Size([len(tokens), token_embedding.embedding_size])
    assert torch.allclose(
        output, token_embedding.embedding(tokens.long()) * token_embedding.weight
    )
