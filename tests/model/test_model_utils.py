import pytest
import torch

from nlp_practice.model.utils import PositionalEncoder, TokenEmbedder


@pytest.fixture
def positional_encoder():
    embedding_size = 512
    max_length = 100
    wave_factor = 10000
    return PositionalEncoder(embedding_size, max_length, wave_factor)


def test_pos_embedding_shape(positional_encoder):
    assert positional_encoder.pos_embedding.shape == torch.Size(
        [1, positional_encoder.max_length, positional_encoder.embedding_size]
    )


def test_forward(positional_encoder):
    token_embedding = torch.rand((10, positional_encoder.embedding_size))
    output = positional_encoder(token_embedding).squeeze(0)

    assert output.shape == token_embedding.shape
    assert not torch.allclose(output, token_embedding)


def test_different_max_length():
    embedding_size = 512
    max_length = 50
    wave_factor = 10000
    positional_encoder = PositionalEncoder(embedding_size, max_length, wave_factor)

    token_embedding = torch.rand((5, embedding_size))
    output = positional_encoder(token_embedding).squeeze(0)

    assert output.shape == token_embedding.shape
    assert not torch.allclose(output, token_embedding)


def test_different_embedding_size():
    embedding_size = 256
    max_length = 100
    wave_factor = 10000
    positional_encoder = PositionalEncoder(embedding_size, max_length, wave_factor)

    token_embedding = torch.rand((10, embedding_size))
    output = positional_encoder(token_embedding).squeeze(0)

    assert output.shape == token_embedding.shape
    assert not torch.allclose(output, token_embedding)


def test_pos_embedding_requires_grad():
    embedding_size = 512
    max_length = 100
    wave_factor = 10000
    positional_encoder = PositionalEncoder(embedding_size, max_length, wave_factor)

    # Check that the positional encoding tensor is not trainable
    assert not positional_encoder.pos_embedding.requires_grad


def test_pos_embedding_values(positional_encoder):
    # Check if the values in the positional encoding tensor match the expected formula
    pos_embedding = positional_encoder.pos_embedding.squeeze(0)
    position = torch.arange(0, positional_encoder.max_length).unsqueeze(1)
    wavelength = torch.exp(
        -(
            torch.arange(0, positional_encoder.embedding_size, 2)
            / positional_encoder.embedding_size
        )
        * torch.log(torch.tensor(positional_encoder.wave_factor))
    )

    expected_values = torch.zeros_like(pos_embedding)
    expected_values[:, 0::2] = torch.sin(position * wavelength)
    expected_values[:, 1::2] = torch.cos(position * wavelength)

    assert torch.allclose(pos_embedding, expected_values)


@pytest.fixture
def token_embedding():
    input_size = 100
    embedding_size = 256
    return TokenEmbedder(input_size, embedding_size)


def test_forward(token_embedding):
    tokens = torch.tensor([1, 2, 3, 4, 5])

    output = token_embedding(tokens)

    assert output.shape == torch.Size([len(tokens), token_embedding.embedding_size])
    assert torch.allclose(
        output, token_embedding.embedding(tokens.long()) * token_embedding.weight
    )
