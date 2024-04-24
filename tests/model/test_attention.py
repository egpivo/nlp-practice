import pytest
import torch

from nlp_practice.model.layers.attention import BahadanauAttention


@pytest.fixture
def attention_model():
    hidden_size = 10
    return BahadanauAttention(hidden_size)


def test_forward(attention_model):
    batch_size = 5
    sequence_length = 100
    # No need to be the same as the sequence length
    query_length = 1

    query = torch.randn((batch_size, query_length, attention_model.w.in_features))
    keys = torch.randn((batch_size, sequence_length, attention_model.u.in_features))

    # Forward pass
    context, weights = attention_model(query, keys)

    # Check if the output tensors have the correct shapes
    assert context.shape == torch.Size([batch_size, 1, attention_model.w.out_features])
    assert weights.shape == torch.Size([batch_size, 1, sequence_length])
