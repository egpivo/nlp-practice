import pytest
import torch

from nlp_practice.model.layers.encoder import EncoderRNN


@pytest.fixture
def encoder_rnn():
    return EncoderRNN(input_size=10, hidden_size=64, dropout_rate=0.1)


def test_encoder_rnn_forward(encoder_rnn):
    batch_size = 5
    seq_length = 8

    input_tensor = torch.zeros(batch_size, seq_length, dtype=torch.long)
    output, hidden = encoder_rnn(input_tensor)

    expected_output_shape = (batch_size, seq_length, 64)
    assert output.shape == expected_output_shape

    expected_hidden_shape = (1, batch_size, 64)
    assert hidden.shape == expected_hidden_shape
