import pytest
import torch

from nlp_practice.model.encoder import EncoderRNN


@pytest.fixture
def encoder_rnn():
    torch.manual_seed(42)
    return EncoderRNN(input_size=10, hidden_size=64, dropout_rate=0.1)


def test_encoder_rnn_forward(encoder_rnn):
    torch.manual_seed(42)
    batch_size = 5
    seq_length = 8

    input_tensor = torch.randint(0, 10, (batch_size, seq_length))
    output, hidden = encoder_rnn(input_tensor)

    expected_output_shape = (batch_size, seq_length, 64)
    assert output.shape == expected_output_shape

    expected_hidden_shape = (1, batch_size, 64)
    assert hidden.shape == expected_hidden_shape

    expected_value = 102.86456298828125
    torch.testing.assert_close(
        torch.sum(output).item(), expected_value, atol=1e-5, rtol=1e-5
    )
