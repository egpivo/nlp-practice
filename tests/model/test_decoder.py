import pytest
import torch

from nlp_practice.case.translation import MAX_LENGTH
from nlp_practice.model.layers.decoder import AttentionDecoderRNN, DecoderRNN


@pytest.fixture
def decoder_rnn():
    torch.manual_seed(42)
    return DecoderRNN(hidden_size=64, output_size=10, dropout_rate=0.1, device="cpu")


@pytest.fixture
def attention_decoder_rnn():
    torch.manual_seed(42)
    return AttentionDecoderRNN(
        hidden_size=64, output_size=10, dropout_rate=0.1, device="cpu"
    )


def test_decoder_rnn_forward(decoder_rnn):
    torch.manual_seed(42)
    batch_size = 5
    max_length = MAX_LENGTH
    target_tensor = torch.randint(0, 10, (batch_size, max_length))

    decoder_probs, _, _ = decoder_rnn(
        torch.rand((batch_size, 10)), torch.rand((1, batch_size, 64)), target_tensor
    )

    expected_shape = (batch_size, max_length, 10)
    assert decoder_probs.shape == expected_shape

    expected_sum_value = -1156.362548828125
    expected_mean_value = -2.312725067138672

    torch.testing.assert_close(
        torch.sum(decoder_probs).item(), expected_sum_value, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        torch.mean(decoder_probs).item(), expected_mean_value, rtol=1e-5, atol=1e-5
    )


def test_attention_decoder_rnn_forward(attention_decoder_rnn):
    torch.manual_seed(42)
    batch_size = 5
    hidden_size = 64
    output_size = 10
    max_length = MAX_LENGTH
    target_tensor = torch.randint(0, 1, (batch_size, max_length))

    decoder_probs, _, attentions = attention_decoder_rnn(
        torch.rand((batch_size, hidden_size, hidden_size)),
        torch.rand((1, batch_size, hidden_size)),
        target_tensor,
    )

    expected_shape = (batch_size, max_length, output_size)
    assert decoder_probs.shape == expected_shape

    expected_sum_value = -1169.149169921875
    expected_mean_value = -2.3382983207702637

    torch.testing.assert_close(
        torch.sum(decoder_probs).item(), expected_sum_value, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        torch.mean(decoder_probs).item(), expected_mean_value, rtol=1e-5, atol=1e-5
    )
