import pytest
import torch

from nlp_practice.model.transformer import Seq2SeqTransformer


@pytest.fixture
def seq2seq_transformer():
    embedding_size = 512
    num_heads = 8
    dim_feedforward = 2048
    num_encoder_layers = 6
    num_decoder_layers = 6
    input_size = 7
    output_size = 10
    max_length = 100

    return Seq2SeqTransformer(
        embedding_size,
        num_heads,
        dim_feedforward,
        num_encoder_layers,
        num_decoder_layers,
        input_size,
        output_size,
        max_length=max_length,
    )


def test_forward(seq2seq_transformer):
    batch_size = 5
    input_size = seq2seq_transformer.input_embedding.embedding.num_embeddings
    output_size = seq2seq_transformer.output_embedding.embedding.num_embeddings

    input_tensor = torch.randint(0, 1, (batch_size, input_size))
    output_tensor = torch.randint(0, 1, (batch_size, output_size))

    input_mask = torch.ones((input_size, input_size))
    output_mask = torch.ones((output_size, output_size))

    logits = seq2seq_transformer(input_tensor, output_tensor, input_mask, output_mask)

    # Check if the output has the correct shape
    assert logits.shape == (batch_size, output_size, output_size)


def test_masking(seq2seq_transformer):
    batch_size = 5
    input_size = seq2seq_transformer.input_embedding.embedding.num_embeddings
    output_size = seq2seq_transformer.output_embedding.embedding.num_embeddings

    input_tensor = torch.randint(0, input_size, (batch_size, input_size))
    output_tensor = torch.randint(0, output_size, (batch_size, output_size))

    assert input_tensor.size(0) == output_tensor.size(0)

    logits = seq2seq_transformer(
        input=input_tensor,
        output=output_tensor,
        input_mask=torch.ones(input_size, input_size),
        output_mask=torch.ones(output_size, output_size),
        memory_mask=torch.ones(output_size, input_size),
        input_padding_mask=torch.ones(batch_size, input_size),
        output_padding_mask=torch.ones(batch_size, output_size),
        memory_key_padding_mask=torch.ones(batch_size, input_size),
    )

    assert logits.shape == (batch_size, output_size, output_size)
