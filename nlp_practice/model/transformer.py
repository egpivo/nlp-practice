import torch
import torch.nn as nn
from torch.nn import Transformer

from nlp_practice.case.translation import MAX_LENGTH
from nlp_practice.model.layers.embedder import PositionalEncoder, TokenEmbedder


class Seq2SeqTransformer(nn.Module):
    """
    References
    ----------
    - https://pytorch.org/tutorials/beginner/translation_transformer.html
    """

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        dim_feedforward: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        input_size: int,
        output_size: int,
        max_length: int = MAX_LENGTH,
    ) -> None:
        super().__init__()

        self.embedding_size = embedding_size
        self.transformer = Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )

        self.input_embedding = TokenEmbedder(input_size, embedding_size)
        self.output_embedding = TokenEmbedder(output_size, embedding_size)
        self.positional_encoder = PositionalEncoder(embedding_size, max_length)
        self.output_layer = nn.Linear(embedding_size, output_size)

    def forward(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        input_mask: torch.Tensor,
        output_mask: torch.Tensor,
        memory_mask: torch.Tensor = None,
        input_padding_mask: torch.Tensor = None,
        output_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        input_embeddings = self.positional_encoder(self.input_embedding(input))
        output_embeddings = self.positional_encoder(self.output_embedding(output))
        output = self.transformer(
            src=input_embeddings,
            tgt=output_embeddings,
            src_mask=input_mask,
            tgt_mask=output_mask,
            memory_mask=memory_mask,
            src_key_padding_mask=input_padding_mask,
            tgt_key_padding_mask=output_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.output_layer(output)
        return logits
