from typing import Optional

import torch
import torch.nn as nn
from torch.nn import Transformer

from nlp_practice.case.translation import MAX_LENGTH
from nlp_practice.model.layers.attention import MultiHeadAttention
from nlp_practice.model.layers.decoder import (
    DecoderBlockTransformer,
    DecoderTransformer,
)
from nlp_practice.model.layers.embedder import PositionalEncoder, TokenEmbedder
from nlp_practice.model.layers.encoder import EncoderBlock, EncoderTransformer
from nlp_practice.model.layers.feed_forward import FeedforwardBlock
from nlp_practice.model.layers.projection import ProjectionLayer


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


class CustomTransformer(nn.Module):
    def __init__(
        self,
        encoder: EncoderTransformer,
        decoder: DecoderTransformer,
        source_embedding: TokenEmbedder,
        target_embedding: TokenEmbedder,
        source_position: PositionalEncoder,
        target_position: PositionalEncoder,
        projection_layer: ProjectionLayer,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.source_position = source_position
        self.target_position = target_position
        self.projection_layer = projection_layer

    def encoder(
        self, source: torch.Tensor, source_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        source = self.source_embedding(source)
        source = self.source_position(source)
        return self.encoder(source, source_mask)

    def decoder(
        self,
        encoder_output: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = self.target_embedding(target)
        target = self.target_position(target)
        return self.decoder(target, encoder_output, source_mask, target_mask)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_layer(x)


class CustomTransformerBuilder:
    def __init__(
        self,
        source_vocabulary_size: int,
        target_vocabulary_size: int,
        source_seq_len: int,
        target_seq_len: int,
        embeddings_size: int = 512,
        num_layers: int = 6,
        heads: int = 8,
        dropout: float = 0.1,
        ff_size: int = 2048,
    ) -> None:
        self.source_vocabulary_size = source_vocabulary_size
        self.target_vocabulary_size = target_vocabulary_size
        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len
        self.embeddings_size = embeddings_size
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.ff_size = ff_size

    def _build_encoder(self) -> EncoderTransformer:
        encoder_blocks = []
        for _ in range(self.num_layers):
            encoder_self_attention = MultiHeadAttention(
                self.embeddings_size, self.heads, self.dropout
            )
            feed_forward_block = FeedforwardBlock(
                self.embeddings_size, self.ff_size, self.dropout
            )
            encoder_block = EncoderBlock(
                encoder_self_attention, feed_forward_block, self.dropout
            )
            encoder_blocks.append(encoder_block)
        return EncoderTransformer(encoder_blocks)

    def _build_decoder(self) -> DecoderTransformer:
        decoder_blocks = []
        for _ in range(self.num_layers):
            decoder_self_attention = MultiHeadAttention(
                self.embeddings_size, self.heads, self.dropout
            )
            decoder_cross_attention = MultiHeadAttention(
                self.embeddings_size, self.heads, self.dropout
            )
            feed_forward_block = FeedforwardBlock(
                self.embeddings_size, self.ff_size, self.dropout
            )
            decoder_block = DecoderBlockTransformer(
                decoder_self_attention,
                decoder_cross_attention,
                feed_forward_block,
                self.dropout,
            )
            decoder_blocks.append(decoder_block)
        return DecoderTransformer(decoder_blocks)

    def build(self) -> nn.Module:
        transformer = CustomTransformer(
            encoder=self._build_encoder(),
            decoder=self._build_decoder(),
            source_embedding=TokenEmbedder(
                self.embeddings_size, self.source_vocabulary_size
            ),
            target_embedding=TokenEmbedder(
                self.embeddings_size, self.target_vocabulary_size
            ),
            source_position=PositionalEncoder(
                self.embeddings_size, self.source_seq_len, dropout=self.dropout
            ),
            target_position=PositionalEncoder(
                self.embeddings_size, self.target_seq_len, dropout=self.dropout
            ),
            projection_layer=ProjectionLayer(
                self.embeddings_size, self.target_vocabulary_size
            ),
        )

        for parameter in transformer.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        return transformer
