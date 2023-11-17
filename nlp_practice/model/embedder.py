import math

import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    """
    Notes
    -----
    - Formula:
        - PE(position, 2i) = sin(position *  exp(- (2i / embedding_size) * log(wave factor)))
        - PE(position, 2i + 1) = cos(position *  exp(- (2i / embedding_size) * log(wave factor)))

    References
    ----------
    - https://pytorch.org/tutorials/beginner/translation_transformer.html
    - https://pytorch-forecasting.readthedocs.io/en/stable/_modules/pytorch_forecasting/models/temporal_fusion_transformer/sub_modules.html#PositionalEncoder
    """

    def __init__(
        self,
        embedding_size: int,
        max_length: int,
        wave_factor: int = 10000,
    ) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.max_length = max_length
        self.wave_factor = wave_factor

        pos_embedding = self._generate_positional_embedding()
        self.register_buffer("pos_embedding", pos_embedding)

    def _generate_positional_embedding(self) -> torch.Tensor:
        position = torch.arange(0, self.max_length).reshape(self.max_length, 1)
        wavelength = torch.exp(
            -(torch.arange(0, self.embedding_size, 2) / self.embedding_size)
            * math.log(self.wave_factor)
        )
        pos_embedding = torch.zeros(self.max_length, self.embedding_size)
        pos_embedding[:, 0::2] = torch.sin(position * wavelength)
        pos_embedding[:, 1::2] = torch.cos(position * wavelength)
        return pos_embedding

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        """Note: shape of token_embedding = (batch_size, seq_length, embedding_size)"""
        with torch.no_grad():
            pos_embedding = self.pos_embedding[: token_embedding.size(1), :].unsqueeze(
                0
            )
            return token_embedding + pos_embedding


class TokenEmbedder(nn.Module):
    """
    References
    ----------
    - https://pytorch.org/tutorials/beginner/translation_transformer.html
    """

    def __init__(self, vocabulary_size: int, embedding_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.embedding_size = embedding_size
        self.weight = math.sqrt(embedding_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens.long()) * self.weight
