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

        pos_embedding = torch.zeros(max_length, embedding_size)
        position = torch.arange(0, max_length).unsqueeze(1)

        wavelength = torch.exp(
            -(torch.arange(0, embedding_size, 2) / embedding_size)
            * math.log(wave_factor)
        )

        pos_embedding[:, 0::2] = torch.sin(position * wavelength)
        pos_embedding[:, 1::2] = torch.cos(position * wavelength)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return token_embedding + self.pos_embedding[:, : token_embedding.size(0)]


class TokenEmbedding(nn.Module):
    """
    References
    ----------
    - https://pytorch.org/tutorials/beginner/translation_transformer.html
    """

    def __init__(self, input_size: int, embedding_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.embedding_size = embedding_size
        self.weight = math.sqrt(embedding_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens.long()) * self.weight
