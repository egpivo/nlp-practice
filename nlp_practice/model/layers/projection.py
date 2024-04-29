import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    def __init__(self, embedding_size: int, vocabulary_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_size, vocabulary_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (batch, seq_len, embedding_size) -> (batch, seq_len, vocabulary_size)"""

        return torch.log_softmax(self.projection(x), dim=-1)
