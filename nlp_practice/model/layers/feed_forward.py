import torch
import torch.nn as nn


class FeedforwardBlock(nn.Module):
    def __init__(self, embedding_size: int, ff_size: int, dropout: float) -> None:
        super().__init__()

        self.linear1 = nn.Linear(embedding_size, ff_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_size, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, embedding_size)"""
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
