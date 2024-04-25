import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahadanauAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()

        self.w = nn.Linear(hidden_size, hidden_size)
        self.u = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, query: torch.tensor, keys: torch.tensor) -> tuple[torch.tensor]:
        scores = self.v(torch.tanh(self.w(query) + self.u(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights


class MultiHeadAttention(nn.Module):
    _epsilon = 1e-9

    def __init__(self, embedding_size: int, heads: int, dropout: float) -> None:
        super().__init__()

        assert embedding_size % heads == 0

        self.embedding_size = embedding_size
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        self.partial_embedding_size = embedding_size // heads

        self.w_query = nn.Linear(embedding_size, embedding_size)
        self.w_key = nn.Linear(embedding_size, embedding_size)
        self.w_value = nn.Linear(embedding_size, embedding_size)
        self.w_output = nn.Linear(embedding_size, embedding_size)

    def _reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, embedding_size) -> (batch, seq_len, heads, partial_size) -> (batch, heads, seq_len, partial_size)
        return tensor.view(
            tensor.shape[0], tensor.shape[1], self.heads, self.partial_embedding_size
        ).transpose(1, 2)

    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # (batch, heads, seq_len, seq_len)
        attention = (query @ key.transpose(-2, -1)) / math.sqrt(
            self.partial_embedding_size
        )
        if mask is not None:
            attention.masked_fill_(mask == 0, self._epsilon)

        attention = self.dropout(attention)
        return attention @ value

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # (batch, seq_len, embedding_size) -> (batch, seq_len, embedding_size)
        query = self.w_query(query)
        key = self.w_key(key)
        value = self.w_value(value)

        # Reshape in terms of heads
        query = self._reshape(query)
        key = self._reshape(key)
        value = self._reshape(value)

        output = self._attention(query, key, value, mask)

        # (batch, heads, seq_len, partial_size) -> (batch, seq_len, heads, partial_size) -> (batch, seq_len, embedding_size)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(output.shape[0], -1, self.embedding_size)
        )
        return self.w_output(output)
