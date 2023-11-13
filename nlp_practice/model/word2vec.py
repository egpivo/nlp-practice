import torch
import torch.nn as nn


class SkipGram(nn.Module):
    def __init__(self, input_size: int, embedding_size: int) -> None:
        super().__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        self.linear_layer = nn.Linear(self.embedding_size, self.input_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input)
        logits = self.linear_layer(embeddings)
        return logits
