import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, device: str) -> None:
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.device = device

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        embedding = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedding, hidden)
        return output, hidden

    def initialize_hidden(self) -> torch.Tensor:
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class DecorderRNN(nn.Module):
    def __init__(self, hidden_size: int, output_size: int) -> None:
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        embedding = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(F.relu(embedding), hidden)
        prob = self.softmax(self.output_layer(output[0]))
        return prob, hidden

    def initialize_hidden(self) -> torch.Tensor:
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
