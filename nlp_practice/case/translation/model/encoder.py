import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor]:
        embedding = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedding)
        return output, hidden
