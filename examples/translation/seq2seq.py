import torch
import torch.nn as nn
import torch.nn.functional as F

from examples.translation import MAX_LENGTH


class EncoderRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, device: str) -> None:
        super().__init__()

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


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()

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


class AttentionDecoderRNN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        dropout_rate: float = 0.1,
        max_length: int = MAX_LENGTH,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attention = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attention_combined = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> tuple[torch.Tensor]:

        embedding = self.embedding(input).view(1, 1, -1)
        dropout_embedding = self.dropout(embedding)

        attention_weights = F.softmax(
            self.attention(torch.cat((dropout_embedding[0], hidden[0]), 1), dim=1)
        )
        attention_applied = torch.bmm(
            attention_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)
        )
        concat_attention = torch.cat((embedding[0], attention_applied[0]), 1)
        combined_attention = self.attention_combined(concat_attention).unsqueeze(0)

        output, hidden = self.gru(F.relu(combined_attention), hidden)
        log_prob = F.log_softmax(self.output_layer(output[0]), dim=1)
        return log_prob, hidden, attention_weights

    def initialize_hidden(self) -> torch.Tensor:
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
