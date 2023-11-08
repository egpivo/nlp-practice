import torch
import torch.nn as nn
import torch.nn.functional as F

from examples.translation.seq2seq.src import MAX_LENGTH, SOS_TOKEN


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


class DecoderRNN(nn.Module):
    def __init__(
        self, hidden_size: int, output_size: int, dropout_rate: float, device: str
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.device = device

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_hidden: torch.Tensor,
        target_tensor: torch.Tensor = None,
    ):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=self.device
        ).fill_(SOS_TOKEN)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden
            )
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, top_i = decoder_output.topk(1)
                decoder_input = top_i.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_probs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_probs, decoder_hidden, None

    def forward_step(
        self, input: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor]:
        embedding = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(F.relu(embedding), hidden)
        logits = self.output_layer(output)
        return logits, hidden


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


class AttentionDecoderRNN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        dropout_rate: float = 0.1,
        device: str = "cpu",
        max_length: int = MAX_LENGTH,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attention = BahadanauAttention(self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.gru = nn.GRU(2 * self.hidden_size, self.hidden_size, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        self.device = device

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_hidden: torch.Tensor,
        target_tensor: torch.Tensor = None,
    ) -> tuple[torch.Tensor]:
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=self.device
        ).fill_(SOS_TOKEN)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden, attention_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attention_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, top_i = decoder_output.topk(1)
                decoder_input = top_i.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_probs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)
        return decoder_probs, decoder_hidden, attentions

    def forward_step(
        self, input: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> tuple[torch.Tensor]:
        embedding = self.dropout(self.embedding(input))
        query = hidden.permute(1, 0, 2)
        context, attention_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedding, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        logits = self.output_layer(output)
        return logits, hidden, attention_weights
