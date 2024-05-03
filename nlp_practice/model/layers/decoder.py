import torch
import torch.nn as nn
import torch.nn.functional as F

from nlp_practice.case.translation import MAX_LENGTH, SOS_TOKEN
from nlp_practice.model.layers.attention import BahadanauAttention, MultiHeadAttention
from nlp_practice.model.layers.feed_forward import FeedforwardBlock
from nlp_practice.model.layers.normalization import LayerNormalization
from nlp_practice.model.layers.skip_connection import ResidualConnection


class DecoderRNN(nn.Module):
    def __init__(
        self, hidden_size: int, output_size: int, dropout_rate: float, device: str
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.device = device

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_rate)

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
        embedding = self.embedding(input)
        output, hidden = self.gru(F.relu(embedding), hidden)
        logits = self.output_layer(output)
        return logits, hidden


class AttentionDecoderRNN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        dropout_rate: float,
        device: str,
        max_length: int = MAX_LENGTH,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.device = device

        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attention = BahadanauAttention(self.hidden_size)
        self.gru = nn.GRU(2 * self.hidden_size, self.hidden_size, batch_first=True)

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


class DecoderBlockTransformer(nn.Module):
    def __init__(
        self,
        self_attention_block: MultiHeadAttention,
        cross_attention_block: MultiHeadAttention,
        feed_forward_block: FeedforwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections = nn.Module(
            ResidualConnection(dropout) for _ in range(3)
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Source: encoder; Target: decoder"""
        x = self.residual_connetions[0](
            x, lambda x: self.self_attention_block(x, x, x, target_mask)
        )
        x = self.residual_connetions[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, source_mask
            ),
        )
        x = self.residual_connetions[2](x, self.feed_forward_block)
        return x


class DecoderTransformer(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization()

    def forward(
        self,
        x: torch.Tensor,
        encoder_ouput: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_ouput, source_mask, target_mask)

        return self.norm(x)
