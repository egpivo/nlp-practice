import torch
import torch.nn as nn

from nlp_practice.model.layers.attention import MultiHeadAttention
from nlp_practice.model.layers.feed_forward import FeedforwardBlock
from nlp_practice.model.layers.normalization import LayerNormalization
from nlp_practice.model.layers.skip_connection import ResidualConnection


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


class EncoderBlock(nn.Module):
    blocks = 2

    def __init__(
        self,
        self_attention_block: MultiHeadAttention,
        feed_forward_block: FeedforwardBlock,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(self.blocks)]
        )

    def forward(self, x: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        x = self.residual_connection[0](
            x, lambda x: self.self_attention_block(x, x, x, source_mask)
        )
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x


class EncoderTransformer(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
