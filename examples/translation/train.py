import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from examples.translation.seq2seq import AttentionDecoderRNN, EncoderRNN


def train_per_epoch(
    dataloader: DataLoader,
    encoder: EncoderRNN,
    decoder: AttentionDecoderRNN,
    encoder_optimizer: optim.Optimizer,
    decoder_optimizer: optim.Optimizer,
    criterion: nn.modules.loss,
):
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)
