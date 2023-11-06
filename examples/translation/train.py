import logging
import time

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from examples.translation.seq2seq import AttentionDecoderRNN, EncoderRNN
from examples.translation.train_dataloader import TrainDataloader
from examples.translation.utils import log_time

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()


def train_per_epoch(
    dataloader: DataLoader,
    encoder: EncoderRNN,
    decoder: AttentionDecoderRNN,
    encoder_optimizer: optim.Optimizer,
    decoder_optimizer: optim.Optimizer,
    criterion: nn.modules.loss,
) -> float:
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


def train(
    train_dataloader: DataLoader,
    encoder: EncoderRNN,
    decoder: AttentionDecoderRNN,
    num_epochs: int,
    learning_rate: float = 0.0001,
    print_log_frequency: int = 2,
):
    start_time = time.time()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()
    log_loss = 0
    for epoch in range(1, num_epochs + 1):
        loss = train_per_epoch(
            train_dataloader,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
        )
        log_loss += loss

        if epoch % print_log_frequency == 0:
            average_loss = log_loss / print_log_frequency
            progress = epoch / num_epochs
            LOGGER.info(f"[{log_time(start_time, progress)}]: {average_loss:.4f}")
            log_loss = 0


if __name__ == "__main__":
    hidden_size = 128
    batch_size = 16
    num_epochs = 10
    dropout_rate = 0.2
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    dataloader_instance = TrainDataloader(batch_size, device)
    dataloader = dataloader_instance.dataloader
    input_langauge = dataloader_instance.input_language
    output_language = dataloader_instance.output_language
    encoder = EncoderRNN(input_langauge.num_words, hidden_size, dropout_rate).to(device)
    decoder = AttentionDecoderRNN(
        hidden_size, output_language.num_words, dropout_rate, device
    ).to(device)

    train(dataloader, encoder, decoder, num_epochs)
