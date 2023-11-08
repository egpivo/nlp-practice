import logging
import time

import torch
import torch.nn as nn
from torch import optim

from examples.translation.seq2seq.dataloader import TrainDataloader
from examples.translation.seq2seq.seq2seq import AttentionDecoderRNN, EncoderRNN
from examples.translation.seq2seq.utils import log_time


class Trainer:
    def __init__(
        self,
        train_dataloader: TrainDataloader,
        encoder: EncoderRNN,
        decoder: AttentionDecoderRNN,
        num_epochs: int,
        checkpoint_path: str,
        learning_rate: float,
        logger: logging.RootLogger,
        print_log_frequency: int = 10,
    ):
        self.train_dataloader = train_dataloader
        self.encoder = encoder
        self.decoder = decoder
        self.num_epochs = num_epochs
        self.checkpoint_path = checkpoint_path
        self.learning_rate = learning_rate

        self.logger = logger
        self.print_log_frequency = print_log_frequency

    def _train_per_epoch(
        self,
        encoder_optimizer: optim.Optimizer,
        decoder_optimizer: optim.Optimizer,
        criterion: nn.modules.loss,
    ) -> float:
        total_loss = 0
        for data in self.train_dataloader:
            input_tensor, target_tensor = data
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            encoder_outputs, encoder_hidden = self.encoder(input_tensor)
            decoder_outputs, _, _ = self.decoder(
                encoder_outputs, encoder_hidden, target_tensor
            )
            decoder_outputs = decoder_outputs.view(-1, decoder_outputs.size(-1))
            loss = criterion(decoder_outputs, target_tensor.view(-1))
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.train_dataloader)

    def train(self) -> None:
        start_time = time.time()
        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        decoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)

        criterion = nn.NLLLoss()
        log_loss = 0
        for epoch in range(1, self.num_epochs + 1):
            loss = self._train_per_epoch(
                encoder_optimizer,
                decoder_optimizer,
                criterion,
            )
            log_loss += loss
            if epoch % self.print_log_frequency == 0:
                average_loss = log_loss / self.print_log_frequency
                progress = epoch / self.num_epochs
                self.logger.info(
                    f"[{log_time(start_time, progress)}]: {average_loss:.4f}"
                )
                log_loss = 0

    def save(self) -> None:
        self.logger.info(f"Save the model state dicts to {self.checkpoint_path}")
        torch.save(
            {
                "encoder_state_dict": self.encoder.state_dict(),
                "decoder_state_dict": self.decoder.state_dict(),
            },
            self.checkpoint_path,
        )
