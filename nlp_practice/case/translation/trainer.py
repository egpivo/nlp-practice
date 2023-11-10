import logging

import torch
import torch.nn as nn
from torch import optim
from tqdm import trange

from nlp_practice.case.translation.data.dataloader import TrainDataloader
from nlp_practice.case.translation.model.decoder import Decoder
from nlp_practice.case.translation.model.encoder import EncoderRNN


class Trainer:
    def __init__(
        self,
        train_dataloader: TrainDataloader,
        encoder: EncoderRNN,
        decoder: Decoder,
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

        self._criterion = nn.NLLLoss()
        self._encoder_optimizer = optim.Adam(
            self.encoder.parameters(), lr=self.learning_rate
        )
        self._decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=self.learning_rate
        )

    def _train_per_epoch(self) -> float:
        total_loss = 0
        for data in self.train_dataloader:
            input_tensor, target_tensor = data
            self._encoder_optimizer.zero_grad()
            self._decoder_optimizer.zero_grad()
            encoder_outputs, encoder_hidden = self.encoder(input_tensor)
            decoder_outputs, _, _ = self.decoder(
                encoder_outputs, encoder_hidden, target_tensor
            )
            loss = self._criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1),
            )
            loss.backward()
            self._encoder_optimizer.step()
            self._decoder_optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.train_dataloader)

    def train(self) -> list[float]:
        return [self._train_per_epoch() for _ in trange(self.num_epochs)]

    def save(self) -> None:
        self.logger.info(f"Save the model state dicts to {self.checkpoint_path}")
        torch.save(
            {
                "encoder_state_dict": self.encoder.state_dict(),
                "decoder_state_dict": self.decoder.state_dict(),
                "num_epochs": self.num_epochs,
                "learning_rate": self.learning_rate,
                "batch_size": self.train_dataloader.batch_size,
                "dropout_rate": self.encoder.dropout_rate,
                "hidden_size": self.encoder.hidden_size,
            },
            self.checkpoint_path,
        )
