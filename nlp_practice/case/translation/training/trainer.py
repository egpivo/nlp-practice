import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import trange

from nlp_practice.case.translation.training.utils import (
    create_masks,
    create_padding_masks,
)
from nlp_practice.model.decoder import Decoder
from nlp_practice.model.encoder import EncoderRNN
from nlp_practice.model.transformer import Seq2SeqTransformer


class Trainer:
    def __init__(
        self,
        train_dataloader: DataLoader,
        encoder: EncoderRNN,
        decoder: Decoder,
        num_epochs: int,
        learning_rate: float,
        print_log_frequency: int = 10,
    ):
        self.train_dataloader = train_dataloader
        self.encoder = encoder
        self.decoder = decoder
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
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
        num_batches = len(self.train_dataloader)

        if num_batches == 0:
            return
        for input_tensor, target_tensor in self.train_dataloader:
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
        return total_loss / num_batches

    def train(self) -> list[float]:
        return [self._train_per_epoch() for _ in trange(self.num_epochs)]


class TransformerTrainer:
    def __init__(
        self,
        train_dataloader: DataLoader,
        transformer: Seq2SeqTransformer,
        num_epochs: int,
        learning_rate: float,
        print_log_frequency: int = 10,
    ):
        self.train_dataloader = train_dataloader
        self.transformer = transformer
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.print_log_frequency = print_log_frequency

        self._criterion = nn.NLLLoss()
        self._optimizer = optim.Adam(
            self.transformer.parameters(), lr=self.learning_rate
        )

    def _train_per_epoch(self) -> float:
        total_loss = 0
        num_batches = len(self.train_dataloader)

        if num_batches == 0:
            return
        for input_tensor, target_tensor in self.train_dataloader:
            self._optimizer.zero_grad()
            target_input, target_output = target_tensor[:, :-1], target_tensor[:, 1:]
            input_mask, output_mask = create_masks(input_tensor, target_input)
            input_padding_mask, output_padding_mask = create_padding_masks(
                input_tensor, target_input
            )
            logits = self.transformer(
                input=input_tensor,
                output=target_input,
                input_mask=input_mask,
                output_mask=output_mask,
                memory_mask=None,
                input_padding_mask=input_padding_mask,
                output_padding_mask=output_padding_mask,
                memory_key_padding_mask=None,
            )
            loss = self._criterion(
                logits.view(-1, logits.size(-1)),
                target_output.view(-1),
            )
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()
        return total_loss / num_batches

    def train(self) -> list[float]:
        return [self._train_per_epoch() for _ in trange(self.num_epochs)]
