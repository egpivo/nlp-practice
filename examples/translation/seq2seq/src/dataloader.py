import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from examples.translation.seq2seq.src import EOS_TOKEN, MAX_LENGTH
from examples.translation.seq2seq.src.data_handler import index_from_sentence
from examples.translation.seq2seq.src.preprocessor import Preprocessor


class TrainDataloader:
    """
    Examples
    --------
    >>> from examples.translation.seq2seq.src.dataloader import TrainDataloader
    >>> dataloader = TrainDataloader(64, "cpu").dataloader
    >>> len(dataloader)
    179
    >>> next(iter(dataloader))[0].shape
    """

    def __init__(self, batch_size: int, device: str) -> None:
        self.batch_size = batch_size
        self.device = device
        self.input_language, self.output_language, self.pairs = Preprocessor(
            "eng", "fra", True
        ).process()

    @property
    def dataloader(self) -> DataLoader:
        n = len(self.pairs)
        input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
        target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

        for index, (input, target) in enumerate(self.pairs):
            inputs = index_from_sentence(self.input_language, input)
            targets = index_from_sentence(self.output_language, target)

            inputs.append(EOS_TOKEN)
            targets.append(EOS_TOKEN)
            input_ids[index, : len(inputs)] = inputs
            target_ids[index, : len(targets)] = targets

        train_data = TensorDataset(
            torch.LongTensor(input_ids).to(self.device),
            torch.LongTensor(target_ids).to(self.device),
        )
        train_dataloader = DataLoader(
            train_data, sampler=RandomSampler(train_data), batch_size=self.batch_size
        )
        return train_dataloader
