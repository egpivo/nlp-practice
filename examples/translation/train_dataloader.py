import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from examples.translation import EOS_TOKEN, MAX_LENGTH
from examples.translation.data_handler import LanguageData
from examples.translation.preprocessor import Preprocessor


def index_from_sentence(language: LanguageData, sentence: str):
    return [language.word_to_index[word] for word in sentence.split(" ")][:MAX_LENGTH]


class TrainDataloader:
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
