import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from nlp_practice.case.translation import EOS_TOKEN, MAX_LENGTH
from nlp_practice.case.translation.data.data_handler import (
    LanguageData,
    index_from_sentence,
)


class PairDataset:
    """
    Examples
    --------
    >>> from nlp_practice.case.translation.data.dataloader import PairDataset
    >>> from nlp_practice.case.translation.data.preprocessor import Preprocessor
    >>> input_language, output_language, pairs = Preprocessor(
    ...     base_path="examples/translation/data",
    ...     first_language="eng",
    ...     second_language="fra",
    ...     does_reverse=True,
    ... ).process()
    >>> dataset = PairDataset(pairs, input_language, output_language)()
    >>> len(dataset)
    11445
    >>> type(dataset())
    <torch.utils.data.dataset.TensorDataset at 0x117961390>
    """

    def __init__(
        self,
        pairs: list[list[str]],
        input_language: LanguageData,
        output_language: LanguageData,
    ) -> None:
        self.pairs = pairs
        self.input_language = input_language
        self.output_language = output_language

    def __len__(self) -> TensorDataset:
        return len(self.pairs)

    def __call__(self) -> TensorDataset:
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

        return TensorDataset(torch.LongTensor(input_ids), torch.LongTensor(target_ids))


class PairDataLoader:
    def __init__(
        self,
        pairs: list[list[str]],
        input_language: LanguageData,
        output_language: LanguageData,
        training_rate: float,
        batch_size: int,
        device: str = "cpu",
        num_workers: int = 4,
        random_seed: float = 0,
    ) -> None:
        self.dataset = PairDataset(pairs, input_language, output_language)
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers

        self.generator = torch.Generator()
        self.generator.manual_seed(random_seed)
        self.train_size = int(training_rate * len(self.dataset))
        self.val_size = 0
        self.test_size = len(self.dataset) - self.train_size

    @property
    def train_dataloader(self) -> DataLoader:
        train_dataset, _, _ = random_split(
            dataset=self.dataset(),
            lengths=[self.train_size, self.val_size, self.test_size],
            generator=self.generator,
        )
        return DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    @property
    def test_dataloader(self) -> DataLoader:
        _, _, test_dataset = random_split(
            dataset=self.dataset(),
            lengths=[self.train_size, self.val_size, self.test_size],
            generator=self.generator,
        )
        return DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
