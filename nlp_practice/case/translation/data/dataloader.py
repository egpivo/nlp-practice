import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split

from nlp_practice.case.translation.data.data_handler import LanguageData
from nlp_practice.case.translation.data.dataset import BilingualDataset, PairDataset
from nlp_practice.case.translation.data.tokenizer import TokenizerBuilder


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


class BilingualDataLoader:
    def __init__(
        self,
        dataset: str,
        config: dict,
        training_rate: float,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.source_language = self.config["source_language"]
        self.target_language = self.config["target_language"]
        self.training_rate = training_rate

        self.raw_data = self._get_raw_data()
        self._source_tokenizer = TokenizerBuilder(
            self.config, self.raw_data, self.source_language
        )
        self._target_tokenizer = TokenizerBuilder(
            self.config, self.raw_data, self.target_language
        )

        self.train_size = int(self.training_rate * len(self.raw_data))
        self.val_size = len(self.raw_data) - self.train_size

    def _get_raw_data(self):
        return load_dataset(
            "opus_books",
            f"{self.source_language}-{self.target_language}",
            split="train",
        )

    @property
    def source_tokenizer(self):
        return self._source_tokenizer

    @property
    def target_tokenizer(self):
        return self._target_tokenizer

    @property
    def train_dataloader(self) -> DataLoader:
        train_raw_data, _ = random_split(
            self.raw_data, [self.train_size, self.val_size]
        )

        train_dataset = BilingualDataset(
            train_raw_data,
            self._source_tokenizer,
            self._target_tokenizer,
            self.source_language,
            self.target_language,
        )

        return DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.config["batch_size"],
        )

    @property
    def val_dataloader(self) -> DataLoader:
        _, val_raw_data = random_split(self.raw_data, [self.train_size, self.val_size])

        val_dataset = BilingualDataset(
            val_raw_data,
            self._source_tokenizer,
            self._target_tokenizer,
            self.source_language,
            self.target_language,
        )

        return DataLoader(
            val_dataset,
            shuffle=True,
            batch_size=self.config["batch_size"],
        )
