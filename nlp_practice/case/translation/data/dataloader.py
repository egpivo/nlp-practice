import torch
from torch.utils.data import DataLoader, random_split

from nlp_practice.case.translation.data.data_handler import LanguageData
from nlp_practice.case.translation.data.dataset import PairDataset


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
