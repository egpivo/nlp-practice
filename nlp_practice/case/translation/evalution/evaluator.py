import logging
from typing import Generator

import torch
from torch.utils.data import DataLoader
from torchmetrics.text.rouge import ROUGEScore

from nlp_practice.case.translation.inference.predictor import Predictor

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()


class Evaluator:
    def __init__(
        self,
        test_dataloader: DataLoader,
        predictor: Predictor,
    ) -> None:
        self.test_dataloader = test_dataloader
        self.predictor = predictor

    @staticmethod
    def _calculate_rouge1_by_batch(
        predicted: torch.Tensor, target: torch.Tensor, metric: str
    ) -> Generator[float, None, None]:
        converter = lambda tensor: " ".join([str(id.item()) for id in tensor])
        for predict_ids, target_ids in zip(predicted, target):
            nonzero_index = target_ids.count_nonzero()
            predicts = converter(predict_ids[:nonzero_index])
            targets = converter(target_ids[:nonzero_index])
            yield ROUGEScore(rouge_keys=("rouge1",))(predicts, targets)[
                f"rouge1_{metric}"
            ]

    @staticmethod
    def _calculate_accuracy(
        predicted: torch.Tensor, target: torch.Tensor
    ) -> Generator[torch.Tensor, None, None]:
        for predict_ids, target_ids in zip(predicted, target):
            nonzero_index = target_ids.count_nonzero()
            yield torch.equal(predict_ids[:nonzero_index], target_ids[:nonzero_index])

    def _calculate_average_rough1(self, metric: str) -> float:
        score = sum(
            sum(
                self._calculate_rouge1_by_batch(
                    self.predictor.predict_by_index(input), target, metric
                )
            ).item()
            / len(target)
            for input, target in self.test_dataloader
        )
        return score / len(self.test_dataloader)

    @property
    def rouge1_precision(self) -> float:
        return self._calculate_average_rough1("precision")

    @property
    def rouge1_recall(self) -> float:
        return self._calculate_average_rough1("recall")

    @property
    def rouge1_f1(self) -> float:
        return self._calculate_average_rough1("fmeasure")

    @property
    def accuracy(self) -> float:
        accuracy = 0
        for input, target in self.test_dataloader:
            predicted = self.predictor.predict_by_index(input)
            accuracy += sum(self._calculate_accuracy(predicted, target)) / len(target)
        return accuracy / len(self.test_dataloader)
