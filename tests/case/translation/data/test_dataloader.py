import pytest

from nlp_practice.case.translation.data.dataloader import PairDataLoader, PairDataset
from nlp_practice.case.translation.data.preprocessor import Preprocessor


@pytest.fixture
def preprocessor():
    return Preprocessor(
        base_path="examples/translation/data",
        first_language="eng",
        second_language="fra",
        does_reverse=True,
    )


@pytest.fixture
def pairs(preprocessor):
    _, _, pairs = preprocessor.process()
    return pairs


def test_pair_dataset(pairs, preprocessor):
    input_language, output_language, _ = preprocessor.process()
    dataset = PairDataset(pairs, input_language, output_language)()
    assert len(dataset) == len(pairs)


def test_pair_dataloader(pairs, preprocessor):
    input_language, output_language, _ = preprocessor.process()
    dataloader = PairDataLoader(
        pairs,
        input_language,
        output_language,
        training_rate=0.8,
        batch_size=64,
        device="cpu",
        num_workers=4,
    )

    assert len(dataloader.train_dataloader.dataset) == int(0.8 * len(pairs))
    assert len(dataloader.test_dataloader.dataset) == len(pairs) - len(
        dataloader.train_dataloader.dataset
    )
