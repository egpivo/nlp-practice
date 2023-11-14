import pytest
import torch

from nlp_practice.case.translation.data.dataloader import PairDataLoader
from nlp_practice.case.translation.data.preprocessor import Preprocessor
from nlp_practice.case.translation.Inference.predictor import Predictor
from nlp_practice.case.translation.training.trainer import Trainer
from nlp_practice.model.decoder import DecoderRNN
from nlp_practice.model.encoder import EncoderRNN


@pytest.fixture
def sample_data():
    input_language, output_language, pairs = Preprocessor(
        base_path="examples/translation/data",
        first_language="eng",
        second_language="fra",
        does_reverse=True,
    ).process()
    data_loader = PairDataLoader(
        pairs=pairs,
        input_language=input_language,
        output_language=output_language,
        training_rate=0.8,
        batch_size=64,
        device="cpu",
        num_workers=4,
        random_seed=0,
    )
    return input_language, output_language, data_loader


def test_predictor(sample_data):
    encoder = EncoderRNN(
        input_size=sample_data[0].num_words, hidden_size=10, dropout_rate=0.1
    )
    decoder = DecoderRNN(
        output_size=sample_data[1].num_words,
        hidden_size=10,
        dropout_rate=0.1,
        device="cpu",
    )
    num_epochs = 1
    trainer = Trainer(
        train_dataloader=sample_data[2].train_dataloader,
        encoder=encoder,
        decoder=decoder,
        num_epochs=num_epochs,
        learning_rate=0.001,
    )
    # Mock the optimization steps (not executing actual optimization)
    trainer._encoder_optimizer.step = lambda: None
    trainer._decoder_optimizer.step = lambda: None
    _ = trainer.train()
    encoder.eval()
    decoder.eval()

    predictor = Predictor(
        encoder=encoder,
        decoder=decoder,
        input_language=sample_data[0],
        output_language=sample_data[1],
    )

    # Test predict_by_index
    input_indexes, _ = next(iter(sample_data[2].test_dataloader))
    decoded_ids = predictor.predict_by_index(input_indexes)
    assert isinstance(decoded_ids, torch.Tensor)
    assert all(isinstance(decoded_id, torch.Tensor) for decoded_id in decoded_ids)
