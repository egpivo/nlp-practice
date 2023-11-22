import pytest
import torch
from torch.utils.data import DataLoader

from nlp_practice.case.translation.data.dataloader import PairDataLoader
from nlp_practice.case.translation.data.preprocessor import Preprocessor
from nlp_practice.case.translation.training.trainer import (
    Seq2SeqTrainer,
    Trainer,
    TransformerTrainer,
)
from nlp_practice.model.decoder import DecoderRNN
from nlp_practice.model.encoder import EncoderRNN
from nlp_practice.model.transformer import Seq2SeqTransformer


@pytest.fixture
def empty_dataloader():
    return DataLoader([])


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
    ).train_dataloader
    return input_language, output_language, data_loader


def test_trainer(sample_data):
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
    trainer = Seq2SeqTrainer(
        train_dataloader=sample_data[2],
        encoder=encoder,
        decoder=decoder,
        num_epochs=num_epochs,
        learning_rate=0.001,
    )
    # Mock the optimization steps (not executing actual optimization)
    trainer._encoder_optimizer.step = lambda: None
    trainer._decoder_optimizer.step = lambda: None
    losses = trainer.train()

    assert len(losses) == num_epochs
    assert all(isinstance(loss, float) for loss in losses)


@pytest.fixture
def seq2seq_transformer():
    # Use the same settings as in the Seq2SeqTransformer tests
    embedding_size = 512
    num_heads = 8
    dim_feedforward = 2048
    num_encoder_layers = 6
    num_decoder_layers = 6
    input_size = 7
    output_size = 10
    max_length = 100

    return Seq2SeqTransformer(
        embedding_size,
        num_heads,
        dim_feedforward,
        num_encoder_layers,
        num_decoder_layers,
        input_size,
        output_size,
        max_length=max_length,
    )


@pytest.fixture
def dataloader():
    # Use the same batch size and other relevant settings
    batch_size = 5
    input_size = 7
    output_size = 10

    input_tensors = torch.randint(0, input_size, (batch_size, input_size))
    target_tensors = torch.randint(0, output_size, (batch_size, output_size))

    dataset = list(zip(input_tensors, target_tensors))
    return DataLoader(dataset)


def test_transformer_trainer_forward(seq2seq_transformer, dataloader):
    # Use the same settings as in the Seq2SeqTransformer tests
    num_epochs = 2
    learning_rate = 0.001
    print_log_frequency = 5

    trainer = TransformerTrainer(
        train_dataloader=dataloader,
        transformer=seq2seq_transformer,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        print_log_frequency=print_log_frequency,
    )
    losses = trainer.train()

    # You might want to add more assertions based on the actual expected losses
    assert len(losses) == num_epochs


def test_empty_dataloader_trainer(sample_data, empty_dataloader):
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
    with pytest.raises(
        ValueError, match="Empty dataloader. Cannot train without any batches."
    ):
        trainer = Seq2SeqTrainer(
            train_dataloader=empty_dataloader,
            encoder=encoder,
            decoder=decoder,
            num_epochs=num_epochs,
            learning_rate=0.001,
        )


@pytest.fixture
def sample_dataloader():
    # Create a DataLoader with some dummy data
    input_size = 7
    output_size = 10
    batch_size = 2

    input_tensors = torch.randint(0, input_size, (batch_size, input_size))
    target_tensors = torch.randint(0, output_size, (batch_size, output_size))

    dataset = list(zip(input_tensors, target_tensors))
    return DataLoader(dataset)


class MockTrainer(Trainer):
    pass


def test_train_per_epoch_abstract_method(sample_dataloader):
    mock_trainer = MockTrainer(sample_dataloader, num_epochs=1, learning_rate=0.001)

    # No exception should be raised here, as we have implemented the abstract method
    try:
        mock_trainer._train_per_epoch()
    except NotImplementedError as e:
        assert "Abstract method _train_per_epoch must be implemented" in str(e)
