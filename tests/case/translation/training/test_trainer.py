import pytest
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from nlp_practice.case.translation.data.dataloader import PairDataLoader
from nlp_practice.case.translation.data.preprocessor import Preprocessor
from nlp_practice.case.translation.training.trainer import Trainer, TransformerTrainer
from nlp_practice.model.decoder import DecoderRNN
from nlp_practice.model.encoder import EncoderRNN
from nlp_practice.model.transformer import Seq2SeqTransformer


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
    trainer = Trainer(
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
def train_dataloader():
    # Create a dummy dataset for testing
    # Replace this with your actual DataLoader creation logic
    dataset = [
        (torch.randint(0, 10, (8,)), torch.randint(0, 10, (10,))) for _ in range(100)
    ]
    return DataLoader(dataset, batch_size=16, shuffle=True)


@pytest.fixture
def transformer():
    # Create a dummy Seq2SeqTransformer for testing
    # Replace this with your actual Seq2SeqTransformer creation logic
    return Seq2SeqTransformer(
        embedding_size=512,
        num_heads=8,
        dim_feedforward=2048,
        num_encoder_layers=6,
        num_decoder_layers=6,
        input_size=10,
        output_size=12,
        max_length=100,
    )


def test_transformer_trainer_init(train_dataloader, transformer):
    num_epochs = 10
    learning_rate = 0.001
    print_log_frequency = 5

    trainer = TransformerTrainer(
        train_dataloader=train_dataloader,
        transformer=transformer,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        print_log_frequency=print_log_frequency,
    )

    assert trainer.train_dataloader == train_dataloader
    assert trainer.transformer == transformer
    assert trainer.num_epochs == num_epochs
    assert trainer.learning_rate == learning_rate
    assert trainer.print_log_frequency == print_log_frequency
    assert isinstance(trainer._criterion, nn.NLLLoss)
    assert isinstance(trainer._optimizer, optim.Adam)
    assert trainer._optimizer.param_groups[0]["lr"] == learning_rate


def test_transformer_trainer_train(train_dataloader, transformer, monkeypatch):
    # Mock data
    num_epochs = 2
    learning_rate = 0.001
    print_log_frequency = 5

    trainer = TransformerTrainer(
        train_dataloader=train_dataloader,
        transformer=transformer,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        print_log_frequency=print_log_frequency,
    )

    # Mock the training loop to avoid actual training
    def mock_train_per_epoch(self):
        return 1.0

    monkeypatch.setattr(TransformerTrainer, "_train_per_epoch", mock_train_per_epoch)

    losses = trainer.train()

    assert len(losses) == num_epochs
    assert all(isinstance(loss, float) for loss in losses)
    assert trainer._optimizer.param_groups[0]["lr"] == learning_rate
