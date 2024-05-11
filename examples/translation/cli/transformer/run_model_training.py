import argparse
import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
import yaml

from nlp_practice.case.translation.data.dataloader import BilingualDataLoader
from nlp_practice.case.translation.training.trainer import TransformerTrainer
from nlp_practice.model.transformer import CustomTransformerBuilder

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()


def fetch_args() -> argparse.Namespace:
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--config_path",
        type=str,
        dest="config_path",
        default="configs/transformer.yaml",
        help="Model config",
    )
    arg_parser.add_argument(
        "--checkpoint_path",
        type=str,
        dest="checkpoint_path",
        default="transformer.pt",
        help="The model checkpoint path",
    )
    arg_parser.add_argument(
        "--device",
        type=str,
        dest="device",
        default="cpu",
        help="The device used in training",
    )
    return arg_parser.parse_args()


def run_training_job(
    args: argparse.Namespace,
    config: dict,
    source_vocabulary_size: int = 100,
    target_vocabulary_size: int = 100,
) -> list[float]:
    dataloader_instance = BilingualDataLoader(
        config=config, training_rate=config["training_rate"]
    )
    train_dataloader = dataloader_instance.train_dataloader
    dataloader_instance.val_dataloader

    source_tokenizer = dataloader_instance.source_tokenizer
    dataloader_instance.target_tokenizer

    transformer = CustomTransformerBuilder(
        source_vocabulary_size=source_tokenizer.get_vocab_size(),
        target_vocabulary_size=source_tokenizer.get_vocab_size(),
        source_seq_len=config["seq_len"],
        target_seq_len=config["seq_len"],
        embeddings_size=config["embedding_size"],
    ).build()

    if Path(args.checkpoint_path).is_file() and args.does_proceed_training:
        LOGGER.info(f"Loading the parameters in checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path)
        transformer.load_state_dict(checkpoint["state_dict"])

    trainer = TransformerTrainer(
        config=config,
        train_dataloader=train_dataloader,
        transformer=transformer,
    )
    loss = trainer.train()

    LOGGER.info(f"Save the model state dicts to {args.checkpoint_path}")
    torch.save(
        {
            "encoder_state_dict": transformer.state_dict(),
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "embedding_size": args.embedding_size,
            "num_heads": args.num_heads,
            "dim_feedforward": args.dim_feedforward,
            "num_decoder_layers": args.num_decoder_layers,
            "num_encoder_layers": args.num_encoder_layers,
            "dropout_rate": args.dropout_rate,
            "training_rate": args.training_rate,
        },
        args.checkpoint_path,
    )
    return loss


if __name__ == "__main__":
    args = fetch_args()
    with open(args.config_path, "rt") as f:
        model_config = yaml.safe_load(f.read())
    run_training_job(args, model_config)
