import argparse
import logging
from argparse import ArgumentParser
from pathlib import Path

import torch

from nlp_practice.case.translation.data.dataloader import PairDataLoader
from nlp_practice.case.translation.data.preprocessor import Preprocessor
from nlp_practice.case.translation.training.trainer import TransformerTrainer
from nlp_practice.model.transformer import Seq2SeqTransformer

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()


def fetch_args() -> argparse.Namespace:
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--embedding_size",
        type=int,
        dest="embedding_size",
        default=128,
        help="The embedding size",
    )
    arg_parser.add_argument(
        "--batch_size",
        type=int,
        dest="batch_size",
        default=32,
        help="The batch size",
    )
    arg_parser.add_argument(
        "--num_epochs",
        type=int,
        dest="num_epochs",
        default=100,
        help="The number of epochs",
    )
    arg_parser.add_argument(
        "--num_heads",
        type=int,
        dest="num_heads",
        default=8,
        help="The number of multi-heads",
    )
    arg_parser.add_argument(
        "--dim_feedforward",
        type=int,
        dest="dim_feedforward",
        default=128,
        help="The dimensionality of the feedforward layer",
    )
    arg_parser.add_argument(
        "--num_decoder_layers",
        type=int,
        dest="num_decoder_layers",
        default=1,
        help="The number of decoder layers",
    )
    arg_parser.add_argument(
        "--num_encoder_layers",
        type=int,
        dest="num_encoder_layers",
        default=1,
        help="The number of encoder layers",
    )
    arg_parser.add_argument(
        "--dropout_rate",
        type=float,
        dest="dropout_rate",
        default=0.1,
        help="The dropout rate",
    )
    arg_parser.add_argument(
        "--training_rate",
        type=float,
        dest="training_rate",
        default=0.7,
        help="The training rate",
    )
    arg_parser.add_argument(
        "--learning_rate",
        type=float,
        dest="learning_rate",
        default=0.001,
        help="The learning rate",
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
    arg_parser.add_argument(
        "--data_base_path",
        type=str,
        dest="data_base_path",
        default="examples/translation/data",
        help="Data base path",
    )
    arg_parser.add_argument(
        "--does_proceed_training",
        action="store_true",
        dest="does_proceed_training",
        help="Proceed model training based on the existing checkpoint",
    )
    arg_parser.add_argument(
        "--does_use_attention_decoder",
        action="store_true",
        dest="does_use_attention_decoder",
        help="Whether using attention in decoder or not",
    )
    return arg_parser.parse_args()


def run_training_job(args: argparse.Namespace) -> list[float]:
    input_language, output_language, pairs = Preprocessor(
        base_path=args.data_base_path,
        first_language="eng",
        second_language="fra",
        does_reverse=True,
    ).process()

    train_dataloader = PairDataLoader(
        pairs=pairs,
        input_language=input_language,
        output_language=output_language,
        training_rate=args.training_rate,
        batch_size=args.batch_size,
        device=args.device,
    ).train_dataloader

    transformer = Seq2SeqTransformer(
        embedding_size=args.embedding_size,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.num_decoder_layers,
        num_encoder_layers=args.num_encoder_layers,
        input_size=input_language.num_words,
        output_size=output_language.num_words,
    ).to(args.device)

    if Path(args.checkpoint_path).is_file() and args.does_proceed_training:
        LOGGER.info(f"Loading the parameters in checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path)
        transformer.load_state_dict(checkpoint["state_dict"])

    trainer = TransformerTrainer(
        train_dataloader=train_dataloader,
        transformer=transformer,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
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
    run_training_job(args)
