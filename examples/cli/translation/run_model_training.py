import logging
from argparse import ArgumentParser
from pathlib import Path

import torch

from nlp_practice.case.translation.data.dataloader import TrainDataloader
from nlp_practice.case.translation.model.decoder import AttentionDecoderRNN, DecoderRNN
from nlp_practice.case.translation.model.encoder import EncoderRNN
from nlp_practice.case.translation.trainer import Trainer

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()


def fetch_args() -> "argparse.Namespace":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--hidden_size",
        type=int,
        dest="hidden_size",
        default=128,
        help="The size of the hidden layer",
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
        "--dropout_rate",
        type=float,
        dest="dropout_rate",
        default=0.1,
        help="The dropout rate",
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
        default="translation.pt",
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
        "--data_bath_path",
        type=str,
        dest="data_bath_path",
        default="./examples/data/translation",
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


def run_training_job(args: "argparse.Namespace") -> list[float]:
    dataloader_instance = TrainDataloader(
        args.batch_size, args.device, args.data_bath_path
    )
    dataloader = dataloader_instance.dataloader
    input_language = dataloader_instance.input_language
    output_language = dataloader_instance.output_language
    encoder = EncoderRNN(
        input_size=input_language.num_words,
        hidden_size=args.hidden_size,
        dropout_rate=args.dropout_rate,
    ).to(args.device)

    decoder_class = (
        AttentionDecoderRNN if args.does_use_attention_decoder else DecoderRNN
    )
    decoder = decoder_class(
        hidden_size=args.hidden_size,
        output_size=output_language.num_words,
        dropout_rate=args.dropout_rate,
        device=args.device,
    ).to(args.device)

    if Path(args.checkpoint_path).is_file() and args.does_proceed_training:
        LOGGER.info(f"Loading the parameters in checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])

    trainer = Trainer(
        train_dataloader=dataloader,
        encoder=encoder,
        decoder=decoder,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        checkpoint_path=args.checkpoint_path,
        logger=LOGGER,
    )
    loss = trainer.train()
    trainer.save()
    return loss


if __name__ == "__main__":
    args = fetch_args()
    run_training_job(args)
