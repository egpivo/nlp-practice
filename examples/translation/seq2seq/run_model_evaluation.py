import logging
import random
from argparse import ArgumentParser

import torch

from examples.translation import (
    AttentionDecoderRNN,
    EncoderRNN,
    Evaluator,
    TrainDataloader,
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()


def fetch_args() -> "argparse.Namespace":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--checkpoint_path",
        type=str,
        dest="checkpoint_path",
        default="seq2seq.pt",
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


def run_evaluation_job(args: "argparse.Namespace") -> None:
    checkpoint = torch.load(args.checkpoint_path)

    dataloader_instance = TrainDataloader(checkpoint["batch_size"], args.device)
    input_language = dataloader_instance.input_language
    output_language = dataloader_instance.output_language

    encoder = EncoderRNN(
        input_size=input_language.num_words,
        hidden_size=checkpoint["hidden_size"],
        dropout_rate=checkpoint["dropout_rate"],
    ).to(args.device)

    decoder = AttentionDecoderRNN(
        hidden_size=checkpoint["hidden_size"],
        output_size=output_language.num_words,
        dropout_rate=checkpoint["dropout_rate"],
        device=args.device,
    ).to(args.device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    encoder.eval()
    decoder.eval()

    input_sentence, answer = random.choice(dataloader_instance.pairs)
    LOGGER.info(f"Translate {input_sentence!r} with the true sentence: {answer!r}")

    evaluator = Evaluator(encoder, decoder, input_language, output_language)
    LOGGER.info(f"Result: {' '.join(evaluator.evaluate(input_sentence))!r}")


if __name__ == "__main__":
    args = fetch_args()
    run_evaluation_job(args)
