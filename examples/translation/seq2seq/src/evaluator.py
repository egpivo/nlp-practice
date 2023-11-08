import logging

import torch

from examples.translation.seq2seq.src import EOS_TOKEN
from examples.translation.seq2seq.src.data_handler import (
    LanguageData,
    index_tensor_from_sentence,
)
from examples.translation.seq2seq.src.seq2seq import AttentionDecoderRNN, EncoderRNN

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()


class Evaluator:
    def __init__(
        self,
        encoder: EncoderRNN,
        decoder: AttentionDecoderRNN,
        input_language: LanguageData,
        output_language: LanguageData,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.input_language = input_language
        self.output_language = output_language

    def evaluate(self, sentence: str) -> list[str]:
        with torch.no_grad():
            input_indexes = index_tensor_from_sentence(self.input_language, sentence)
            encoder_outputs, encoder_hidden = self.encoder(input_indexes)
            decoder_outputs, decoder_hidden, _ = self.decoder(
                encoder_outputs, encoder_hidden
            )

            _, selected_ids = decoder_outputs.topk(1)
            decoded_ids = selected_ids.squeeze()

            decoded_words = []
            for index in decoded_ids:
                if index.item() == EOS_TOKEN:
                    decoded_words.append("<EOS>")
                    break
                decoded_words.append(self.output_language.index_to_word[index.item()])
            return decoded_words
