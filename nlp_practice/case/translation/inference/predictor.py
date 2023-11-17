import torch

from nlp_practice.case.translation import EOS_TOKEN
from nlp_practice.case.translation.data.data_handler import (
    LanguageData,
    index_tensor_from_sentence,
)
from nlp_practice.model.decoder import AttentionDecoderRNN
from nlp_practice.model.encoder import EncoderRNN


class Predictor:
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

    def predict_by_index(self, input_indexes: list[int]) -> torch.Tensor:
        with torch.no_grad():
            encoder_outputs, encoder_hidden = self.encoder(input_indexes)
            decoder_outputs, decoder_hidden, _ = self.decoder(
                encoder_outputs, encoder_hidden
            )

            _, selected_ids = decoder_outputs.topk(1)
            decoded_ids = selected_ids.squeeze()
        return decoded_ids

    def predict_by_sentence(self, sentence: str) -> torch.Tensor:
        input_indexes = index_tensor_from_sentence(self.input_language, sentence)
        return self.predict_by_index(input_indexes)

    def translate(self, sentence: str) -> list[str]:
        decoded_words = []
        for index in self.predict_by_sentence(sentence):
            if index.item() == EOS_TOKEN:
                break
            decoded_words.append(self.output_language.index_to_word[index.item()])
        return decoded_words
