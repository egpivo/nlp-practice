import logging
import random

import torch

from examples import EOS_TOKEN
from examples.translation.data_handler import LanguageData, index_tensor_from_sentence
from examples.translation.seq2seq import AttentionDecoderRNN, EncoderRNN
from examples.translation.train_dataloader import TrainDataloader

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()


def evaluate(
    encoder: EncoderRNN,
    decoder: AttentionDecoderRNN,
    sentence: str,
    input_language: LanguageData,
    output_language: LanguageData,
):
    with torch.no_grad():
        input_indexes = index_tensor_from_sentence(input_language, sentence)
        encoder_outputs, encoder_hidden = encoder(input_indexes)
        decoder_outputs, decoder_hidden, _ = decoder(encoder_outputs, encoder_hidden)

        _, selected_ids = decoder_outputs.topk(1)
        decoded_ids = selected_ids.squeeze()

        decoded_words = []
        for index in decoded_ids:
            if index.item() == EOS_TOKEN:
                decoded_words.append("<EOS>")
                break
            decoded_words.append(output_language.index_to_word[index.item()])
        return decoded_words


if __name__ == "__main__":
    hidden_size = 128
    batch_size = 32
    num_epochs = 100
    dropout_rate = 0.2
    checkpoint_path = "seq2seq.pt"
    device = "cpu"

    dataloader_instance = TrainDataloader(batch_size, device)
    dataloader = dataloader_instance.dataloader
    input_language = dataloader_instance.input_language
    output_language = dataloader_instance.output_language
    checkpoint = torch.load(checkpoint_path)

    encoder = EncoderRNN(input_language.num_words, hidden_size, dropout_rate).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])

    decoder = AttentionDecoderRNN(
        hidden_size, output_language.num_words, dropout_rate, device
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    encoder.eval()
    decoder.eval()
    input_sentence, answer = random.choice(dataloader_instance.pairs)

    tanslated_sentence = evaluate(
        encoder, decoder, input_sentence, input_language, output_language
    )
    LOGGER.info(
        f"Translate {input_sentence} to {''.join(tanslated_sentence)} | True: {answer}"
    )
