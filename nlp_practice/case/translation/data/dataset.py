from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from nlp_practice.case.translation import EOS_TOKEN, MAX_LENGTH
from nlp_practice.case.translation.data.data_handler import (
    LanguageData,
    index_from_sentence,
)
from nlp_practice.case.translation.utils import causal_mask


class PairDataset:
    """
    Examples
    --------
    >>> from nlp_practice.case.translation.data.dataset import PairDataset
    >>> from nlp_practice.case.translation.data.preprocessor import Preprocessor
    >>> input_language, output_language, pairs = Preprocessor(
    ...     base_path="examples/translation/data",
    ...     first_language="eng",
    ...     second_language="fra",
    ...     does_reverse=True,
    ... ).process()
    >>> dataset = PairDataset(pairs, input_language, output_language)()
    >>> len(dataset)
    11445
    >>> type(dataset())
    <torch.utils.data.dataset.TensorDataset at 0x117961390>
    """

    def __init__(
        self,
        pairs: list[list[str]],
        input_language: LanguageData,
        output_language: LanguageData,
    ) -> None:
        self.pairs = pairs
        self.input_language = input_language
        self.output_language = output_language

    def __len__(self) -> int:
        return len(self.pairs)

    def __call__(self) -> TensorDataset:
        n = len(self.pairs)
        input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
        target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

        for index, (input, target) in enumerate(self.pairs):
            inputs = index_from_sentence(self.input_language, input)
            targets = index_from_sentence(self.output_language, target)

            inputs.append(EOS_TOKEN)
            targets.append(EOS_TOKEN)
            input_ids[index, : len(inputs)] = inputs
            target_ids[index, : len(targets)] = targets

        return TensorDataset(torch.LongTensor(input_ids), torch.LongTensor(target_ids))


class BilingualDataset(Dataset):
    def __init__(
        self,
        dataset,
        source_tokenizer,
        target_tokenizer,
        source_langauge: str,
        target_language: str,
        seq_len: int,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_language = source_langauge
        self.target_language = target_language
        self.seq_len = seq_len

        self.sos_token = torch.Tensor(
            [source_tokenizer.token_to_id(["[SOS]"])], dtype=torch.int64
        )
        self.eos_token = torch.Tensor(
            [source_tokenizer.token_to_id(["[EOS]"])], dtype=torch.int64
        )
        self.pad_token = torch.Tensor(
            [source_tokenizer.token_to_id(["[PAD]"])], dtype=torch.int64
        )

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def meet_post_condition(
        self, encoder_input, decoder_input, encoder_mask, decoder_mask, label
    ):
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert encoder_mask.size(-1) == self.seq_len
        assert decoder_mask.size(-1) == self.seq_len
        assert label.size(0) == self.seq_len

    def __getitem__(self, index: Any) -> Any:
        source_target_pair = self.dataset[index]["translation"]
        source_text = source_target_pair[self.source_language]
        target_text = source_target_pair[self.target_language]

        tokens_in_encoder = self.target_tokenizer(source_text).ids
        tokens_in_decoder = self.target_tokenizer(target_text).ids

        # Encoder: 2 = SOS + EOS
        num_padding_tokens_in_encoder = self.seq_len - len(tokens_in_encoder) - 2
        # Decoder: SOS (only) c.f., we add the label with EOS at the end of the sentence
        num_padding_tokens_in_decoder = self.seq_len - len(tokens_in_decoder) - 1

        if num_padding_tokens_in_encoder < 0 or num_padding_tokens_in_decoder < 0:
            raise ValueError(f"Sentence length exceeds the requirement, {self.seq_len}")

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(tokens_in_encoder, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * num_padding_tokens_in_encoder, dtype=torch.int64
                ),
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(tokens_in_decoder, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * num_padding_tokens_in_decoder, dtype=torch.int64
                ),
            ]
        )
        label = torch.cat(
            [
                torch.tensor(tokens_in_decoder, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * num_padding_tokens_in_decoder, dtype=torch.int64
                ),
            ]
        )
        # Create encoder_mask by checking non-padding elements in encoder_input
        encoder_mask = (
            (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).to(torch.int)
        )
        # Create decoder_mask by combining non-padding elements in decoder_input with causal_mask
        decoder_mask = (
            (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0)
            & causal_mask(decoder_input.size(0))
        ).to(torch.int)

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "source_text": source_text,
            "target_text": target_text,
        }
