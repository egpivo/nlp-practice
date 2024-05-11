from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer


class TokenizerBuilder:
    def __init__(self, config, dataset, language) -> None:
        self.config = config
        self.dataset = dataset
        self.language = language

    def _get_sentence(self):
        for item in self.dataset:
            yield item["translation"][self.language]

    def build(self):
        tokenizer_path = str(Path(self.config["tokenizer_file"].fomat(self.anguage)))
        if not Path.exists(tokenizer_path):
            tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(
                special_tokens=["[UNK]", "[PAD]", "[SOS]", "EOS"], min_frequency=2
            )
            tokenizer.train_from_iterator(self._get_sentence(), trainer=trainer)
            tokenizer.save(tokenizer_path)
        else:
            tokenizer = Tokenizer.from_file(tokenizer_path)
        return tokenizer
