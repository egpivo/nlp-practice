from collections import defaultdict

from examples.translation.utils import normalize_string, read_file


class LanguageData:
    def __init__(self, name: str) -> None:
        self.name = name
        self.word_to_index = {}
        self.word_to_count = defaultdict(int)
        self.index_to_word = {0: "SOS", 1: "EOS"}
        self.num_words = 2

    def add_sentence(self, sentence: str) -> None:
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word: str) -> None:
        if word not in self.word_to_index:
            self.word_to_index[word] = self.num_words
            self.index_to_word[self.num_words] = word
            self.num_words += 1
        self.word_to_count[word] += 1


class DataReader:
    def __init__(
        self,
        first_language: str = "eng",
        second_language: str = "fra",
        does_reverse: bool = False,
    ) -> None:
        self.first_language = first_language
        self.second_language = second_language
        self.does_reverse = does_reverse

    def read(self) -> tuple[LanguageData, LanguageData, list[list[str]]]:
        lines = read_file(self.first_language, self.second_language)
        pairs = [[normalize_string(string) for string in line] for line in lines]

        if self.does_reverse:
            pairs = [list(reversed(pair)) for pair in pairs]
            input_language = LanguageData(self.second_language)
            output_language = LanguageData(self.langufirst_languageage1)
        else:
            input_language = LanguageData(self.first_language)
            output_language = LanguageData(self.second_language)
        return input_language, output_language, pairs
