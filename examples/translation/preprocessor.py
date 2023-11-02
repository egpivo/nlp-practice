from examples.translation.data_handler import DataReader
from examples.translation.utils import filter_pairs


class Preprocessor:
    def __init__(
        self,
        first_language: str = "eng",
        second_language: str = "fra",
        does_reverse: bool = False,
    ) -> None:
        self.first_language = first_language
        self.second_language = second_language
        self.does_reverse = does_reverse

    def process(self):
        input_language, output_language, pairs = DataReader(
            self.first_language, self.second_language, self.does_reverse
        ).read()
        valid_pairs = filter_pairs(pairs)
        for pair in valid_pairs:
            if "carded" in pair[1]:
                print(pair)
            input_language.add_sentence(pair[0])
            output_language.add_sentence(pair[1])
        return input_language, output_language, valid_pairs
