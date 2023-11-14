from nlp_practice.case.translation import ENGLISH_PREFIXES
from nlp_practice.case.translation.data.data_handler import DataReader, LanguageData
from nlp_practice.case.translation.utils import filter_pairs


class Preprocessor:
    """
    Examples
    --------
    >>> import random
    >>> from nlp_practice.case.translation.data.preprocessor import Preprocessor
    >>> input_language, output_language, valid_pairs = Preprocessor(does_reverse=True).process()
    >>> random.choice(valid_pairs)
    ['il est mon epoux', 'he s my husband']
    """

    def __init__(
        self,
        base_path: str = "./data/translation",
        first_language: str = "eng",
        second_language: str = "fra",
        does_reverse: bool = False,
    ) -> None:
        self.base_path = base_path
        self.first_language = first_language
        self.second_language = second_language
        self.does_reverse = does_reverse

    def process(self) -> tuple[LanguageData, LanguageData, list[list[str]]]:
        input_language, output_language, pairs = DataReader(
            base_path=self.base_path,
            first_language=self.first_language,
            second_language=self.second_language,
            does_reverse=self.does_reverse,
        ).read()
        valid_pairs = filter_pairs(pairs, ENGLISH_PREFIXES)
        for pair in valid_pairs:
            input_language.add_sentence(pair[0])
            output_language.add_sentence(pair[1])

        return input_language, output_language, valid_pairs
