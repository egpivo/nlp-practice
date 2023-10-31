import re
import unicodedata
from io import open

from examples.translation import ENGLISH_PREFIXES, MAX_LENGTH


def remove_accents(string: str) -> str:
    """
    Examples
    --------
    >>> convert_unicode_to_ascii(u'A\u2019A')
    'A’A'

    References
    ----------
    - https://stackoverflow.com/a/66020622
    """
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", string)
        if not unicodedata.combining(char)
    )


def normalize_string(string: str) -> str:
    """
    Examples
    --------
    >>> normalize_string(u'I\u2019m    happy.Really,,, ~..!')
    'i m happy .really . . !'
    """
    lower_string = string.lower().strip()
    removed_accents = remove_accents(lower_string)
    target_punctuations = re.sub(r"([.!?])", r" \1", removed_accents)
    removed_punctuations = re.sub(r"[^a-zA-Z.!?]+", r" ", target_punctuations)
    return removed_punctuations


def read_file(language1: str = "eng", language2: str = "fra") -> list[str]:
    """
    Examples
    --------
    >>> read_file()[0]
    'Go.\tVa !'
    """
    with open(
        f"./data/translation/{language1}-{language2}.txt", encoding="utf-8"
    ) as file:
        return file.read().strip().split("\n")


def is_valid_pair(pair: list[str], prefixes: tuple[str] = ENGLISH_PREFIXES) -> bool:
    return (
        len(pair[0].split(" ")) < MAX_LENGTH
        and len(pair[1].split(" ") < MAX_LENGTH)
        and pair[1].startswith(prefixes)
    )


def filter_pairs(pairs: list[list[str]]) -> list[list[str]]:
    return [pair for pair in pairs if is_valid_pair(pair)]
