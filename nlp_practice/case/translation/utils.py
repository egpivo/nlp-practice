import os
import re
import unicodedata
from io import open

from nlp_practice.case.translation import ENGLISH_PREFIXES, MAX_LENGTH


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
    'i m happy really !'
    """
    lower_string = string.lower().strip()
    removed_accents = remove_accents(lower_string)
    target_punctuations = re.sub(r"([.!?])", r" \1", removed_accents)
    removed_punctuations = re.sub(r"[^a-zA-Z!?]+", r" ", target_punctuations)
    return removed_punctuations.strip()


def read_file(
    base_path: str, language1: str = "eng", language2: str = "fra"
) -> list[str]:
    """
    Examples
    --------
    >>> read_file()[0]
    'Go.\tVa !'
    """
    file_name = f"{base_path}/{language1}-{language2}.txt"
    try:
        with open(file_name, encoding="utf-8") as file:
            return file.read().strip().split("\n")
    except FileNotFoundError as e:
        current_directory = os.getcwd()
        raise FileNotFoundError(
            f"The file `{file_name}` was not found in the current directory `{current_directory}`. Error: {e}"
        )


def is_valid_pair(pair: list[str], prefixes: tuple[str] = ENGLISH_PREFIXES) -> bool:
    return (
        len(pair[0].split(" ")) < MAX_LENGTH
        and len(pair[1].split(" ")) < MAX_LENGTH
        and pair[1].startswith(prefixes)
    )


def filter_pairs(pairs: list[list[str]]) -> list[list[str]]:
    return [pair for pair in pairs if is_valid_pair(pair)]
