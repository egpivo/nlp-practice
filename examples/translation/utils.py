import unicodedata
import re
from io import open


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
    return u"".join(
        char for char in unicodedata.normalize("NFKD", string)
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
    removed_punctuations = re.sub(r"[^a-zA-Z.!?]+", r" ",target_punctuations)
    return removed_punctuations


def read_file(language1: str ="eng", language2: str = "fra") -> list[str]:
    """
    Examples
    --------
    >>> read_file()[0]
    'Go.\tVa !'
    """
    with open(f"./data/translation/{language1}-{language2}.txt", encoding="utf-8") as file:
        return file.read().strip().split("\n")
