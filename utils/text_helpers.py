import re
import string
import unicodedata

def normalize_answer(s: str) -> str:
    """
    Standard SQuAD normalization: lower case, remove punctuation and articles.
    """
    # Normalize Unicode characters (e.g., accents)
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))