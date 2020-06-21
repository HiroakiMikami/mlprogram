import re
from nltk import tokenize
from typing import List
from mlprogram.utils import Query, Token

tokenizer = tokenize.WhitespaceTokenizer()


def tokenize_query(query: str) -> Query:
    """
    Tokenize annotation

    Parameters
    ----------
    query: str

    Returns
    -------
    Query
    """

    reference = []
    for word in tokenizer.tokenize(query):
        reference.append(Token[str](None, word))

        vars = list(filter(lambda x: len(x) > 0,
                           word.split('.')))  # split by '.'
        if len(vars) > 1:
            for v in vars:
                reference.append(Token[str](None, v))
    return Query(reference, list(map(lambda x: x.value, reference)))


def tokenize_token(value: str, split_camel_case: bool = False) -> List[str]:
    if split_camel_case and re.search(
            r"^[A-Z].*", value) and (" " not in value):
        # Camel Case
        words = re.findall(r"[A-Z][a-z]+", value)
        if "".join(words) == value:
            return words
        else:
            return [value]
    else:
        # Divide by space
        words = re.split(r"( +)", value)
        return [word for word in words if word != ""]
