import re
from nltk import tokenize
from typing import List
from mlprogram.utils import Query

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

    query_for_synth = []
    for word in tokenizer.tokenize(query):
        query_for_synth.append(word)

        vars = list(filter(lambda x: len(x) > 0,
                           word.split('.')))  # split by '.'
        if len(vars) > 1:
            for v in vars:
                query_for_synth.append(v)
    return Query(query_for_synth, query_for_synth)


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
