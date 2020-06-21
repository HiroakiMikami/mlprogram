import re
from nltk import tokenize
from typing import List
from mlprogram.utils import Query, Token

tokenizer = tokenize.WhitespaceTokenizer()


def get_subtokens(token: str) -> List[Token[str]]:
    return list(map(lambda x: Token[str](None, x),
                    re.findall(r"[A-Za-z]+|\d+|\s+|.", token)))


def tokenize_query(query: str) -> Query:
    """
    Tokenize query

    Parameters
    ----------
    query: str

    Returns
    -------
    Query
    """

    reference = []
    for word in tokenizer.tokenize(query):
        subtokens = get_subtokens(word)
        assert(word == "".join(map(lambda x: x.value, subtokens)))

        if len(subtokens) == 1:
            reference.append(Token[str](None, word))
        else:
            reference.append(Token[str](None, "SUB_START"))
            reference.extend(subtokens)
            reference.append(Token[str](None, "SUB_END"))
    return Query(reference, list(map(lambda x: x.value, reference)))


def tokenize_token(value: str) -> List[str]:
    tokens = list(map(lambda x: x.value, get_subtokens(value)))
    assert(value == "".join(tokens))

    return tokens
