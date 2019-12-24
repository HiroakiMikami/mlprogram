import re
from nltk import tokenize
from typing import List
from nl2prog.utils.data.nl2code import Query

tokenizer = tokenize.WhitespaceTokenizer()


def get_subtokens(token: str) -> List[str]:
    return list(re.findall(r"[A-Za-z]+|\d+|\s+|.", token))


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

    query_for_synth = []
    for word in tokenizer.tokenize(query):
        subtokens = get_subtokens(word)
        assert(word == "".join(subtokens))

        if len(subtokens) == 1:
            query_for_synth.append(word)
        else:
            query_for_synth.append("SUB_START")
            query_for_synth.extend(subtokens)
            query_for_synth.append("SUB_END")
    return Query(query_for_synth, query_for_synth)


def tokenize_token(value: str) -> List[str]:
    tokens = []
    tokens = get_subtokens(value)
    assert(value == "".join(tokens))

    return tokens
