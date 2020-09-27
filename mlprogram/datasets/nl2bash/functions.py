import re
from nltk import tokenize
from typing import List
from mlprogram.languages import Token

tokenizer = tokenize.WhitespaceTokenizer()


def get_subtokens(token: str) -> List[Token[str, str]]:
    return list(map(lambda x: Token[str, str](None, x, x),
                    re.findall(r"[A-Za-z]+|\d+|\s+|.", token)))


class TokenizeQuery:
    def __call__(self, query: str) -> List[Token]:
        """
        Tokenize query

        Parameters
        ----------
        query: str

        Returns
        -------
        List[Token]
        """

        reference = []
        for word in tokenizer.tokenize(query):
            subtokens = get_subtokens(word)
            assert(word == "".join(map(lambda x: x.value, subtokens)))

            if len(subtokens) == 1:
                reference.append(Token[str, str](None, word, word))
            else:
                reference.append(Token[str, str](None, "SUB_START", ""))
                reference.extend(subtokens)
                reference.append(Token[str, str](None, "SUB_END", ""))
        return reference


class TokenizeToken:
    def __call__(self, value: str) -> List[str]:
        tokens = list(map(lambda x: x.value, get_subtokens(value)))
        assert(value == "".join(tokens))

        return tokens
