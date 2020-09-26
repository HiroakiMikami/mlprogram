from mlprogram.utils import Query
from mlprogram.languages import Token
from typing import Optional, List


class TokenizeQuery:
    def __call__(self, query: str) -> Query:
        """
        Tokenize query

        Parameters
        ----------
        query: str

        Returns
        -------
        Query
        """
        words = query.split(" ")
        reference = []

        value: Optional[str] = None
        for word in words:
            if word.endswith("_END"):
                if value is not None:
                    reference.append(Token[str, str](None, value, value))
                reference.append(Token[str, str](None, word, word))
                value = None
            else:
                if value is None:
                    value = word
                else:
                    value += f" {word}"
        if value is not None:
            reference.append(Token[str, str](None, value, value))
        return Query(reference, list(map(lambda x: x.value, reference)))


class TokenizeToken:
    def __call__(self, token: str) -> List[str]:
        return [token]
