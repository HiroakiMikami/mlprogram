from mlprogram.languages import Token
from typing import Optional, List


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
        return reference


class SplitToken:
    def __call__(self, token: Token) -> List[Token]:
        return [token]
