from typing import List, Optional

from mlprogram.languages import Token


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


class SplitValue:
    def __call__(self, token: str) -> List[str]:
        return [token]
