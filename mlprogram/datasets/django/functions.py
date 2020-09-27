import re
from nltk import tokenize
from typing import Dict, List
from mlprogram.languages import Token

tokenizer = tokenize.WhitespaceTokenizer()


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
        # Preprocess annotation
        def placeholder(id: int) -> str:
            return f"####{id}####"

        # Replace quoted string literals with placeholders
        mappings: Dict[str, str] = {}
        word_to_placeholder: Dict[str, str] = {}
        literal = r'\'\\\'\'|\"[^\"]+\"|\'[^\']+\'|`[^`]+`|"""[^"]+"""'
        while True:
            m = re.search(literal, query)
            if m is None:
                break

            w = m.group(0)[1:len(m.group(0)) - 1]
            if str(w) in word_to_placeholder:
                p = word_to_placeholder[str(w)]
            else:
                p = placeholder(len(mappings))
            query = query[:m.start()] + p + query[m.end():]

            assert (not ("####" in w))
            mappings[p] = str(w)
            word_to_placeholder[str(w)] = p

        reference = []
        for word in tokenizer.tokenize(query):
            if word in mappings:
                reference.append(Token[str, str](None, word, mappings[word]))
            else:
                reference.append(Token[str, str](None, word, word))

            vars = list(filter(lambda x: len(x) > 0,
                               word.split('.')))  # split by '.'
            if len(vars) > 1:
                for v in vars:
                    reference.append(Token(None, v, v))
        return reference


class TokenizeToken:
    def __init__(self, split_camel_case: bool = False):
        self.split_camel_case = split_camel_case

    def __call__(self, value: str) -> List[str]:
        if self.split_camel_case and re.search(
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
