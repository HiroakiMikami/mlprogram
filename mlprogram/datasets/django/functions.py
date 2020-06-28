import re
from nltk import tokenize
from typing import Dict
from mlprogram.utils import Query, Token

tokenizer = tokenize.WhitespaceTokenizer()


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
        query_for_dnn = []
        for word in tokenizer.tokenize(query):
            if word in mappings:
                reference.append(Token[str](None, mappings[word]))
            else:
                reference.append(Token[str](None, word))
            query_for_dnn.append(word)

            vars = list(filter(lambda x: len(x) > 0,
                               word.split('.')))  # split by '.'
            if len(vars) > 1:
                for v in vars:
                    reference.append(Token(None, v))
                    query_for_dnn.append(v)
        return Query(reference, query_for_dnn)
