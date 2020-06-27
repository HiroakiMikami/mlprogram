import re
from nltk import tokenize
from typing import List

tokenizer = tokenize.WhitespaceTokenizer()


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
