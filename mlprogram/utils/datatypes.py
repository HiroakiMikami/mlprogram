from dataclasses import dataclass
from mlprogram.languages import Token
from typing import List


@dataclass
class Query:
    reference: List[Token[str, str]]
    query_for_dnn: List[str]
