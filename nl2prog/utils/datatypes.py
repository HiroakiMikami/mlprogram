from dataclasses import dataclass
from typing import List


@dataclass
class Query:
    query_for_synth: List[str]
    query_for_dnn: List[str]
