from typing import Any, Tuple, List, Generator, Dict, Optional
from mlprogram.action import Action
from mlprogram.ast import AST
from dataclasses import dataclass


@dataclass
class Progress:
    id: int
    parent: Optional[int]
    score: float
    action: Action
    is_complete: bool


@dataclass
class Candidate:
    score: float
    ast: AST


class Synthesizer:
    def synthesize(self, input: Any) \
            -> Generator[Tuple[List[Candidate], List[Progress]], None, None]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError
