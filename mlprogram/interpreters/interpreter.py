from typing import Generic, TypeVar, Dict, List, Tuple
from mlprogram.utils import Reference


Code = TypeVar("Code")
Value = TypeVar("Value")


class Interpreter(Generic[Code, Value]):
    def eval(self, code: Code) -> Value:
        raise NotImplementedError

    def eval_references(self, code: List[Tuple[Reference, Code]]) \
            -> Dict[Reference, Value]:
        raise NotImplementedError
