from typing import Dict, Any
from mlprogram.action.ast import AST

Environment = Dict[Any, Any]


class Interpreter:
    def eval(self, env: Environment, code: AST) -> Environment:
        raise NotImplementedError
