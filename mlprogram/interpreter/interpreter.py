from typing import Dict, Any

Environment = Dict[Any, Any]


class Interpreter:
    def eval(self, env: Environment, code: Any) -> Environment:
        raise NotImplementedError
