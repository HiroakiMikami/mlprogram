"""
from typing import TypeVar
from typing import Generic

from mlprogram.interpreters import Interpreter


Input = TypeVar("Input")
Code = TypeVar("Code")
Value = TypeVar("Value")

class EvaluateCode(Generic[Input, Code, Value]):
    def __init__(self, interpreter: Interpreter[Input, Code, Value]):
        self.interpreter = interpreter

    def __call__(self, entry: Environment) -> Environment:
        code = entry.inputs["code"]
        input, _ = entry.inputs["input"]
        reference = entry.states["reference"]
        refs = [token.value for token in reference]
        result = self.interpreter.eval_references(code, input)
        variables = [result[ref] for ref in refs]
        # TODO variables may be in inputs
        entry.inputs["variables"] = variables
        return entry
"""
