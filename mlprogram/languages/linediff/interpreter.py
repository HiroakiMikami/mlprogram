from mlprogram.languages import BatchedState
from mlprogram.languages import Interpreter as BaseInterpreter
from mlprogram.languages.linediff import AST
from mlprogram.languages.linediff import Delta
from mlprogram.languages.linediff import Insert
from mlprogram.languages.linediff import Remove
from mlprogram.languages.linediff import Replace
from mlprogram.languages.linediff import Diff
from typing import List
from typing import Tuple
from functools import lru_cache


class Interpreter(BaseInterpreter[AST, str, str, str]):
    def eval(self, code: AST, inputs: List[str]) -> List[str]:
        if isinstance(code, Delta):
            code = Diff([code])
        assert isinstance(code, Diff)
        return self._eval(code, inputs)

    def execute(self, code: AST, inputs: List[str],
                state: BatchedState[AST, str, str]) \
            -> BatchedState[AST, str, str]:
        assert len(state.environment) <= 1
        if len(state.environment) == 1:
            inputs = state.environment[state.history[-1]]
        outputs = self.eval(code, inputs)
        next = state.clone()
        next.history.append(code)
        next.type_environment[code] = code.get_type_name()
        next.environment = {code: outputs}
        return next

    def _eval(self, diff: Diff, inputs: List[str]) -> List[str]:
        for delta in diff.deltas:
            inputs = self._apply(delta, tuple(inputs))
        return inputs

    @lru_cache(maxsize=1000)
    def _apply(self, delta: Delta, inputs: Tuple[str]) -> List[str]:
        line_inputs = [list(input.split("\n")) for input in inputs]
        if isinstance(delta, Insert):
            for line_input in line_inputs:
                line_input.insert(delta.line_number, delta.value)
        elif isinstance(delta, Remove):
            for line_input in line_inputs:
                del line_input[delta.line_number]
        elif isinstance(delta, Replace):
            for line_input in line_inputs:
                line_input[delta.line_number] = delta.value
        else:
            raise AssertionError(f"invalid type: {type(delta)}")
        return ["\n".join(line_input) for line_input in line_inputs]
