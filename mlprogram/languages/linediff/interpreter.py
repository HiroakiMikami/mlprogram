from functools import lru_cache
from typing import List, Tuple, cast

from mlprogram import logging
from mlprogram.languages import BatchedState
from mlprogram.languages import Interpreter as BaseInterpreter
from mlprogram.languages.linediff import AST, Delta, Diff, Insert, Remove, Replace

logger = logging.Logger(__name__)


class Interpreter(BaseInterpreter[AST, str, str, str, str]):
    def eval(self, code: AST, inputs: List[str]) -> List[str]:
        if isinstance(code, Delta):
            code = Diff([code])
        assert isinstance(code, Diff)
        return self._eval(code, inputs)

    def create_state(self, inputs: List[str]) -> BatchedState[AST, str, str, str]:
        return BatchedState(
            type_environment={},
            environment={},
            history=[],
            context=inputs,
        )

    def execute(self, code: AST, state: BatchedState[AST, str, str, str]) \
            -> BatchedState[AST, str, str, str]:
        inputs = state.context
        outputs = self.eval(code, inputs)
        next = cast(BatchedState[AST, str, str, str], state.clone())
        next.history.append(code)
        next.type_environment[code] = code.get_type_name()
        next.environment = {code: outputs}
        next.context = outputs
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
                if delta.line_number < len(line_input):
                    line_input[delta.line_number] = delta.value
                else:
                    logger.warning(f"Input has only {len(line_input)} lines, "
                                   f"{delta} cannot be applied")
        else:
            raise AssertionError(f"invalid type: {type(delta)}")
        return ["\n".join(line_input) for line_input in line_inputs]
