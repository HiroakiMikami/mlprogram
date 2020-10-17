import numpy as np
import torch
from torch import nn
from typing \
    import TypeVar, Generic, Generator, Optional, Callable, \
    List, Set, Tuple
from mlprogram import Environment
from mlprogram.languages import AST, Node, Leaf
from mlprogram.languages import Token
from mlprogram.samplers import SamplerState, DuplicatedSamplerState, Sampler
from mlprogram.interpreters import Reference
from mlprogram.interpreters import Statement
from mlprogram.interpreters import SequentialProgram
from mlprogram.synthesizers import Synthesizer
from mlprogram.utils.data import Collate
from mlprogram import logging

logger = logging.Logger(__name__)

Input = TypeVar("Input")
Code = TypeVar("Code")


class SequentialProgramSampler(Sampler[Input, SequentialProgram[Code],
                                       Environment],
                               Generic[Input, Code]):
    def __init__(self,
                 synthesizer: Synthesizer[Environment, AST],
                 transform_input: Callable[[Input], Environment],
                 collate: Collate,
                 encoder: nn.Module,
                 to_code: Callable[[AST], Optional[Code]],
                 remove_used_variable: bool = True,
                 rng: Optional[np.random.RandomState] = None):
        self.synthesizer = synthesizer
        self.transform_input = transform_input
        self.collate = collate
        self.encoder = encoder
        self.to_code = to_code
        self.remove_used_variable = remove_used_variable
        self.rng = \
            rng or np.random.RandomState(np.random.randint(0, 2 << 32 - 1))

    @logger.function_block("initialize")
    def initialize(self, input: Input) -> Environment:
        self.encoder.eval()
        state_list = self.transform_input(input)
        state_tensor = self.collate.collate([state_list])
        with torch.no_grad(), logger.block("encode_state"):
            state_tensor = self.encoder(state_tensor)
        state = self.collate.split(state_tensor)[0]
        state.states["reference"] = []
        state.inputs["unused_reference"] = []
        state.inputs["code"] = SequentialProgram([])
        return state

    def create_output(self, input, state: Environment) \
            -> Optional[Tuple[SequentialProgram[Code], bool]]:
        if len(state.inputs["code"].statements) == 0:
            return None
        output = state.inputs["code"].statements[-1].reference
        unused_reference = set([ref.value
                                for ref in state.inputs["unused_reference"]
                                if ref.value != output])
        code = SequentialProgram[Code](
            [statement for statement in state.inputs["code"].statements
             if statement.reference not in unused_reference])
        return code, False

    def batch_k_samples(self, states: List[SamplerState[Environment]],
                        ks: List[int]) \
            -> Generator[DuplicatedSamplerState[Environment],
                         None, None]:
        assert all([len(state.state.outputs) for state in states]) == 0
        assert all([len(state.state.supervisions) for state in states]) == 0

        orig_inputs = [state.state.inputs.to_dict() for state in states]
        orig_states = [state.state.states.to_dict() for state in states]
        with logger.block("batch_k_samples"):
            def find_variables(node: AST) -> Set[Reference]:
                retval: Set[Reference] = set()
                if isinstance(node, Node):
                    for field in node.fields:
                        if isinstance(field.value, list):
                            for v in field.value:
                                retval = retval | find_variables(v)
                        else:
                            retval = retval | find_variables(field.value)
                elif isinstance(node, Leaf):
                    if isinstance(node.value, Reference):
                        retval.add(node.value)
                return retval

            for orig_input, orig_state, state, k in zip(orig_inputs,
                                                        orig_states, states,
                                                        ks):
                cnt = 0
                for result in self.synthesizer(state.state,
                                               n_required_output=k):
                    if cnt == k:
                        break

                    # TODO recompute state (?)
                    new_state = Environment(inputs=orig_input,
                                            states=orig_state)
                    # Copy reference
                    new_state.states["reference"] = \
                        list(state.state.states["reference"])
                    new_state.inputs["unused_reference"] = \
                        list(state.state.inputs["unused_reference"])
                    new_code = list(state.state.inputs["code"].statements)
                    n_var = len(state.state.inputs["code"].statements)
                    ref = Reference(f"v{n_var}")
                    code = self.to_code(result.output)
                    if code is None:
                        continue
                    type_name = result.output.get_type_name()
                    if type_name is not None:
                        type_name = str(type_name)
                    vars = find_variables(result.output)
                    new_state.inputs["unused_reference"] = \
                        [token
                         for token in new_state.inputs["unused_reference"]
                         if token.value not in vars]
                    if self.remove_used_variable:
                        new_state.states["reference"] = [
                            ref
                            for ref in new_state.inputs["unused_reference"]]
                    new_state.states["reference"].append(
                        Token(type_name, ref, ref)
                    )
                    new_state.inputs["unused_reference"].append(
                        Token(type_name, ref, ref)
                    )
                    new_code.append(Statement(ref, code))
                    new_state.inputs["code"] = SequentialProgram(new_code)
                    yield DuplicatedSamplerState(
                        SamplerState(result.score, new_state), result.num)
                    cnt += 1
