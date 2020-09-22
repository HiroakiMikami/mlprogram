import numpy as np
import torch
from torch import nn
from typing \
    import TypeVar, Generic, Generator, Optional, Dict, Any, Callable, \
    List, Set, Tuple
from mlprogram.languages.ast import AST, Node, Leaf
from mlprogram.samplers import SamplerState, DuplicatedSamplerState, Sampler
from mlprogram.interpreters import Reference
from mlprogram.interpreters import Statement
from mlprogram.interpreters import SequentialProgram
from mlprogram.synthesizers import Synthesizer
from mlprogram.utils import Token
from mlprogram.utils.data import Collate
from mlprogram import logging

logger = logging.Logger(__name__)

Input = TypeVar("Input")
Code = TypeVar("Code")


class AstReferenceSampler(Sampler[Input, SequentialProgram[Code],
                                  Dict[str, Any]],
                          Generic[Input, Code]):
    def __init__(self,
                 synthesizer: Synthesizer[Dict[str, Any], AST],
                 transform_input: Callable[[Input], Dict[str, Any]],
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
    def initialize(self, input: Input) -> Dict[str, Any]:
        self.encoder.eval()
        state_list = self.transform_input(input)
        state_tensor = self.collate.collate([state_list])
        with torch.no_grad(), logger.block("encode_state"):
            state_tensor = self.encoder(state_tensor)
        state = self.collate.split(state_tensor)[0]
        state["reference"] = []
        state["unused_reference"] = []
        state["code"] = SequentialProgram([])
        return state

    def create_output(self, input, state: Dict[str, Any]) \
            -> Optional[Tuple[SequentialProgram[Code], bool]]:
        if len(state["code"].statements) == 0:
            return None
        output = state["code"].statements[-1].reference
        unused_reference = set([ref.value for ref in state["unused_reference"]
                                if ref.value != output])
        code = SequentialProgram[Code](
            [statement for statement in state["code"].statements
             if statement.reference not in unused_reference])
        return code, False

    def batch_k_samples(self, states: List[SamplerState[Dict[str, Any]]],
                        ks: List[int]) \
            -> Generator[DuplicatedSamplerState[Dict[str, Any]],
                         None, None]:
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

            for state, k in zip(states, ks):
                cnt = 0
                for result in self.synthesizer(state.state,
                                               n_required_output=k):
                    if cnt == k:
                        break

                    new_state = {key: value for key,
                                 value in state.state.items()}
                    # Copy reference
                    new_state["reference"] = list(new_state["reference"])
                    new_state["unused_reference"] = \
                        list(new_state["unused_reference"])
                    new_code = list(new_state["code"].statements)
                    n_var = len(new_state["code"].statements)
                    ref = Reference(f"v{n_var}")
                    code = self.to_code(result.output)
                    if code is None:
                        continue
                    type_name = result.output.get_type_name()
                    if type_name is not None:
                        type_name = str(type_name)
                    vars = find_variables(result.output)
                    new_state["unused_reference"] = \
                        [token for token in new_state["unused_reference"]
                            if token.value not in vars]
                    if self.remove_used_variable:
                        new_state["reference"] = \
                            [ref for ref in new_state["unused_reference"]]
                    new_state["reference"].append(
                        Token(type_name, ref))
                    new_state["unused_reference"].append(
                        Token(type_name, ref))
                    new_code.append(Statement(ref, code))
                    new_state["code"] = SequentialProgram(new_code)
                    yield DuplicatedSamplerState(
                        SamplerState(result.score, new_state), result.num)
                    cnt += 1
