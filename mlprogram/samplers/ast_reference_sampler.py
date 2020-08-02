import numpy as np
from torch import nn
import logging
from typing \
    import TypeVar, Generic, Generator, Optional, Dict, Any, Callable, \
    List, Set, Tuple
from mlprogram.asts import AST, Node, Leaf
from mlprogram.samplers import SamplerState, Sampler
from mlprogram.synthesizers import Synthesizer
from mlprogram.utils import Token, Reference
from mlprogram.utils.data import Collate
from mlprogram.utils import random

logger = logging.getLogger(__name__)

Input = TypeVar("Input")
Code = TypeVar("Code")


class AstReferenceSampler(Sampler[Input, List[Tuple[Reference, Code]],
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
        if rng is None:
            self.rng = np.random
        else:
            self.rng = rng

    def initialize(self, input: Input) -> Dict[str, Any]:
        self.encoder.eval()
        state_list = self.transform_input(input)
        state_tensor = self.collate.collate([state_list])
        state_tensor = self.encoder(state_tensor)
        state = self.collate.split(state_tensor)[0]
        state["reference"] = []
        state["code"] = []
        return state

    def create_output(self, state: Dict[str, Any]) \
            -> Optional[List[Tuple[Reference, Code]]]:
        return state["code"]

    def k_samples(self, states: List[SamplerState[Dict[str, Any]]], n: int) \
            -> Generator[SamplerState[Dict[str, Any]],
                         None, None]:
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

        ks = random.split(self.rng, n, len(states), 1e-8)
        for state, k in zip(states, ks):
            for i, result in enumerate(self.synthesizer(state.state)):
                new_state = {key: value for key, value in state.state.items()}
                # Copy reference
                new_state["reference"] = list(new_state["reference"])
                new_state["code"] = list(new_state["code"])
                n_var = len(new_state["code"])
                ref = Reference(f"v{n_var}")
                code = self.to_code(result.output)
                if code is None:
                    continue
                type_name = result.output.get_type_name()
                if type_name is not None:
                    type_name = str(type_name)
                if self.remove_used_variable:
                    vars = find_variables(result.output)
                    new_state["reference"] = \
                        [token for token in new_state["reference"]
                         if token.value not in vars]
                new_state["reference"].append(
                    Token(type_name, ref))
                new_state["code"].append((ref, code))
                yield SamplerState(result.score, new_state)
