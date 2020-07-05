import numpy as np
import torch
from torch import nn
import logging
from typing \
    import TypeVar, Generic, Generator, Optional, Dict, Any, Callable, \
    List, Set
from mlprogram.asts import AST, Node, Leaf
from mlprogram.samplers import SamplerState, Sampler
from mlprogram.synthesizers import Synthesizer
from mlprogram.utils import Token
from mlprogram.utils.torch import StateDict
from mlprogram.utils.data import Collate
from dataclasses import dataclass


logger = logging.getLogger(__name__)

Input = TypeVar("Input")


# TODO move to other file
@dataclass
class Reference:
    name: str

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Reference):
            return self.name == other.name
        else:
            return False


class AstSetSampler(Sampler[Input, Dict[Reference, AST], Dict[str, Any]],
                    Generic[Input]):
    def __init__(self,
                 synthesizer: Synthesizer[Dict[str, Any], AST],
                 transform_input: Callable[[Input], Dict[str, Any]],
                 collate: Collate,
                 encoder: nn.Module,
                 remove_used_variable: bool = True,
                 rng: Optional[np.random.RandomState] = None):
        self.synthesizer = synthesizer
        self.transform_input = transform_input
        self.collate = collate
        self.encoder = encoder
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
        state["mapping"] = {}
        return state

    def create_output(self, state: Dict[str, Any]) \
            -> Optional[Dict[Reference, AST]]:
        return state["mapping"]

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

        sampler_states = []
        for state in states:
            for i, result in enumerate(self.synthesizer(state.state)):
                new_state = {key: value for key, value in state.state.items()}
                # Copy reference
                new_state["reference"] = list(new_state["reference"])
                new_state["mapping"] = dict(new_state["mapping"].items())
                n_var = len(new_state["mapping"])
                ref = Reference(f"v{n_var}")
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
                new_state["mapping"][ref] = result.output
                sampler_states.append(SamplerState(state.score + result.score,
                                                   new_state))
                if i == n:
                    break

        if len(sampler_states) <= n:
            for s in sampler_states:
                yield s
        else:
            # log_prob -> prob
            probs = [np.exp(s.score) for s in sampler_states]
            # normalize
            probs = [p / sum(probs) for p in probs]
            resamples = self.rng.multinomial(n, probs)
            for s, m in zip(sampler_states, resamples):
                for _ in range(m):
                    yield s

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        s = StateDict(state_dict)
        self.synthesizer.load_state_dict(s["synthesizer"])
        self.encoder.load_state_dict(s["encoder"])

    def to(self, device: torch.device) -> None:
        self.synthesizer.to(device)
        self.encoder.to(device)
