import torch
import logging
from typing \
    import TypeVar, Generic, Generator, Optional, Dict, Any
from mlprogram.samplers import SamplerState, Sampler
from mlprogram.utils.torch import StateDict

logger = logging.getLogger(__name__)

Input = TypeVar("Input")
Output = TypeVar("Output")
State = TypeVar("State")


class SamplerWithValueNetwork(Sampler[Input, Output, State],
                              Generic[Input, Output, State]):
    def __init__(self,
                 sampler: Sampler[Input, Output, State],
                 value_network: torch.nn.Module):
        self.sampler = sampler
        self.value_network = value_network

    def initialize(self, input: Input) -> State:
        return self.sampler.initialize(input)

    def create_output(self, state: State) \
            -> Optional[Output]:
        return self.sampler.create_output(state)

    def random_samples(self, state: SamplerState[State], n: int) \
            -> Generator[SamplerState[State],
                         None, None]:
        self.value_network.eval()
        for state in self.sampler.random_samples(state, n):
            # TODO batch computation
            value = self.value_network(state.state)
            yield SamplerState(value.item(), state.state)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        s = StateDict(state_dict)
        self.sampler.load_state_dict(s["sampler"])
        self.value_network.load_state_dict(s["value_network"])

    def to(self, device: torch.device) -> None:
        self.sampler.to(device)
        self.value_network.to(device)
