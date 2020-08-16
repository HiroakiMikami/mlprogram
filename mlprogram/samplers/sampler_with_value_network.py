import torch
from typing \
    import TypeVar, Generic, Generator, Optional, List, Callable, Any
from mlprogram.samplers import SamplerState, Sampler
from mlprogram.utils.data import Collate
from mlprogram.utils import logging

logger = logging.Logger(__name__)

Input = TypeVar("Input")
Output = TypeVar("Output")
State = TypeVar("State")


class SamplerWithValueNetwork(Sampler[Input, Output, State],
                              Generic[Input, Output, State]):
    def __init__(self,
                 sampler: Sampler[Input, Output, State],
                 transform: Callable[[State], Any],
                 collate: Collate,
                 value_network: torch.nn.Module):
        self.sampler = sampler
        self.transform = transform
        self.collate = collate
        self.value_network = value_network

    def initialize(self, input: Input) -> State:
        return self.sampler.initialize(input)

    def create_output(self, state: State) \
            -> Optional[Output]:
        return self.sampler.create_output(state)

    def k_samples(self, states: List[SamplerState[State]], n: int) \
            -> Generator[SamplerState[State],
                         None, None]:
        self.value_network.eval()
        for state in self.sampler.k_samples(states, n):
            input = self.transform(state.state)
            with torch.no_grad(), logger.block("calculate_value"):
                value = self.value_network(self.collate([input]))
            yield SamplerState(value.item(), state.state)
