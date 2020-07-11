import torch
import logging
from typing \
    import TypeVar, Generic, Generator, Optional, List
from mlprogram.samplers import SamplerState, Sampler

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

    def k_samples(self, states: List[SamplerState[State]], n: int) \
            -> Generator[SamplerState[State],
                         None, None]:
        self.value_network.eval()
        for state in self.sampler.k_samples(states, n):
            # TODO batch computation
            value = self.value_network(state.state)
            yield SamplerState(value.item(), state.state)
