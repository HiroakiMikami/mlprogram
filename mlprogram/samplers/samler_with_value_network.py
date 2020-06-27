import torch
import logging
from typing \
    import TypeVar, Generic, Generator, Optional, Dict, Any
from mlprogram.samplers import SamplerState, Sampler
from mlprogram.decoders import Decoder

logger = logging.getLogger(__name__)

Input = TypeVar("Input")
Output = TypeVar("Output")


class SamplerWithValueNetwork(Sampler[Input, Output, Output],
                              Generic[Input, Output]):
    def __init__(self,
                 decoder: Decoder[Output, Output],
                 value_network: torch.nn.Module):
        self.decoder = decoder
        self.value_network = value_network

    def initialize(self, input: Input) \
            -> Output:
        raise NotImplementedError

    def create_output(self, state: Output) \
            -> Optional[Output]:
        return state

    def random_samples(self, state, n: int) \
            -> Generator[SamplerState[Output],
                         None, None]:
        cnt = 0
        self.value_network.eval()
        for output in self.decoder(state):
            value = self.value_network(output.output)
            yield SamplerState(value.item(), output.output)
            cnt += 1
            if cnt == n:
                break

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # TODO
        self.decoder.load_state_dict(state_dict["decoder"])
        self.value_network.load_state_dict(state_dict["value_network"])

    def to(self, device: torch.device) -> None:
        self.decoder.to(device)
        self.value_network.to(device)
