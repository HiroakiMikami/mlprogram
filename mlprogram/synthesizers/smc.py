import torch
from typing import TypeVar, Generic, Optional, Generator, Dict, Tuple, Any
from mlprogram.samplers import Sampler, SamplerState
from mlprogram.synthesizers import Result, Synthesizer
import math
import numpy as np

Input = TypeVar("Input")
Output = TypeVar("Output")
State = TypeVar("State")


class SMC(Synthesizer[Input, Output], Generic[Input, Output, State]):
    def __init__(self, max_step_size: int, max_retry_num: int,
                 initial_particle_size: int,
                 sampler: Sampler[Input, Output, State],
                 factor: int = 2,
                 rng: Optional[np.random.RandomState] = None):
        self.max_step_size = max_step_size
        self.max_retry_num = max_retry_num
        self.initial_particle_size = initial_particle_size
        self.factor = factor
        self.sampler = sampler
        self.rng = rng or np.random

    def __call__(self, input: Input) -> Generator[Result[Output], None, None]:
        num_particle = self.initial_particle_size
        initial_state = SamplerState(0.0, self.sampler.initialize(input))
        for _ in range(self.max_retry_num):
            # Initialize state
            particles = [(initial_state, num_particle)]
            step = 0
            while step < self.max_step_size:
                # Generate particles
                samples: Dict[State, Tuple[float, int]] = {}
                for state, n in particles:
                    for sample in self.sampler.k_samples([state], n):
                        if sample.state in samples:
                            samples[sample.state] = \
                                (samples[sample.state][0],
                                 samples[sample.state][1] + 1)
                        else:
                            samples[sample.state] = (sample.score, 1)

                        output_opt = self.sampler.create_output(sample.state)
                        if output_opt is not None:
                            yield Result(output_opt, sample.score)

                # Resample
                list_samples = [(SamplerState(score, state), n)
                                for state, (score, n) in samples.items()]
                log_weights = [math.log(n) + state.score
                               for state, n in list_samples]
                probs = [math.exp(log_weight - max(log_weights))
                         for log_weight in log_weights]
                probs = [p / sum(probs) for p in probs]
                particles = []
                resampled = self.rng.multinomial(num_particle, probs)
                for (state, _), n in zip(list_samples, resampled):
                    if n > 0:
                        particles.append((state, n))

                step += 1

            num_particle *= self.factor

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.sampler.load_state_dict(state_dict)

    def to(self, device: torch.device) -> None:
        self.sampler.to(device)
