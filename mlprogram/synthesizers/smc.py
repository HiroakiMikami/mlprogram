from typing \
    import TypeVar, Generic, Optional, Generator, Dict, Callable, cast
from mlprogram.samplers import Sampler, SamplerState, DuplicatedSamplerState
from mlprogram.synthesizers import Result, Synthesizer
from mlprogram.utils import logging
import math
import numpy as np

logger = logging.Logger(__name__)

Input = TypeVar("Input")
Output = TypeVar("Output")
State = TypeVar("State")
Key = TypeVar("Key")


class SMC(Synthesizer[Input, Output], Generic[Input, Output, State, Key]):
    def __init__(self, max_step_size: int,
                 initial_particle_size: int,
                 sampler: Sampler[Input, Output, State],
                 to_key: Callable[[State], Key] = lambda x: cast(Key, x),
                 max_try_num: Optional[int] = None,
                 factor: int = 2,
                 rng: Optional[np.random.RandomState] = None):
        self.max_step_size = max_step_size
        self.max_try_num = max_try_num
        self.initial_particle_size = initial_particle_size
        self.to_key = to_key
        self.factor = factor
        self.sampler = sampler
        self.rng = \
            rng or np.random.RandomState(np.random.randint(0, 2 << 32 - 1))

    def __call__(self, input: Input, n_required_output: Optional[int] = None) \
            -> Generator[Result[Output], None, None]:
        if n_required_output is None:
            n_particle = self.initial_particle_size
        else:
            n_particle = n_required_output
        initial_state = self.sampler.initialize(input)
        i = 0
        with logger.block("__call__"):
            while True:
                logger.debug(f"start {i} th trial: n_particle={n_particle}")
                i += 1
                # Initialize state
                particles = [DuplicatedSamplerState(
                    SamplerState(0.0, initial_state),
                    n_particle
                )]
                step = 0
                while step < self.max_step_size:
                    # Generate particles
                    samples: Dict[Key, DuplicatedSamplerState[State]] = {}
                    for sample in self.sampler.k_samples(
                        [state.state for state in particles],
                        [state.num for state in particles]
                    ):
                        key = self.to_key(sample.state.state)
                        if key in samples:
                            samples[key] = DuplicatedSamplerState(
                                samples[key].state,
                                samples[key].num + sample.num
                            )
                        else:
                            samples[key] = sample

                            output_opt = \
                                self.sampler.create_output(sample.state.state)
                            if output_opt is not None:
                                yield Result(output_opt, sample.state.score,
                                             sample.num)

                    if len(samples) == 0:
                        break
                    # Resample
                    list_samples = list(samples.values())
                    log_weights = [math.log(state.num) + state.state.score
                                   for state in list_samples]
                    probs = [math.exp(log_weight - max(log_weights))
                             for log_weight in log_weights]
                    probs = [p / sum(probs) for p in probs]
                    particles = []
                    resampled = self.rng.multinomial(n_particle, probs)
                    for state, n in zip(list_samples, resampled):
                        if n > 0:
                            particles.append(
                                DuplicatedSamplerState(
                                    state.state, n
                                )
                            )
                    step += 1

                n_particle *= self.factor
                if i == self.max_try_num:
                    break
