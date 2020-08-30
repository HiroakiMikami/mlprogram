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
            n_initial_particle = self.initial_particle_size
        else:
            n_initial_particle = n_required_output
        initial_state = self.sampler.initialize(input)
        with logger.block("__call__"):
            i = 0
            while True:
                logger.debug(
                    f"start {i} th trial: n_particle={n_initial_particle}")
                i += 1
                # Initialize state
                n_particle = n_initial_particle
                particles = [DuplicatedSamplerState(
                    SamplerState(0.0, initial_state),
                    n_particle
                )]
                step = 0
                while step < self.max_step_size and n_particle > 0:
                    # Generate particles
                    samples: Dict[Key, DuplicatedSamplerState[State]] = {}
                    for sample in self.sampler.batch_k_samples(
                        [state.state for state in particles],
                        [state.num for state in particles]
                    ):
                        if sample.num == 0:
                            # This sample does not exist
                            continue
                        key = self.to_key(sample.state.state)
                        if key in samples:
                            state = samples[key]
                            samples[key] = \
                                DuplicatedSamplerState(
                                    state.state, state.num + sample.num
                            )
                        else:
                            samples[key] = sample

                    if len(samples) == 0:
                        # Output last particle with is_finished=True
                        for state in particles:
                            output_opt = \
                                self.sampler.create_output(input,
                                                           state.state.state)
                            if output_opt is not None:
                                output, _ = output_opt
                                yield Result(output, state.state.score,
                                             True, state.num)
                        break

                    # Resample
                    list_samples = [state for state in samples.values()]
                    log_weights = [math.log(state.num) + state.state.score
                                   for state in list_samples]
                    probs = [math.exp(log_weight - max(log_weights))
                             for log_weight in log_weights]
                    probs = [p / sum(probs) for p in probs]
                    particles = []
                    resampled = self.rng.multinomial(n_particle, probs)
                    for state, n in zip(list_samples, resampled):
                        # Create output
                        output_opt = \
                            self.sampler.create_output(input,
                                                       state.state.state)
                        if output_opt is not None:
                            output, is_finished = output_opt
                            if step == self.max_step_size - 1:
                                is_finished = True
                            yield Result(output, state.state.score,
                                         is_finished, n)
                        else:
                            is_finished = False

                        if is_finished:
                            # Exclude finished particles
                            n_particle -= n

                        if n > 0:
                            particles.append(
                                DuplicatedSamplerState(
                                    state.state, n
                                )
                            )
                    step += 1

                n_initial_particle *= self.factor
                if i == self.max_try_num:
                    break
