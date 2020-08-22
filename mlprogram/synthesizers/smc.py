from typing \
    import TypeVar, Generic, Optional, Generator, Dict, Callable, cast, Tuple
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
                while step < self.max_step_size:
                    # Generate particles
                    samples: Dict[Key, Tuple[DuplicatedSamplerState[State],
                                             Optional[Tuple[Output, bool]]
                                             ]] = {}
                    for sample in self.sampler.k_samples(
                        [state.state for state in particles],
                        [state.num for state in particles]
                    ):
                        key = self.to_key(sample.state.state)
                        if key in samples:
                            state, output_opt = samples[key]
                            samples[key] = (
                                DuplicatedSamplerState(
                                    state.state, state.num + sample.num
                                ),
                                output_opt
                            )
                        else:
                            samples[key] = \
                                (sample,
                                 self.sampler.create_output(
                                     input, sample.state.state))

                        _, output_opt = samples[key]
                        if output_opt is not None:
                            output, is_finished = output_opt
                            if step == self.max_step_size - 1:
                                # The step is last
                                is_finished = True
                            yield Result(output, sample.state.score,
                                         is_finished, sample.num)

                    # Exclude finished particles
                    for key, (state, output_opt) in list(samples.items()):
                        if output_opt is not None:
                            _, is_finished = output_opt
                            # Exclude key
                            if is_finished:
                                del samples[key]
                    n_particle = \
                        sum([state.num for state, _ in samples.values()])

                    if len(samples) == 0:
                        break
                    # Resample
                    list_samples = [state for state, _ in samples.values()]
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

                n_initial_particle *= self.factor
                if i == self.max_try_num:
                    break
