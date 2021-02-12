import torch
from torch import nn
from torch import optim

from typing import Generator, Generic, Optional, TypeVar, List, Callable

from mlprogram import logging
from mlprogram.builtins import Environment
from mlprogram.synthesizers.synthesizer import Result, Synthesizer

logger = logging.Logger(__name__)

Input = TypeVar("Input")
Output = TypeVar("Output")


class REINFORCESynthesizer(Synthesizer[Environment, Output], Generic[Output]):
    def __init__(self, synthesizer: Synthesizer[Environment, Output],
                 model: nn.Module, optimizer: optim.Optimizer,
                 loss_fn: nn.Module, reward: nn.Module,
                 collate: Callable[[List[Environment]], Environment],
                 n_rollout: int,
                 device: torch.device,
                 baseline_momentum: float,
                 ):
        self.synthesizer = synthesizer
        self.model = model
        self.loss_fn = loss_fn
        self.reward = reward
        self.optimizer = optimizer
        self.collate = collate
        self.n_rollout = n_rollout
        self.device = device
        self.baseline_momentum = baseline_momentum
        self.model_state_dict = {
            key: value.clone() for key, value in self.model.state_dict().items()
        }
        # TODO clone state dict
        self.optimizer_state_dict = self.optimizer.state_dict()

    def __call__(self, input: Environment, n_required_output: Optional[int] = None) \
            -> Generator[Result[Output], None, None]:
        assert n_required_output is None

        baseline = 0

        # Reset model and optimizer
        self.model.load_state_dict(self.model_state_dict)
        self.optimizer.load_state_dict(self.optimizer_state_dict)

        idx = 0

        with logger.block("__call__"):
            to_rollout = input.clone_without_supervision()
            to_rollout.to(self.device)

            while True:
                rollouts = []
                reward = 0.0
                with logger.block("rollout"):
                    with torch.no_grad():
                        self.model.eval()
                        for rollout in logger.iterable_block(
                            "sample",
                            self.synthesizer(to_rollout,
                                             n_required_output=self.n_rollout)
                        ):
                            yield rollout
                            if not rollout.is_finished:
                                continue
                            for _ in range(rollout.num):
                                output = input.clone()
                                output["ground_truth"] = rollout.output
                                output.mark_as_supervision("ground_truth")
                                r = self.reward(input.clone(), rollout.output)
                                reward += r
                                output["reward"] = torch.tensor(r - baseline)
                                rollouts.append(output)

                with logger.block("calculate_baseline"):
                    reward = reward / len(rollouts)
                    m = self.baseline_momentum
                    baseline = (1 - m) * reward + m * baseline

                if idx % 1000 == 0:
                    logger.info(f"idx={idx} reward={baseline}")

                if len(rollouts) == 0:
                    logger.warning("No rollout")
                    continue
                if len(rollouts) != self.n_rollout:
                    logger.warning(
                        "#rollout is unexpected: "
                        f"expected={self.n_rollout} actual={len(rollouts)}")

                with logger.block("train"):
                    self.model.train()
                    with logger.block("collate"):
                        batch = self.collate(rollouts)
                    with logger.block("to"):
                        batch.to(self.device)
                    with logger.block("forward"):
                        self.model.train()
                        output = self.model(batch)
                        loss = self.loss_fn(output)
                    with logger.block("backward"):
                        self.model.zero_grad()
                        loss.backward()
                    with logger.block("optimizer.step"):
                        self.optimizer.step()

                idx += 1
