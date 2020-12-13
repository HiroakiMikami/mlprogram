from collections import OrderedDict
from typing import Callable, Generic, TypeVar, Optional

from torch import nn

from mlprogram.builtins import Apply, Environment, Pick
from mlprogram.functools import Sequence
from mlprogram.nn import Function

Value = TypeVar("Value")


class Metric(nn.Module, Generic[Value]):
    def __init__(self, metric: Callable, in_keys, value_key: str,
                 transform: Optional[Callable[[float], float]] = None):
        super().__init__()
        self.value_key = value_key
        self.metric = Sequence(OrderedDict([
            ("metric", Apply(
                module=Function(metric),
                in_keys=in_keys,
                out_key="metric",
            )),
            ("pick", Pick("metric"))
        ]))
        self.transform = transform

    def forward(self, env: Environment, value: Value) -> float:
        env[self.value_key] = value
        out = self.metric(env)
        if self.transform is not None:
            out = self.transform(out)
        return out


def use_environment(metric: Callable, in_keys, value_key: str,
                    transform: Optional[Callable[[float], float]] = None):
    return Metric(metric, in_keys, value_key, transform)
