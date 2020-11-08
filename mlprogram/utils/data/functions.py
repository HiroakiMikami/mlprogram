from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F

from mlprogram import Environment, logging
from mlprogram.actions import ActionSequence, ApplyRule, CloseVariadicFieldRule, Rule
from mlprogram.encoders import Samples
from mlprogram.languages import Analyzer, Parser, Token
from mlprogram.nn.utils import rnn
from mlprogram.utils.data.utils import ListDataset

logger = logging.Logger(__name__)


def get_words(dataset: torch.utils.data.Dataset,
              extract_reference: Callable[[Any], List[Token]],
              ) -> Sequence[str]:
    words = []

    for sample in dataset:
        reference = extract_reference(sample.inputs["text_query"])
        words.extend([token.value for token in reference])

    return words


def get_characters(dataset: torch.utils.data.Dataset,
                   extract_reference: Callable[[Any], List[Token]],
                   ) -> Sequence[str]:
    chars: List[str] = []

    for sample in dataset:
        reference = extract_reference(sample.inputs["text_query"])
        for token in reference:
            chars.extend(token.value)

    return chars


def get_samples(dataset: torch.utils.data.Dataset,
                parser: Parser[Any]) -> Samples:
    rules: List[Rule] = []
    node_types = []
    tokens: List[Tuple[str, str]] = []

    for sample in dataset:
        ground_truth = sample.supervisions["ground_truth"]
        ast = parser.parse(ground_truth)
        if ast is None:
            continue
        action_sequence = ActionSequence.create(ast)
        for action in action_sequence.action_sequence:
            if isinstance(action, ApplyRule):
                rule = action.rule
                if not isinstance(rule, CloseVariadicFieldRule):
                    rules.append(rule)
                    node_types.append(rule.parent)
                    for _, child in rule.children:
                        node_types.append(child)
            else:
                assert action.kind is not None
                tokens.append((action.kind, action.value))

    return Samples(rules, node_types, tokens)


@dataclass
class CollateOptions:
    use_pad_sequence: bool
    dim: int
    padding_value: float


class Collate:
    def __init__(self, device: torch.device,
                 **kwargs: CollateOptions):
        self.device = device
        self.options: Dict[str, CollateOptions] = kwargs

    def collate(self, tensors: Sequence[Optional[Environment]]) -> Environment:
        tmp: Dict[str, List[Any]] = {}
        for i, t in enumerate(tensors):
            if t is None:
                continue
            for name, item in t.to_dict().items():
                if name not in tmp:
                    tmp[name] = []
                tmp[name].append(item)

        retval = Environment()
        for name, values in tmp.items():
            if all([x is None for x in values]):
                retval[name] = None
                continue

            _, key = Environment.parse_key(name)
            if key not in self.options:
                retval[name] = values
                continue
            option = self.options[key]
            if option.use_pad_sequence:
                retval[name] = \
                    rnn.pad_sequence(values,
                                     padding_value=option.padding_value) \
                    .to(self.device)
            else:
                # pad tensors
                shape: List[int] = []
                for item in values:
                    if len(shape) == 0:
                        for x in item.shape:
                            shape.append(x)
                    else:
                        for i, x in enumerate(item.shape):
                            shape[i] = max(shape[i], x)
                padded_ts = []
                for item in values:
                    p = []
                    for dst, src in zip(shape, item.shape):
                        p.append(dst - src)
                        p.append(0)
                    p.reverse()
                    padded_ts.append(F.pad(item, p,
                                           value=option.padding_value))

                retval[name] = \
                    torch.stack(padded_ts, dim=option.dim).to(self.device)
        return retval

    def split(self, values: Environment) -> Sequence[Environment]:
        retval: List[Environment] = []
        B = None
        for name, t in values.to_dict().items():
            _, key = Environment.parse_key(name)
            if key in self.options:
                option = self.options[key]
                if option.use_pad_sequence:
                    B = t.data.shape[1]
                else:
                    B = t.data.shape[option.dim]
                break
            elif isinstance(t, list):
                B = len(t)
                break

        assert B is not None

        for _ in range(B):
            retval.append(Environment())

        for name, t in values.to_dict().items():
            _, key = Environment.parse_key(name)
            if key in self.options:
                option = self.options[key]
                if option.use_pad_sequence:
                    B = t.data.shape[1]
                else:
                    B = t.data.shape[option.dim]

                if option.use_pad_sequence:
                    for b in range(B):
                        inds = torch.nonzero(t.mask[:, b], as_tuple=False)
                        data = t.data[:, b]
                        shape = data.shape[1:]
                        data = data[inds]
                        data = data.reshape(-1, *shape)
                        retval[b][name] = data
                else:
                    shape = list(t.data.shape)
                    del shape[option.dim]
                    data = torch.split(t, 1, dim=option.dim)
                    for b in range(B):
                        d = data[b]
                        if len(shape) == 0:
                            d = d.reshape(())
                        else:
                            d = d.reshape(*shape)
                        retval[b][name] = d
            elif isinstance(t, list):
                for b in range(B):
                    retval[b][name] = t[b]
            else:
                logger.debug(f"{name} is invalid type: {type(t)}")

        return retval


def split_by_n_error(dataset: torch.utils.data.Dataset,
                     analyzer: Analyzer) -> Dict[str, torch.utils.data.Dataset]:
    no_error = []
    with_error = []
    for data in dataset:
        if len(analyzer(data.inputs["code"])) == 0:
            no_error.append(data)
        else:
            with_error.append(data)
    return {
        "no_error": ListDataset(no_error),
        "with_error": ListDataset(with_error)
    }
