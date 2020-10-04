import torch
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Callable, Sequence, Any, Optional, Union, Dict, List
from typing import Tuple
from mlprogram.actions \
    import Rule, ApplyRule, CloseVariadicFieldRule
from mlprogram.languages import Parser
from mlprogram.actions import ActionSequence
from mlprogram.encoders import Samples
from mlprogram.languages import Token
from mlprogram.nn.utils import rnn
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


def get_words(dataset: torch.utils.data.Dataset,
              extract_reference: Callable[[Any], List[Token]],
              ) -> Sequence[str]:
    words = []

    for group in dataset:
        for input in group["input"]:
            reference = extract_reference(input)
            words.extend([token.value for token in reference])

    return words


def get_characters(dataset: torch.utils.data.Dataset,
                   extract_reference: Callable[[Any], List[Token]],
                   ) -> Sequence[str]:
    chars: List[str] = []

    for group in dataset:
        for input in group["input"]:
            reference = extract_reference(input)
            for token in reference:
                chars.extend(token.value)

    return chars


def get_samples(dataset: torch.utils.data.Dataset,
                parser: Parser[Any]) -> Samples:
    rules: List[Rule] = []
    node_types = []
    tokens: List[Tuple[str, str]] = []

    for sample in dataset:
        ground_truth = sample["ground_truth"]
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

    def __call__(self, tensors: Sequence[Optional[Dict[str, Any]]]) \
            -> Dict[str, Union[torch.Tensor, PaddedSequenceWithMask]]:
        return self.collate(tensors)

    def collate(self, tensors: Sequence[Optional[Dict[str, Any]]]) \
            -> Dict[str, Any]:
        retval: Dict[str, Any] = {}
        tmp: Dict[str, List[torch.Tensor]] = {}
        for i, t in enumerate(tensors):
            if t is None:
                continue
            for name, tensor in t.items():
                if name not in tmp:
                    tmp[name] = []
                tmp[name].append(tensor)
        for name, ts in tmp.items():
            if name not in self.options:
                retval[name] = ts
                continue
            option = self.options[name]
            if all(map(lambda x: x is None, ts)):
                retval[name] = None
            elif option.use_pad_sequence:
                retval[name] = \
                    rnn.pad_sequence(ts,
                                     padding_value=option.padding_value) \
                    .to(self.device)
            else:
                # pad tensors
                shape: List[int] = []
                for tensor in ts:
                    if len(shape) == 0:
                        for x in tensor.shape:
                            shape.append(x)
                    else:
                        for i, x in enumerate(tensor.shape):
                            shape[i] = max(shape[i], x)
                padded_ts = []
                for tensor in ts:
                    p = []
                    for dst, src in zip(shape, tensor.shape):
                        p.append(dst - src)
                        p.append(0)
                    p.reverse()
                    padded_ts.append(F.pad(tensor, p,
                                           value=option.padding_value))

                retval[name] = \
                    torch.stack(padded_ts, dim=option.dim).to(self.device)
        return retval

    def split(self, tensors: Dict[str, Any]) \
            -> Sequence[Dict[str, Any]]:
        retval: List[Dict[str, torch.Tensor]] = []
        for name, t in tensors.items():
            if name in self.options:
                option = self.options[name]
                if option.use_pad_sequence:
                    B = t.data.shape[1]
                else:
                    B = t.data.shape[option.dim]
                if len(retval) == 0:
                    for _ in range(B):
                        retval.append({})

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
                B = len(t)
                if len(retval) == 0:
                    for _ in range(B):
                        retval.append({})
                for b in range(B):
                    retval[b][name] = t[b]

        return retval
