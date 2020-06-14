import torch
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Callable, Sequence, Any, Optional, Union, Dict, List
from mlprogram.actions \
    import Rule, CloseNode, ApplyRule, CloseVariadicFieldRule
from mlprogram.actions import ActionSequence
from mlprogram.encoders import Samples
from mlprogram.utils.data import ListDataset
from mlprogram.utils import Query
from mlprogram.nn.utils import rnn
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


def get_words(dataset: torch.utils.data.Dataset,
              extract_query: Callable[[Any], Query],
              ) -> Sequence[str]:
    words = []

    for group in dataset:
        for input in group["input"]:
            query = extract_query(input)
            words.extend(query.query_for_dnn)

    return words


def get_characters(dataset: torch.utils.data.Dataset,
                   extract_query: Callable[[Any], Query],
                   ) -> Sequence[str]:
    chars: List[str] = []

    for group in dataset:
        for input in group["input"]:
            query = extract_query(input)
            for token in query.query_for_dnn:
                chars.extend(token)

    return chars


def get_samples(dataset: torch.utils.data.Dataset,
                tokenize_token: Callable[[str], Sequence[str]],
                to_action_sequence: Callable[[Any],
                                             Optional[ActionSequence]]
                ) -> Samples:
    rules: List[Rule] = []
    node_types = []
    tokens: List[Union[str, CloseNode]] = []
    options = None

    for group in dataset:
        for gt in group["ground_truth"]:
            action_sequence = to_action_sequence(gt)
            if action_sequence is None:
                continue
            if options is not None:
                assert(options == action_sequence._options)
            options = action_sequence._options
            for action in action_sequence.action_sequence:
                if isinstance(action, ApplyRule):
                    rule = action.rule
                    if not isinstance(rule, CloseVariadicFieldRule):
                        rules.append(rule)
                        node_types.append(rule.parent)
                        for _, child in rule.children:
                            node_types.append(child)
                else:
                    token = action.token
                    if not isinstance(token, CloseNode):
                        ts = tokenize_token(token)
                        tokens.extend(ts)

    assert options is not None
    return Samples(rules, node_types, tokens, options)


def to_eval_dataset(dataset: torch.utils.data.Dataset) \
        -> torch.utils.data.Dataset:
    entries = []
    for group in dataset:
        gts = []
        for ground_truth in group["ground_truth"]:
            gts.append(ground_truth)
        for input in group["input"]:
            entries.append((input, gts))
    return ListDataset(entries)


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

    def __call__(self, tensors: Sequence[Optional[Dict[str,
                                                       Optional[torch.Tensor]
                                                       ]]]) \
            -> Dict[str, Union[torch.Tensor, PaddedSequenceWithMask]]:
        return self.collate(tensors)

    def collate(self, tensors: Sequence[Optional[Dict[str,
                                                      Optional[torch.Tensor]
                                                      ]]]) \
            -> Dict[str, Union[torch.Tensor, PaddedSequenceWithMask]]:
        retval: Dict[str, Union[torch.Tensor, PaddedSequenceWithMask]] = {}
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

    def split(self, tensors: Dict[str, Union[torch.Tensor,
                                             PaddedSequenceWithMask]]) \
            -> Sequence[Dict[str, torch.Tensor]]:
        tensors = {key: value for key, value in tensors.items()
                   if key in self.options.keys()}
        retval: List[Dict[str, torch.Tensor]] = []
        for name, t in tensors.items():
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
                    d = d.reshape(*shape)
                    retval[b][name] = d

        return retval
