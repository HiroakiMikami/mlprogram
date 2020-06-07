import torch
import itertools
from dataclasses import dataclass
from typing import Callable, List, Any, Optional, Tuple, Union, Iterable, Dict
from mlprogram.action \
    import Rule, CloseNode, ApplyRule, CloseVariadicFieldRule
from mlprogram.action import ActionSequence
from mlprogram.encoders import Samples
from mlprogram.utils.data import ListDataset
from mlprogram.utils import Query
from mlprogram.nn.utils import rnn
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


def get_words(dataset: torch.utils.data.Dataset,
              extract_query: Callable[[Any], Query],
              ) -> List[str]:
    words = []

    for group in dataset:
        for entry in group:
            query = extract_query(entry.input)
            words.extend(query.query_for_dnn)

    return words


def get_characters(dataset: torch.utils.data.Dataset,
                   extract_query: Callable[[Any], Query],
                   ) -> List[str]:
    chars: List[str] = []

    for group in dataset:
        for entry in group:
            query = extract_query(entry.input)
            for token in query.query_for_dnn:
                chars.extend(token)

    return chars


def get_samples(dataset: torch.utils.data.Dataset,
                tokenize_token: Callable[[str], List[str]],
                to_action_sequence: Callable[[Any],
                                             Optional[ActionSequence]]
                ) -> Samples:
    rules: List[Rule] = []
    node_types = []
    tokens: List[Union[str, CloseNode]] = []
    options = None

    for group in dataset:
        for entry in group:
            action_sequence = to_action_sequence(entry.ground_truth)
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
        for entry in group:
            gts.append(entry.ground_truth)
        for entry in group:
            entries.append((entry.input, gts))
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

    def collate(self, tensors: List[Dict[str, torch.Tensor]]) \
            -> Dict[str, Union[torch.Tensor, PaddedSequenceWithMask]]:
        retval: Dict[str, Union[torch.Tensor, PaddedSequenceWithMask]] = {}
        tmp: Dict[str, List[torch.Tensor]] = {}
        for i, t in enumerate(tensors):
            for name, tensor in t.items():
                if name not in tmp:
                    tmp[name] = []
                tmp[name].append(tensor)
        for name, ts in tmp.items():
            option = self.options[name]
            if option.use_pad_sequence:
                retval[name] = \
                    rnn.pad_sequence(ts, padding_value=option.padding_value) \
                    .to(self.device)
            else:
                retval[name] = torch.stack(ts, dim=option.dim).to(self.device)
        return retval

    def split(self, tensors: Dict[str, Union[torch.Tensor,
                                             PaddedSequenceWithMask]]) \
            -> List[Dict[str, torch.Tensor]]:
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


class CollateGroundTruth:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, ground_truths: List[torch.Tensor]) \
            -> PaddedSequenceWithMask:
        pad_ground_truths = rnn.pad_sequence(ground_truths, padding_value=-1)

        return pad_ground_truths.to(self.device)


class CollateAll:
    def __init__(self, collate_input: Callable[[List[Any]], Any],
                 collate_action_sequence: Callable[[List[Any]], Any],
                 collate_query: Callable[[List[Any]], Any],
                 collate_ground_truth: Callable[[List[Any]], Any]):
        self.collate_input = collate_input
        self.collate_action_sequence = collate_action_sequence
        self.collate_query = collate_query
        self.collate_ground_truth = collate_ground_truth

    def __call__(self, data: List[Tuple[Any, Any, Any, Any]]) \
            -> Tuple[Any, Any, Any, Any]:
        inputs = self.collate_input([elem[0] for elem in data])
        action_sequences = \
            self.collate_action_sequence([elem[1] for elem in data])
        queries = self.collate_query([elem[2] for elem in data])
        ground_truths = self.collate_ground_truth([elem[3] for elem in data])

        return inputs, action_sequences, queries, ground_truths


class CollateNlFeature:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, data: List[PaddedSequenceWithMask]) \
            -> PaddedSequenceWithMask:
        nl_features = []
        for nl_feature in data:
            nl_feature_tensor = nl_feature.data
            L = nl_feature_tensor.shape[0]
            nl_feature_tensor = nl_feature_tensor.view(L, -1)
            nl_features.append(nl_feature_tensor)

        return rnn.pad_sequence(nl_features).to(self.device)


def collate_none(data: List[Any]) -> None:
    return None


def split_none(state: Tuple[Any]) -> Iterable[None]:
    return itertools.repeat(None)
