import torch
import numpy as np
from enum import Enum
from typing \
    import List, TypeVar, Generic, Generator, Optional, Callable, Union, \
    Tuple, cast, Dict, Any
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.asts import Root
from mlprogram.actions \
    import ExpandTreeRule, ApplyRule, NodeConstraint, \
    GenerateToken, Action, NodeType, CloseVariadicFieldRule
from mlprogram.actions import ActionSequence
from mlprogram.asts import AST, Node
from mlprogram.samplers import SamplerState, Sampler
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask
from mlprogram.utils import TopKElement, Token
from mlprogram.utils.data import Collate
from mlprogram.utils import random
from mlprogram.utils import logging

logger = logging.Logger(__name__)

V = TypeVar("V")


class Enumeration(Enum):
    Random = 1
    Top = 2


Tensor = Union[torch.Tensor, PaddedSequenceWithMask]


class LazyActionSequence(object):
    def __init__(self, action_sequence: ActionSequence,
                 action: Optional[Action]):
        if action is None:
            self.action_sequence: Optional[ActionSequence] = action_sequence
        else:
            self.old_action_sequence = action_sequence
            self.action = action
            self.action_sequence = None

    def __call__(self):
        if self.action_sequence is not None:
            return self.action_sequence
        self.action_sequence = self.old_action_sequence.clone()
        self.action_sequence.eval(self.action)
        return self.action_sequence

    def __str__(self) -> str:
        return str(self.__call__())

    def __repr__(self) -> str:
        return str(self.__call__())


class ActionSequenceSampler(Sampler[Dict[str, Any], AST, Dict[str, Any]],
                            Generic[V]):
    def __init__(self,
                 encoder: ActionSequenceEncoder,
                 get_token_type: Optional[Callable[[V], Optional[str]]],
                 is_subtype: Callable[[Union[str, Root], Union[str, Root]],
                                      bool],
                 transform_input: Callable[[Dict[str, Any]], Dict[str, Any]],
                 transform_action_sequence: Callable[[Dict[str, Any]],
                                                     Dict[str, Any]],
                 collate: Collate,
                 module: torch.nn.Module,
                 max_samples: int = 1000,
                 eps: float = 1e-5,
                 rng: np.random.RandomState = np.random
                 ):
        self.encoder = encoder
        self.get_token_type = get_token_type or (lambda x: None)
        self.is_subtype = is_subtype
        self.transform_input = transform_input
        self.transform_action_sequence = transform_action_sequence
        self.collate = collate
        self.module = module
        self.max_samples = max_samples
        self.eps = eps
        self.rng = rng

    @logger.function_block("initialize")
    def initialize(self, input: Dict[str, Any]) \
            -> Dict[str, Any]:
        self.module.encoder.eval()
        action_sequence = ActionSequence()
        state_list = self.transform_input(input)
        state_tensor = self.collate.collate([state_list])
        with torch.no_grad(), logger.block("encode_state"):
            state_tensor = self.module.encoder(state_tensor)
        state = self.collate.split(state_tensor)[0]

        # Add initial rule
        action_sequence.eval(ApplyRule(
            ExpandTreeRule(NodeType(None, NodeConstraint.Node, False),
                           [("root",
                             NodeType(Root(), NodeConstraint.Node, False))])))
        state["action_sequence"] = LazyActionSequence(action_sequence, None)
        return state

    def create_output(self, state: Dict[str, Any]) \
            -> Optional[AST]:
        if state["action_sequence"]().head is None:
            # complete
            ast = cast(Node, state["action_sequence"]().generate())
            return cast(AST, ast.fields[0].value)
        return None

    @logger.function_block("batch_infer")
    def batch_infer(self,
                    states: List[SamplerState[Dict[str, Any]]]):
        N = len(states)

        state_list: List[Dict[str, Any]] = []
        for s in states:
            tmp = self.transform_action_sequence(s.state)
            if tmp is not None:
                state = {}
                for key, value in s.state.items():
                    state[key] = value
                for key, value in tmp.items():
                    state[key] = value
                state_list.append(state)
            else:
                logger.warning(
                    "Invalid action_sequence is in the set of hypothesis" +
                    str(s.state["action_sequence"]))
        states_tensor = self.collate.collate(state_list)

        with torch.no_grad(), logger.block("decode_state"):
            next_states = self.module.decoder(states_tensor)

        rule_pred = next_states.pop("rule_probs").data.cpu().reshape(N, -1)
        token_pred = next_states.pop("token_probs").data.cpu().reshape(N, -1)
        reference_pred = \
            next_states.pop("reference_probs").data.cpu().reshape(N, -1)
        next_state_list = self.collate.split(next_states)
        return rule_pred, token_pred, reference_pred, next_state_list

    def enumerate_samples_per_state(self,
                                    rule_pred: torch.Tensor,
                                    token_pred: torch.Tensor,
                                    reference_pred: torch.Tensor,
                                    next_state: Dict[str, Any],
                                    state: SamplerState[Dict[str, Any]],
                                    enumerateion: Enumeration,
                                    k: int) \
            -> Generator[Tuple[float,
                               SamplerState[Dict[str, Any]],
                               Dict[str, Any], Action],
                         None, None]:
        def indices(pred: torch.Tensor):
            # 0 is unknown token
            if enumerateion == Enumeration.Top:
                _, indices = torch.sort(pred[1:], descending=True)
                for index in indices:
                    yield index + 1
            else:
                with logger.block("normalize_prob"):
                    s = pred[1:].sum().item()
                    if s < self.eps:
                        return
                    ps = (pred[1:] / s - self.eps).numpy()
                    npred = [max(0, p) for p in ps]
                for _ in range(self.max_samples):
                    yield self.rng.multinomial(1, npred).nonzero()[0][0] + 1

        with logger.block("enumerate_samples_per_state"):
            head = state.state["action_sequence"]().head
            if head is None:
                return
            head_field = \
                cast(ExpandTreeRule, cast(
                    ApplyRule,
                    state.state["action_sequence"](
                    ).action_sequence[head.action]
                ).rule).children[head.field][1]
            if head_field.constraint == NodeConstraint.Token:
                # Generate token
                if len(state.state["reference"]) == 0:
                    ref_ids = torch.tensor([]).long()
                else:
                    ref_ids = self.encoder._token_encoder.batch_encode(
                        [token.value for token in state.state["reference"]]
                    )
                tokens = list(self.encoder._token_encoder.vocab) + \
                    state.state["reference"]
                # the score will be merged into predefined token
                for i, ref_id in enumerate(ref_ids):
                    # merge token and reference pred
                    token_pred[ref_id] += reference_pred[i]
                reference_pred[ref_ids != 0] = 0.0
                pred = torch.cat([token_pred, reference_pred], dim=0)

                # CloseVariadicFieldRule is a candidate if variadic fields
                if head_field.is_variadic:
                    close_rule_idx = \
                        self.encoder._rule_encoder.encode(
                            CloseVariadicFieldRule())
                    p = rule_pred[close_rule_idx].item()
                    tokens.append(ApplyRule(CloseVariadicFieldRule()))
                    pred = torch.cat([pred, torch.tensor([p])], dim=0)

                with logger.block("exclude_invalid_tokens"):
                    for x, p in enumerate(pred[1:]):
                        x += 1
                        token = tokens[x]
                        if isinstance(token, ApplyRule):
                            action: Action = token
                        else:
                            if isinstance(token, Token):
                                t = token.type_name
                                token = token.value
                            else:
                                t = self.get_token_type(token)
                            if t is not None and \
                                    not self.is_subtype(t,
                                                        head_field.type_name):
                                pred[x] = 0.0

                n_action = 0
                for x in logger.iterable_block("sample-tokens", indices(pred)):
                    # Finish enumeration
                    if n_action == k:
                        return

                    p = pred[x].item()
                    token = tokens[x]

                    if isinstance(token, ApplyRule):
                        action = token
                    else:
                        if isinstance(token, Token):
                            t = token.type_name
                            token = token.value
                        action = GenerateToken(token)

                    if p == 0.0:
                        continue
                    elif p < self.eps:
                        lp = np.log(self.eps)
                    else:
                        lp = np.log(p)
                    n_action += 1
                    yield (state.score + lp, state, next_state,
                           action)
            else:
                # Apply rule
                with logger.block("exclude_invalid_rules"):
                    for x, p in enumerate(rule_pred[1:]):
                        x += 1
                        rule = self.encoder._rule_encoder.vocab[x]
                        if isinstance(rule, ExpandTreeRule):
                            if not (rule.parent.type_name is not None and
                                    self.is_subtype(
                                    rule.parent.type_name,
                                    head_field.type_name)):
                                rule_pred[x] = 0.0
                        else:
                            # CloseVariadicFieldRule
                            if not(head_field is not None and
                                    head_field.is_variadic):
                                rule_pred[x] = 0.0

                n_rule = 0
                for x in logger.iterable_block("sample-rule",
                                               indices(rule_pred)):
                    # Finish enumeration
                    if n_rule == k:
                        return

                    p = rule_pred[x].item()
                    if p == 0.0:
                        continue
                    elif p < self.eps:
                        lp = np.log(self.eps)
                    else:
                        lp = np.log(p)
                    rule = self.encoder._rule_encoder.vocab[x]
                    n_rule += 1
                    yield (state.score + lp, state, next_state,
                           ApplyRule(rule))

    def top_k_samples(
        self, states: List[SamplerState[Dict[str, Any]]], k: int) \
            -> Generator[SamplerState[Dict[str, Any]], None, None]:
        with logger.block("top_k_samples"):
            self.module.eval()
            rule_pred, token_pred, reference_pred, next_states = \
                self.batch_infer(states)
            topk = TopKElement(k)
            for i, state in logger.iterable_block("find_top_k_per_state",
                                                  enumerate(states)):
                for elem in self.enumerate_samples_per_state(
                        rule_pred[i], token_pred[i], reference_pred[i],
                        next_states[i], state, enumerateion=Enumeration.Top,
                        k=k):
                    topk.add(elem[0], tuple(elem[1:]))

            # Instantiate top-k hypothesis
            with logger.block("find_top_k_among_all_states"):
                for score, (state, new_state, action) in topk.elements:
                    s = {key: value for key, value in new_state.items()}
                    s["action_sequence"] = \
                        LazyActionSequence(state.state["action_sequence"](),
                                           action)
                    yield SamplerState[Dict[str, Any]](score, s, 1)

    def k_samples(
        self, states: List[SamplerState[Dict[str, Any]]], n: int) \
            -> Generator[SamplerState[Dict[str, Any]],
                         None, None]:
        with logger.block("k_samples"):
            self.module.eval()

            rule_pred, token_pred, reference_pred, next_states = \
                self.batch_infer(states)

            # Split and decide the number of samples per state
            ks = random.split(self.rng, n, len(states), self.eps)

            for r, t, c, ns, s, k in zip(rule_pred, token_pred, reference_pred,
                                         next_states, states, ks):
                for score, state, new_state, action in \
                        self.enumerate_samples_per_state(
                            r, t, c, ns, s, Enumeration.Random, k=k):
                    x = {key: value for key, value in new_state.items()}
                    x["action_sequence"] = \
                        LazyActionSequence(state.state["action_sequence"](),
                                           action)
                    yield SamplerState[Dict[str, Any]](score, x, 1)
