import torch
import numpy as np
from enum import Enum
from typing \
    import List, TypeVar, Generic, Generator, Optional, Callable, Union, \
    cast, Dict, Tuple

from mlprogram import Environment
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.languages import Root
from mlprogram.languages import Token
from mlprogram.actions \
    import ExpandTreeRule, ApplyRule, NodeConstraint, \
    GenerateToken, Action, NodeType, CloseVariadicFieldRule
from mlprogram.actions import ActionSequence
from mlprogram.languages import AST, Node
from mlprogram.samplers import SamplerState, DuplicatedSamplerState, Sampler
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask
from mlprogram.collections import TopKElement
from mlprogram.utils.data import Collate
from mlprogram import logging

logger = logging.Logger(__name__)

Input = TypeVar("Input")
V = TypeVar("V")


class Enumeration(Enum):
    Random = 1
    Top = 2


Tensor = Union[torch.Tensor, PaddedSequenceWithMask]


class LazyActionSequence(object):
    def __init__(self, action_sequence: ActionSequence,
                 action: Action):
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


class ActionSequenceSampler(Sampler[Environment, AST, Environment],
                            Generic[Input, V]):
    def __init__(self,
                 encoder: ActionSequenceEncoder,
                 is_subtype: Callable[[Union[str, Root], Union[str, Root]],
                                      bool],
                 transform_input: Callable[[Input], Environment],
                 transform_action_sequence: Callable[[Environment],
                                                     Environment],
                 collate: Collate,
                 module: torch.nn.Module,
                 eps: float = 1e-5,
                 rng: Optional[np.random.RandomState] = None
                 ):
        self.encoder = encoder
        self.is_subtype = is_subtype
        self.transform_input = transform_input
        self.transform_action_sequence = transform_action_sequence
        self.collate = collate
        self.module = module
        self.eps = eps
        self.rng = \
            rng or np.random.RandomState(np.random.randint(0, 2 << 32 - 1))

        self.token_kind_to_idx: Dict[str, List[int]] = {}
        for token in self.encoder._token_encoder.vocab:
            if isinstance(token, tuple):
                kind, _ = token
                if kind not in self.token_kind_to_idx:
                    self.token_kind_to_idx[kind] = []
                self.token_kind_to_idx[kind].append(
                    self.encoder._token_encoder.encode(token).item()
                )
        self.rule_kind_to_idx: Dict[str, List[int]] = {}
        for rule in self.encoder._rule_encoder.vocab:
            if isinstance(rule, ExpandTreeRule):
                kind = rule.parent.type_name
                if kind not in self.rule_kind_to_idx:
                    self.rule_kind_to_idx[kind] = []
                self.rule_kind_to_idx[kind].append(
                    self.encoder._rule_encoder.encode(rule).item()
                )

    @ logger.function_block("initialize")
    def initialize(self, input: Input) -> Environment:
        self.module.encoder.eval()
        state_list = self.transform_input(input)
        state_tensor = self.collate.collate([state_list])
        with torch.no_grad(), logger.block("encode_state"):
            state_tensor = self.module.encoder(state_tensor)
        state = self.collate.split(state_tensor)[0]

        # Add initial rule
        action_sequence = ActionSequence()
        action_sequence.eval(ApplyRule(
            ExpandTreeRule(NodeType(None, NodeConstraint.Node, False),
                           [("root",
                             NodeType(Root(), NodeConstraint.Node, False))])))
        state.states["action_sequence"] = action_sequence

        state.mutable(
            inputs=False,
            supervisions=False
        )
        return state

    def create_output(self, input, state: Environment) \
            -> Optional[Tuple[AST, bool]]:
        if state.states["action_sequence"].head is None:
            # complete
            ast = cast(Node, state.states["action_sequence"].generate())
            return cast(AST, ast.fields[0].value), True
        return None

    @ logger.function_block("batch_infer")
    def batch_infer(self,
                    states: List[SamplerState[Environment]]):
        N = len(states)

        state_list: List[Environment] = []
        for s in logger.iterable_block("transform_state", states):
            tmp = self.transform_action_sequence(s.state)
            if tmp is not None:
                state_list.append(tmp)
            else:
                logger.warning(
                    "Invalid action_sequence is in the set of hypothesis" +
                    str(s.state.outputs["action_sequence"]))
        states_tensor = self.collate.collate(state_list)

        with torch.no_grad(), logger.block("decode_state"):
            next_states = self.module.decoder(states_tensor)

        rule_pred = next_states.outputs["rule_probs"].data.cpu().reshape(N, -1)
        token_pred = \
            next_states.outputs["token_probs"].data.cpu().reshape(N, -1)
        reference_pred = \
            next_states.outputs["reference_probs"].data.cpu().reshape(N, -1)
        next_state_list = self.collate.split(next_states)
        return rule_pred, token_pred, reference_pred, next_state_list

    def enumerate_samples_per_state(self,
                                    rule_pred: torch.Tensor,
                                    token_pred: torch.Tensor,
                                    reference_pred: torch.Tensor,
                                    next_state: Environment,
                                    state: SamplerState[Environment],
                                    enumerateion: Enumeration,
                                    k: int) \
            -> Generator[DuplicatedSamplerState[Environment], None, None]:
        def indices(pred: torch.Tensor):
            # 0 is unknown token
            if enumerateion == Enumeration.Top:
                _, indices = torch.sort(pred[1:], descending=True)
                for index in indices[:k]:
                    yield index + 1, 1
            else:
                with logger.block("normalize_prob"):
                    s = pred[1:].sum().item()
                    if s < self.eps:
                        return
                    ps = (pred[1:] / s - self.eps).numpy()
                    npred = [max(0, p) for p in ps]
                for i, n in enumerate(self.rng.multinomial(k, npred)):
                    if n == 0:
                        continue
                    yield i + 1, n

        with logger.block("enumerate_samples_per_state"):
            head = state.state.states["action_sequence"].head
            assert head is not None
            head_field = \
                cast(ExpandTreeRule, cast(
                    ApplyRule,
                    state.state.states["action_sequence"]
                    .action_sequence[head.action]
                ).rule).children[head.field][1]
            if head_field.constraint == NodeConstraint.Token:
                # Generate token
                ref_ids = self.encoder.batch_encode_raw_value(
                    [x.raw_value for x in state.state.states["reference"]]
                )
                tokens = list(self.encoder._token_encoder.vocab) + \
                    state.state.states["reference"]
                # the score will be merged into predefined token
                for i, ids in enumerate(ref_ids):
                    for ref_id in ids:
                        # merge token and reference pred
                        token_pred[ref_id] += reference_pred[i]
                        if ref_id != 0:
                            reference_pred[i] = 0.0
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
                    # token
                    for kind, idxes in self.token_kind_to_idx.items():
                        if kind is not None and \
                                not self.is_subtype(kind,
                                                    head_field.type_name):
                            pred[idxes] = 0.0
                    # reference
                    for x, (p, token) in enumerate(
                            zip(pred[len(token_pred):],
                                tokens[len(token_pred):])):
                        x += len(token_pred)
                        if not isinstance(token, ApplyRule):
                            if isinstance(token, Token):
                                t = token.kind
                            else:
                                t = token[0]
                            if t is not None and \
                                    not self.is_subtype(t,
                                                        head_field.type_name):
                                pred[x] = 0.0

                n_action = 0
                for x, n in logger.iterable_block("sample-tokens",
                                                  indices(pred)):
                    # Finish enumeration
                    if n_action == k:
                        return

                    p = pred[x].item()
                    token = tokens[x]

                    if isinstance(token, ApplyRule):
                        action: Action = token
                    elif isinstance(token, Token):
                        action = GenerateToken(token.kind, token.raw_value)
                    else:
                        action = GenerateToken(token[0], token[1])

                    if p == 0.0:
                        continue
                    elif p < self.eps:
                        lp = np.log(self.eps)
                    else:
                        lp = np.log(p)

                    n_action += n
                    next_state = next_state.clone()
                    next_state.outputs.clear()
                    next_state.states["action_sequence"] = \
                        LazyActionSequence(
                            state.state.states["action_sequence"], action)
                    yield DuplicatedSamplerState(
                        SamplerState(state.score + lp, next_state),
                        n)
            else:
                # Apply rule
                with logger.block("exclude_invalid_rules"):
                    # expand tree rule
                    for kind, idxes in self.rule_kind_to_idx.items():
                        if not (kind is not None and
                                self.is_subtype(
                                    kind,
                                    head_field.type_name)):
                            rule_pred[idxes] = 0.0
                    # CloseVariadicField
                    idx = self.encoder._rule_encoder.encode(
                        CloseVariadicFieldRule())
                    if not(head_field is not None and
                           head_field.is_variadic):
                        rule_pred[idx] = 0.0

                n_rule = 0
                for x, n in logger.iterable_block("sample-rule",
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

                    n_rule += n
                    rule = self.encoder._rule_encoder.vocab[x]

                    next_state = next_state.clone()
                    next_state.states["action_sequence"] = \
                        LazyActionSequence(
                            state.state.states["action_sequence"],
                            ApplyRule(rule))
                    yield DuplicatedSamplerState(
                        SamplerState(state.score + lp, next_state),
                        n)

    def top_k_samples(
        self, states: List[SamplerState[Environment]], k: int) \
            -> Generator[DuplicatedSamplerState[Environment], None, None]:
        assert all([len(state.state.outputs) for state in states]) == 0
        assert all([len(state.state.supervisions) for state in states]) == 0

        with logger.block("top_k_samples"):
            states = [
                state for state in states
                if state.state.states["action_sequence"].head is not None]
            if len(states) == 0:
                return

            self.module.eval()
            rule_pred, token_pred, reference_pred, next_states = \
                self.batch_infer(states)
            topk = TopKElement(k)
            for i, state in logger.iterable_block("find_top_k_per_state",
                                                  enumerate(states)):
                for state in self.enumerate_samples_per_state(
                        rule_pred[i], token_pred[i], reference_pred[i],
                        next_states[i], state, enumerateion=Enumeration.Top,
                        k=k):
                    topk.add(state.state.score, state)

            # Instantiate top-k hypothesis
            with logger.block("find_top_k_among_all_states"):
                for score, state in topk.elements:
                    state.state.state.states["action_sequence"] = \
                        state.state.state.states["action_sequence"]()
                    yield state

    def batch_k_samples(
        self, states: List[SamplerState[Environment]], ks: List[int]) \
            -> Generator[DuplicatedSamplerState[Environment],
                         None, None]:
        assert all([len(state.state.outputs) for state in states]) == 0
        assert all([len(state.state.supervisions) for state in states]) == 0

        with logger.block("batch_k_samples"):
            self.module.eval()
            ks = [
                ks[i] for i in range(len(states))
                if states[i].state.states["action_sequence"].head is not None]
            states = [
                state for state in states
                if state.state.states["action_sequence"].head is not None]
            if len(states) == 0:
                return

            rule_pred, token_pred, reference_pred, next_states = \
                self.batch_infer(states)

            for r, t, c, ns, s, k in zip(rule_pred, token_pred, reference_pred,
                                         next_states, states, ks):
                for state in self.enumerate_samples_per_state(
                        r, t, c, ns, s, Enumeration.Random, k=k):
                    state.state.state.states["action_sequence"] = \
                        state.state.state.states["action_sequence"]()
                    yield state
