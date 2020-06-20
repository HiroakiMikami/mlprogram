import torch
import numpy as np
import logging
from typing \
    import List, TypeVar, Generic, Generator, Optional, Callable, Union, \
    Tuple, cast, Dict, Any
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.asts import Root
from mlprogram.actions \
    import ExpandTreeRule, ApplyRule, NodeConstraint, ActionOptions, \
    GenerateToken, Action, NodeType
from mlprogram.actions import ActionSequence
from mlprogram.asts import AST
from mlprogram.samplers import SamplerState, Sampler
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask
from mlprogram.utils import TopKElement
from mlprogram.utils.data import Collate
from dataclasses import dataclass

logger = logging.getLogger(__name__)

V = TypeVar("V")


Tensor = Union[torch.Tensor, PaddedSequenceWithMask]


@dataclass
class Token(Generic[V]):
    type_name: Optional[str]
    value: V


@dataclass
class ActionSequenceSamplerInput(Generic[V]):
    state: Dict[str, Any]
    reference: List[Token[V]]


@dataclass
class ActionSequenceSamplerState(Generic[V]):
    state: Dict[str, Any]
    reference: List[Token[V]]
    action_sequence: ActionSequence

    # TODO eq, hash


class ActionSequenceSampler(Sampler[ActionSequenceSamplerInput, AST,
                                    ActionSequenceSamplerState[V]],
                            Generic[V]):
    def __init__(self,
                 encoder: ActionSequenceEncoder,
                 get_token_type: Callable[[V], Optional[str]],
                 is_subtype: Callable[[Union[str, Root], Union[str, Root]],
                                      bool],
                 transform_action_sequence: Callable[
                     [ActionSequence, List[Token]],
                     Optional[Dict[str, Any]]],
                 collate: Collate,
                 decoder_module: torch.nn.Module,
                 options: ActionOptions = ActionOptions(True, True),
                 eps: float = 1e-8,
                 rng: np.random.RandomState = np.random
                 ):
        self.encoder = encoder
        self.get_token_type = get_token_type
        self.is_subtype = is_subtype
        self.transform_action_sequence = transform_action_sequence
        self.collate = collate
        self.decoder_module = decoder_module
        self.options = options
        self.eps = eps
        self.rng = rng

    def initialize(self, input: ActionSequenceSamplerInput) \
            -> ActionSequenceSamplerState[V]:
        action_sequence = ActionSequence(self.options)
        # Add initial rule
        action_sequence.eval(ApplyRule(
            ExpandTreeRule(NodeType(Root(), NodeConstraint.Node),
                           [("root", NodeType(Root(), NodeConstraint.Node))])))
        return ActionSequenceSamplerState[V](input.state, input.reference,
                                             action_sequence)

    def create_output(self, state: ActionSequenceSamplerState[V]) \
            -> Optional[AST]:
        if state.action_sequence.head is None:
            # complete
            return state.action_sequence.generate()
        return None

    def batch_infer(self,
                    states: List[SamplerState[ActionSequenceSamplerState[V]]]):
        N = len(states)

        state_list: List[Dict[str, Any]] = []
        for s in states:
            tmp = self.transform_action_sequence(s.state.action_sequence,
                                                 s.state.reference)
            if tmp is not None:
                state = {}
                for key, value in s.state.state.items():
                    state[key] = value
                for key, value in tmp.items():
                    state[key] = value
                state_list.append(state)
            else:
                logger.warn(
                    "Invalid action_sequence is in the set of hypothesis")
        states_tensor = self.collate.collate(state_list)

        with torch.no_grad():
            next_states = self.decoder_module(**states_tensor)

        rule_pred = next_states.pop("rule_probs").data.cpu().reshape(N, -1)
        token_pred = next_states.pop("token_probs").data.cpu().reshape(N, -1)
        copy_pred = next_states.pop("copy_probs").data.cpu().reshape(N, -1)
        next_state_list = self.collate.split(next_states)
        return rule_pred, token_pred, copy_pred, next_state_list

    def all_actions(
        self, states: List[SamplerState[ActionSequenceSamplerState[V]]]) \
            -> Generator[Tuple[float,
                               SamplerState[ActionSequenceSamplerState[V]],
                               Tensor, Action],
                         None, None]:
        rule_pred, token_pred, copy_pred, next_states = \
            self.batch_infer(states)
        for i, state in enumerate(states):
            state = states[i]
            head = state.state.action_sequence.head
            is_token = False
            if head is None:
                continue
            head_field = \
                cast(ExpandTreeRule, cast(
                    ApplyRule,
                    state.state.action_sequence.action_sequence[head.action]
                ).rule).children[head.field][1]
            if head_field.constraint == NodeConstraint.Token:
                is_token = True
            if is_token:
                # Generate token
                probs: Dict[V, float] = {}
                # predefined token
                for j in range(1, token_pred.shape[1]):
                    # 0 is unknown token
                    p = token_pred[i, j].item()
                    token = self.encoder._token_encoder.vocab[j]
                    t = self.get_token_type(token)
                    if t is not None and \
                            not self.is_subtype(t, head_field.type_name):
                        continue
                    if token not in probs:
                        probs[token] = 0.0
                    probs[token] += p
                # reference
                for j in range(len(state.state.reference)):
                    p = copy_pred[i, j].item()
                    token = state.state.reference[j].value
                    t = state.state.reference[j].type_name
                    if t is not None and \
                            not self.is_subtype(t, head_field.type_name):
                        continue
                    if token not in probs:
                        probs[token] = 0.0
                    probs[token] += p

                # Convert to log probs
                for token, p in probs.items():
                    if p < self.eps:
                        lp = np.log(self.eps)
                    else:
                        lp = np.log(p)
                    yield (state.score + lp, state, next_states[i],
                           GenerateToken(token))
            else:
                # Apply rule
                for j in range(1, rule_pred.shape[1]):
                    # 0 is unknown rule
                    p = rule_pred[i, j].item()
                    if p < self.eps:
                        lp = np.log(self.eps)
                    else:
                        lp = np.log(p)
                    rule = self.encoder._rule_encoder.vocab[j]
                    if isinstance(rule, ExpandTreeRule):
                        if self.is_subtype(
                                rule.parent.type_name,
                                head_field.type_name):
                            yield (state.score + lp, states[i],
                                   next_states[i], ApplyRule(rule))
                    else:
                        # CloseVariadicFieldRule
                        if self.options.retain_variadic_fields and \
                            head_field is not None and \
                            head_field.constraint == \
                                NodeConstraint.Variadic:
                            yield (state.score + lp, states[i], next_states[i],
                                   ApplyRule(rule))

    def top_k_samples(
        self, states: List[SamplerState[ActionSequenceSamplerState]], k: int) \
            -> Generator[SamplerState[ActionSequenceSamplerState], None, None]:
        topk = TopKElement(k)
        for score, state, new_state, action in self.all_actions(states):
            topk.add(score, (state, new_state, action))

        # Instantiate top-k hypothesis
        for score, (state, new_state, action) in topk.elements:
            action_sequence = state.state.action_sequence.clone()
            action_sequence.eval(action)
            yield SamplerState[ActionSequenceSamplerState](
                score,
                ActionSequenceSamplerState(
                    new_state,
                    state.state.reference,
                    action_sequence))

    def random_samples(
        self, state: SamplerState[ActionSequenceSamplerState[V]], n: int) \
            -> Generator[SamplerState[ActionSequenceSamplerState[V]],
                         None, None]:
        actions = list(self.all_actions([state]))
        # log_prob -> prob
        probs = [np.exp(score) for score, _, _, _ in actions]
        # normalize
        probs = [p / sum(probs) for p in probs]
        resamples = self.rng.multinomial(n, probs)
        for (score, state, new_state, action), m in zip(actions, resamples):
            for _ in range(m):
                action_sequence = state.state.action_sequence.clone()
                action_sequence.eval(action)
                yield SamplerState[ActionSequenceSamplerState](
                    score, ActionSequenceSamplerState(
                        cast(Dict[str, Any], new_state), state.state.reference,
                        action_sequence))
