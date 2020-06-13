import torch
import numpy as np
import logging
from typing \
    import List, TypeVar, Generic, Generator, Optional, Callable, Union, \
    Tuple, cast, Dict
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.asts import Root
from mlprogram.actions \
    import ExpandTreeRule, ApplyRule, NodeConstraint, ActionOptions, \
    GenerateToken, Action
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
class Input(Generic[V]):
    input: Optional[Tensor]
    raw_reference: List[Token[V]]
    reference: Optional[Tensor]


@dataclass
class State(Generic[V]):
    input: Optional[Tensor]
    raw_reference: List[Token[V]]
    reference: Optional[Tensor]
    state: Optional[Tensor]
    evaluator: ActionSequence

    # TODO eq, hash


class ActionSequenceSampler(Sampler[Input, AST, State[V]], Generic[V]):
    def __init__(self,
                 encoder: ActionSequenceEncoder,
                 get_token_type: Callable[[V], Optional[str]],
                 is_subtype: Callable[[Union[str, Root], Union[str, Root]],
                                      bool],
                 transform_evaluator: Callable[[ActionSequence, List[Token]],
                                               Optional[Tuple[Tensor,
                                                              Optional[Tensor]]
                                                        ]],
                 collate_input: Collate,
                 collate_reference: Collate,
                 collate_action_sequence: Collate,
                 collate_query: Collate,
                 collate_state: Collate,
                 decoder_module: torch.nn.Module,
                 options: ActionOptions = ActionOptions(True, True),
                 eps: float = 1e-8,
                 rng: np.random.RandomState = np.random
                 ):
        self.encoder = encoder
        self.get_token_type = get_token_type
        self.is_subtype = is_subtype
        self.transform_evaluator = transform_evaluator
        self.collate_input = collate_input
        self.collate_action_sequence = collate_action_sequence
        self.collate_reference = collate_reference
        self.collate_query = collate_query
        self.collate_state = collate_state
        self.decoder_module = decoder_module
        self.options = options
        self.eps = eps
        self.rng = rng

    def initialize(self, input: Input) -> State[V]:
        return State[V](input.input, input.raw_reference, input.reference,
                        None, ActionSequence(self.options))

    def create_output(self, state: State[V]) -> Optional[AST]:
        if state.evaluator.head is None:
            # complete
            return state.evaluator.generate()
        return None

    def batch_infer(self, states: List[SamplerState[State[V]]]):
        N = len(states)

        inputs: List[Dict[str, Tensor]] = []
        references: List[Dict[str, Tensor]] = []
        action_sequences: List[Dict[str, Tensor]] = []
        queries: List[Dict[str, Tensor]] = []
        states_list: List[Dict[str, Optional[Tensor]]] = []
        for s in states:
            tmp = self.transform_evaluator(s.state.evaluator,
                                           s.state.raw_reference)
            if tmp is not None:
                inputs.append({"input": s.state.input})
                references.append({"reference": s.state.reference})
                action_sequence, query = tmp
                action_sequences.append({"action_sequence": action_sequence})
                queries.append({"query": query})
                states_list.append({"state": s.state.state})
            else:
                logger.warn("Invalid evaluator is in the set of hypothesis")
        inputs_tensor = self.collate_input.collate(inputs)
        references_tensor = self.collate_reference.collate(references)
        action_sequences_tensor = \
            self.collate_action_sequence.collate(action_sequences)
        queries_tensor = self.collate_query.collate(queries)
        states_tensor = self.collate_state.collate(states_list)

        with torch.no_grad():
            results, state = self.decoder_module(
                inputs=inputs_tensor,
                references=references_tensor,
                action_sequences=action_sequences_tensor,
                queries=queries_tensor,
                states=states_tensor)
        rule_pred = results[0].data.cpu().reshape(N, -1)
        token_pred = results[1].data.cpu().reshape(N, -1)
        copy_pred = results[2].data.cpu().reshape(N, -1)
        if state is not None:
            next_state: List[Dict[str, Optional[torch.Tensor]]] = \
                self.collate_state.split(state)
        else:
            next_state = [{"state": None}] * len(states)
        return rule_pred, token_pred, copy_pred, next_state

    def all_actions(self, states: List[SamplerState[State[V]]]) \
            -> Generator[Tuple[float, SamplerState[State[V]], Tensor, Action],
                         None, None]:
        rule_pred, token_pred, copy_pred, next_states = \
            self.batch_infer(states)
        for i, state in enumerate(states):
            state = states[i]
            head = state.state.evaluator.head
            is_token = False
            if head is None:
                continue
            head_field = \
                cast(ExpandTreeRule, cast(
                    ApplyRule,
                    state.state.evaluator.action_sequence[head.action]
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
                    if t is not None and self.is_subtype(t,
                                                         head_field.type_name):
                        continue
                    if token not in probs:
                        probs[token] = 0.0
                    probs[token] += p
                # reference
                for j in range(len(state.state.raw_reference)):
                    p = copy_pred[i, j].item()
                    token = state.state.raw_reference[j].value
                    t = state.state.raw_reference[j].type_name
                    if t is not None and self.is_subtype(t,
                                                         head_field.type_name):
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
                        if head_field.type_name == Root() and \
                            (rule.parent.constraint !=
                                NodeConstraint.Variadic) and \
                            (rule.parent.type_name !=
                                Root()):
                            yield (state.score + lp, states[i], next_states[i],
                                   ApplyRule(rule))
                        else:
                            if head_field.type_name == Root() and \
                                (rule.parent.type_name !=
                                    Root()) and \
                                (rule.parent.constraint !=
                                    NodeConstraint.Variadic):
                                yield (state.score + lp, states[i],
                                       next_states[i], ApplyRule(rule))
                            elif rule.parent.type_name != Root() and \
                                ((head_field.constraint ==
                                    NodeConstraint.Variadic) or
                                 (rule.parent.constraint ==
                                    NodeConstraint.Variadic)):
                                if rule.parent == \
                                        head_field:
                                    yield (state.score + lp, states[i],
                                           next_states[i], ApplyRule(rule))
                            elif rule.parent.type_name != Root() and \
                                self.is_subtype(
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

    def top_k_samples(self, states: List[SamplerState[State]], k: int) \
            -> Generator[SamplerState[State], None, None]:
        topk = TopKElement(k)
        for score, state, new_state, action in self.all_actions(states):
            topk.add(score, (state, new_state, action))

        # Instantiate top-k hypothesis
        for score, (state, new_state, action) in topk.elements:
            evaluator = state.state.evaluator.clone()
            evaluator.eval(action)
            yield SamplerState[State](score, State(
                state.state.input, state.state.raw_reference,
                state.state.reference,
                new_state, evaluator))

    def random_samples(self, state: SamplerState[State[V]], n: int) \
            -> Generator[SamplerState[State[V]], None, None]:
        actions = list(self.all_actions([state]))
        # log_prob -> prob
        probs = [np.exp(score) for score, _, _, _ in actions]
        # normalize
        probs = [p / sum(probs) for p in probs]
        resamples = self.rng.multinomial(n, probs)
        for (score, state, new_state, action), m in zip(actions, resamples):
            for _ in range(m):
                evaluator = state.state.evaluator.clone()
                evaluator.eval(action)
                yield SamplerState[State](score, State(
                    state.state.input, state.state.raw_reference,
                    state.state.reference,
                    new_state, evaluator))
