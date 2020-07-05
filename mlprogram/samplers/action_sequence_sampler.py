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
    GenerateToken, Action, NodeType, CloseVariadicFieldRule
from mlprogram.actions import ActionSequence
from mlprogram.asts import AST, Node
from mlprogram.samplers import SamplerState, Sampler
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask
from mlprogram.utils import TopKElement, Token
from mlprogram.utils.data import Collate

logger = logging.getLogger(__name__)

V = TypeVar("V")


Tensor = Union[torch.Tensor, PaddedSequenceWithMask]


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
                 options: ActionOptions = ActionOptions(True, True),
                 eps: float = 1e-8,
                 rng: np.random.RandomState = np.random
                 ):
        self.encoder = encoder
        self.get_token_type = get_token_type or (lambda x: None)
        self.is_subtype = is_subtype
        self.transform_input = transform_input
        self.transform_action_sequence = transform_action_sequence
        self.collate = collate
        self.module = module
        self.options = options
        self.eps = eps
        self.rng = rng

    def initialize(self, input: Dict[str, Any]) \
            -> Dict[str, Any]:
        action_sequence = ActionSequence(self.options)
        state_list = self.transform_input(input)
        state_tensor = self.collate.collate([state_list])
        state_tensor = self.module.encoder(state_tensor)
        state = self.collate.split(state_tensor)[0]

        # Add initial rule
        action_sequence.eval(ApplyRule(
            ExpandTreeRule(NodeType(None, NodeConstraint.Node, False),
                           [("root",
                             NodeType(Root(), NodeConstraint.Node, False))])))
        state["action_sequence"] = action_sequence
        return state

    def create_output(self, state: Dict[str, Any]) \
            -> Optional[AST]:
        if state["action_sequence"].head is None:
            # complete
            ast = cast(Node, state["action_sequence"].generate())
            return cast(AST, ast.fields[0].value)
        return None

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

        with torch.no_grad():
            next_states = self.module.decoder(states_tensor)

        rule_pred = next_states.pop("rule_probs").data.cpu().reshape(N, -1)
        token_pred = next_states.pop("token_probs").data.cpu().reshape(N, -1)
        copy_pred = next_states.pop("copy_probs").data.cpu().reshape(N, -1)
        next_state_list = self.collate.split(next_states)
        return rule_pred, token_pred, copy_pred, next_state_list

    def enumerate_samples_per_state(self,
                                    rule_pred: torch.Tensor,
                                    token_pred: torch.Tensor,
                                    copy_pred: torch.Tensor,
                                    next_state: Dict[str, Any],
                                    state: SamplerState[Dict[str, Any]],
                                    k: Optional[int] = None) \
            -> Generator[Tuple[float,
                               SamplerState[Dict[str, Any]],
                               Dict[str, Any], Action],
                         None, None]:
        head = state.state["action_sequence"].head
        is_token = False
        if head is None:
            return
        head_field = \
            cast(ExpandTreeRule, cast(
                ApplyRule,
                state.state["action_sequence"].action_sequence[head.action]
            ).rule).children[head.field][1]
        if head_field.constraint == NodeConstraint.Token:
            is_token = True
        if is_token:
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
            token_pred[ref_ids] += copy_pred  # merge token and copy pred
            copy_pred[ref_ids != 0] = 0.0
            pred = torch.cat([token_pred, copy_pred], dim=0)

            # CloseVariadicFieldRule is a candidate if split_non_terminals=True
            if self.options.split_non_terminal:
                close_rule_idx = \
                    self.encoder._rule_encoder.encode(CloseVariadicFieldRule())
                p = rule_pred[close_rule_idx].item()
                tokens.append(ApplyRule(CloseVariadicFieldRule()))
                pred = torch.cat([pred, torch.tensor([p])], dim=0)
            n_action = 0
            # 0 is unknown token
            _, indices = torch.sort(pred[1:], descending=True)
            for j in range(len(indices)):
                # Finish enumeration
                if n_action == k:
                    return

                x = indices[j] + 1
                p = pred[x].item()
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
                            not self.is_subtype(t, head_field.type_name):
                        continue
                    action = GenerateToken(token)

                if p < self.eps:
                    lp = np.log(self.eps)
                else:
                    lp = np.log(p)
                n_action += 1
                yield (state.score + lp, state, next_state,
                       action)
        else:
            # Apply rule
            # 0 is unknown rule
            _, indices = torch.sort(rule_pred[1:], descending=True)
            n_rule = 0
            for j in range(min(rule_pred.shape[0],
                               self.encoder._rule_encoder.vocab_size) - 1):
                # Finish enumeration
                if n_rule == k:
                    return

                x = indices[j] + 1
                p = rule_pred[x].item()
                if p < self.eps:
                    lp = np.log(self.eps)
                else:
                    lp = np.log(p)
                rule = self.encoder._rule_encoder.vocab[x]
                if isinstance(rule, ExpandTreeRule):
                    if rule.parent.type_name is not None and \
                        self.is_subtype(
                            rule.parent.type_name,
                            head_field.type_name):
                        n_rule += 1
                        yield (state.score + lp, state,
                               next_state, ApplyRule(rule))
                else:
                    # CloseVariadicFieldRule
                    if self.options.retain_variadic_fields and \
                            head_field is not None and \
                            head_field.is_variadic:
                        n_rule += 1
                        yield (state.score + lp, state, next_state,
                               ApplyRule(rule))

    def top_k_samples(
        self, states: List[SamplerState[Dict[str, Any]]], k: int) \
            -> Generator[SamplerState[Dict[str, Any]], None, None]:
        self.module.eval()
        rule_pred, token_pred, copy_pred, next_states = \
            self.batch_infer(states)
        topk = TopKElement(k)
        for i, state in enumerate(states):
            for elem in self.enumerate_samples_per_state(rule_pred[i],
                                                         token_pred[i],
                                                         copy_pred[i],
                                                         next_states[i],
                                                         state,
                                                         k=k):
                topk.add(elem[0], tuple(elem[1:]))

        # Instantiate top-k hypothesis
        for score, (state, new_state, action) in topk.elements:
            action_sequence = state.state["action_sequence"].clone()
            action_sequence.eval(action)
            s = {key: value for key, value in new_state.items()}
            s["action_sequence"] = action_sequence
            yield SamplerState[Dict[str, Any]](score, s)

    def k_samples(
        self, states: List[SamplerState[Dict[str, Any]]], n: int) \
            -> Generator[SamplerState[Dict[str, Any]],
                         None, None]:
        self.module.eval()

        rule_pred, token_pred, copy_pred, next_states = \
            self.batch_infer(states)
        actions = []
        for r, t, c, ns, s in zip(rule_pred, token_pred, copy_pred,
                                  next_states, states):
            actions.extend(
                list(self.enumerate_samples_per_state(r, t, c, ns, s)))
        # log_prob -> prob
        probs = [np.exp(score) for score, _, _, _ in actions]
        # normalize
        probs = [p / sum(probs) for p in probs]
        resamples = self.rng.multinomial(n, probs)
        for (score, state, new_state, action), m in zip(actions, resamples):
            for _ in range(m):
                action_sequence = state.state["action_sequence"].clone()
                action_sequence.eval(action)
                x = {key: value for key, value in new_state.items()}
                x["action_sequence"] = action_sequence
                yield SamplerState[Dict[str, Any]](score, x)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.module.load_state_dict(state_dict)

    def to(self, device: torch.device) -> None:
        self.module.to(device)
