import torch
import numpy as np
from typing import List, Union, Callable
from dataclasses import dataclass
from nl2prog.nn.nl2code import Predictor
from nl2prog.language.evaluator import Evaluator
from nl2prog.language.ast import AST
from nl2prog.language.action import NodeConstraint, NodeType
from nl2prog.language.action import ExpandTreeRule
from nl2prog.language.action import Action, ApplyRule, GenerateToken
from nl2prog.language.encoder import Encoder
from nl2prog._utils import TopKElement
from nl2prog.nn.utils.rnn import pad_sequence

"""
True if the argument 0 is subtype of the argument 1
"""
IsSubtype = Callable[[NodeType, NodeType], bool]


@dataclass
class Hypothesis:
    id: int
    parent: Union[int, None]
    score: float
    evaluator: Evaluator
    history: torch.FloatTensor
    h_n: torch.FloatTensor
    c_n: torch.FloatTensor


@dataclass
class Progress:
    id: int
    parent: Union[int, None]
    score: float
    action: Action
    is_complete: bool


@dataclass
class Candidate:
    score: float
    ast: AST


class BeamSearchSynthesizer:
    def __init__(self, beam_size: int,
                 predictor: Predictor, action_sequence_encoder: Encoder,
                 is_subtype: IsSubtype, eps: float = 1e-8,
                 max_steps: Union[int, None] = None):
        """
        Parameters
        ----------
        beam_size: int
            The number of candidates
        predictor: Predictor
            The module to predict the probabilities of actions
        action_sequence_encoder: Encoder
        is_subtype: IsSubType
            The function to check the type relations between 2 node types.
            This returns true if the argument 0 is subtype of the argument 1.
        eps: float
        max_steps: Union[int, None]
        """
        self._beam_size = beam_size
        self._predictor = predictor
        self._hidden_size = predictor.hidden_size  # TODO
        self._action_sequence_encoder = action_sequence_encoder
        self._is_subtype = is_subtype
        self._eps = eps
        self._max_steps = max_steps

    def synthesize(self, query: List[str],
                   query_embedding: torch.FloatTensor):
        """
        Synthesize the program from the query

        Parameters
        ----------
        query: List[str]
            The query
        query_embedding: torch.FloatTensor
            The embedding of query. The shape should be (len(query), *).

        Yields
        ------
        candidates: List[Candidate]
            The candidate of AST
        progress: List[Progress]
            The progress of synthesizing.
        """
        candidates: List[Candidate] = []
        n_ids = 0

        device = list(self._predictor.parameters())[0].device

        # Create initial hypothesis
        h_0 = torch.zeros(self._hidden_size, device=device)  # (hidden_size)
        c_0 = torch.zeros(self._hidden_size, device=device)  # (hidden_size)
        hist_0 = torch.zeros(0, self._hidden_size,
                             device=device)  # (0, hidden_size)
        hs: List[Hypothesis] = \
            [Hypothesis(0, None, 0.0, Evaluator(), hist_0, h_0, c_0)]
        n_ids += 1

        steps = 0
        while len(candidates) < self._beam_size:
            if self._max_steps is not None:
                if steps > self._max_steps:
                    break
                steps += 1

            # Create batch of hypothesis
            query_seq = []
            action = []
            prev_action = []
            hist = []
            h_n = []
            c_n = []
            for elem in hs:
                query_seq.append(query_embedding)
                # (L_a + 1, 6)
                action_tensor = self._action_sequence_encoder.encode(
                    elem.evaluator,
                    query)
                action.append(action_tensor.action[-1]
                              .to(device).view(1, -1))  # (1, 3)
                prev_action.append(
                    action_tensor.previous_action[-1]
                    .to(device).view(1, -1))  # (1, 3)
                hist.append(elem.history.to(device))
                h_n.append(elem.h_n.to(device))
                c_n.append(elem.c_n.to(device))
            query_seq = \
                pad_sequence(query_seq)  # (L_q, len(hs), query_state_size)
            action = pad_sequence(action)  # (1, len(hs), 3)
            prev_action = pad_sequence(prev_action)  # (1, len(hs), 3)
            hist = torch.stack(hist, dim=1)  # (L_a, len(hs), state_size)
            h_n = torch.stack(h_n, dim=0)  # (len(hs), state_size)
            c_n = torch.stack(c_n, dim=0)  # (len(hs), state_size)

            results = self._predictor(query_seq, action, prev_action,
                                      hist, (h_n, c_n))
            rule_pred = results[0].data  # (1, len(hs), n_rules)
            rule_pred = \
                torch.split(rule_pred.reshape(len(hs), -1),
                            1, dim=0)  # (1, n_rules)
            token_pred = results[1].data  # (1, len(hs), n_tokens)
            token_pred = \
                torch.split(token_pred.reshape(len(hs), -1),
                            1, dim=0)  # (1, n_tokens)
            copy_pred = results[2].data  # (1, len(hs), query_length)
            copy_pred = \
                torch.split(copy_pred.reshape(len(hs), -1),
                            1, dim=0)  # (query_length)
            history = results[3]  # (L_a + 1, len(hs), state_size)
            state_size = history.shape[2]
            history = torch.split(history, 1,
                                  dim=1)  # (L_a + 1, 1, state_size)
            history = [x.reshape(-1, state_size)
                       for x in history]  # (L_a + 1, state_size)
            h_n, c_n = results[4]  # (len(hs), state_size)
            h_n = torch.split(h_n, 1, dim=0)  # (1, state_size)
            h_n = [x.view(-1) for x in h_n]  # (state_size)
            c_n = torch.split(c_n, 1, dim=0)  # (,1 state_size)
            c_n = [x.view(-1) for x in c_n]  # (state_size)

            elem_size = self._beam_size - len(candidates)
            topk = TopKElement(elem_size)
            for i, (h, rule, token, copy) in enumerate(zip(hs,
                                                           rule_pred,
                                                           token_pred,
                                                           copy_pred)):
                rule = rule.view(-1)
                token = token.view(-1)
                copy = copy.view(-1)

                # Create hypothesis from h
                head = h.evaluator.head
                if head is None and len(h.evaluator.action_sequence) != 0:
                    continue
                is_token = False
                head_field: Union[NodeType, None] = None
                if head is not None:
                    head_field = \
                        h.evaluator.action_sequence[head.action]\
                        .rule.children[head.field][1]
                    if head_field.constraint == NodeConstraint.Token:
                        is_token = True
                if is_token:
                    # Generate token
                    n_tokens = token.numel()
                    n_words = len(query)
                    token_np = token.detach().cpu().numpy()
                    for j in range(1, n_tokens):  # 0 is UnknownToken
                        x = token_np[j]
                        if x < self._eps:
                            p = np.log(self._eps)
                        else:
                            p = np.log(x)
                        topk.add(h.score + p, (i, "token", j))
                    copy_np = copy.detach().cpu().numpy()
                    for j in range(n_words):
                        x = copy_np[j]
                        if x < self._eps:
                            p = np.log(self._eps)
                        else:
                            p = np.log(x)
                        topk.add(h.score + p, (i, "copy", j))
                else:
                    # Apply rule
                    n_rules = rule.numel()
                    # TODO
                    idx_to_rule = \
                        self._action_sequence_encoder._rule_encoder.vocab
                    rule_np = rule.detach().cpu().numpy()
                    for j in range(1, n_rules):  # 0 is unknown rule
                        x = rule_np[j]
                        if x < self._eps:
                            p = np.log(self._eps)
                        else:
                            p = np.log(x)
                        if isinstance(idx_to_rule[j], ExpandTreeRule):
                            if head_field is None or \
                                    self._is_subtype(idx_to_rule[j].parent,
                                                     head_field):
                                topk.add(h.score + p,
                                         (i, "rule", idx_to_rule[j]))
                        else:
                            # CloseVariadicFieldRule
                            if head_field is not None and \
                                head_field.constraint == \
                                    NodeConstraint.Variadic:
                                topk.add(h.score + p,
                                         (i, "rule", idx_to_rule[j]))

            # Instantiate top-k hypothesis
            hs_new = []
            cs = []
            ps = []
            for score, (i, action, arg) in topk.elements:
                h = hs[i]
                id = n_ids
                n_ids += 1
                evaluator = h.evaluator.clone()
                if action == "rule":
                    evaluator.eval(ApplyRule(arg))
                elif action == "token":
                    # Token
                    # TODO
                    token = \
                        self._action_sequence_encoder._token_encoder.decode(
                            torch.LongTensor([arg]))
                    evaluator.eval(GenerateToken(token))
                else:
                    evaluator.eval(GenerateToken(query[arg]))

                if evaluator.head is None:
                    # Complete
                    c = Candidate(score, evaluator.generate_ast())
                    cs.append(c)
                    candidates.append(c)
                    ps.append(Progress(id, h.id, score,
                                       evaluator.action_sequence[-1], True))
                else:
                    hs_new.append(Hypothesis(
                        id, h.id, score, evaluator,
                        history[i], h_n[i], c_n[i]
                    ))
                    ps.append(Progress(id, h.id, score,
                                       evaluator.action_sequence[-1], False))

            hs = hs_new
            yield cs, ps
            if len(hs) == 0:
                break
