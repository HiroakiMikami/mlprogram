import torch
import numpy as np
from typing import List, Union, Callable
from dataclasses import dataclass
from nl2prog.nn.nl2code import Predictor
from nl2prog.language.action \
    import ActionOptions
from nl2prog.utils.data.nl2code import Encoder, Query
from nl2prog.utils \
    import BeamSearchSynthesizer as BaseBeamSearchSynthesizer, \
    IsSubtype, LazyLogProbability
from nl2prog.nn.utils.rnn import pad_sequence


@dataclass
class State:
    query: List[str]
    query_tensor: torch.FloatTensor
    history: torch.FloatTensor
    h_n: torch.FloatTensor
    c_n: torch.FloatTensor


class BeamSearchSynthesizer(BaseBeamSearchSynthesizer):
    def __init__(self, beam_size: int,
                 tokenizer: Callable[[str], Query],
                 query_encoder: Callable[[torch.LongTensor],
                                         torch.FloatTensor],
                 predictor: Predictor, encoder: Encoder,
                 is_subtype: IsSubtype, options=ActionOptions(True, True),
                 eps: float = 1e-8,
                 max_steps: Union[int, None] = None):
        """
        Parameters
        ----------
        beam_size: int
            The number of candidates
        tokenize: Callable[[str], Query]
        query_encoder: Callble[[torch.LongTensor], torch.FloatTensor]
            The encoder module
        predictor: Predictor
            The module to predict the probabilities of actions
        encoder: Encoder
        is_subtype: IsSubType
            The function to check the type relations between 2 node types.
            This returns true if the argument 0 is subtype of the argument 1.
        options: ActionOptions
        eps: float
        max_steps: Union[int, None]
        """
        device = list(predictor.parameters())[0].device
        hidden_size = predictor.hidden_size

        def initialize(query: str):
            query = tokenizer(query)
            query_tensor = \
                encoder.annotation_encoder.batch_encode(query.query_for_dnn)
            query_tensor = query_encoder(query_tensor)

            # Create initial hypothesis
            h_0 = torch.zeros(hidden_size, device=device)  # (hidden_size)
            c_0 = torch.zeros(hidden_size, device=device)  # (hidden_size)
            hist_0 = torch.zeros(0, hidden_size,
                                 device=device)  # (0, hidden_size)
            return State(query.query_for_synth, query_tensor, hist_0, h_0, c_0)

        def batch_update(hs):
            # Create batch of hypothesis
            query_seq = []
            action = []
            prev_action = []
            hist = []
            h_n = []
            c_n = []
            for h in hs:
                query_seq.append(h.state.query_tensor)
                # (L_a + 1, 4)
                a = encoder.action_sequence_encoder.encode_action(
                    h.evaluator, h.state.query)
                # (L_a + 1, 3)
                p = encoder.action_sequence_encoder.encode_parent(h.evaluator)
                # (1, 3)
                action.append(
                    torch.cat([a[-1, 0].view(1, -1),
                               p[-1, 1:3].view(1, -1)], dim=1).to(device)
                )
                if a.shape[0] == 1:
                    prev_action.append(
                        a[-1, 1:].to(device).view(1, -1))  # (1, 3)
                else:
                    prev_action.append(
                        a[-2, 1:].to(device).view(1, -1))  # (1, 3)
                hist.append(h.state.history.to(device))
                h_n.append(h.state.h_n.to(device))
                c_n.append(h.state.c_n.to(device))
            query_seq = \
                pad_sequence(query_seq)  # (L_q, len(hs), query_state_size)
            action = pad_sequence(action)  # (1, len(hs), 3)
            prev_action = pad_sequence(prev_action)  # (1, len(hs), 3)
            hist = torch.stack(hist, dim=1)  # (L_a, len(hs), state_size)
            h_n = torch.stack(h_n, dim=0)  # (len(hs), state_size)
            c_n = torch.stack(c_n, dim=0)  # (len(hs), state_size)

            results = predictor(query_seq, action, prev_action,
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

            retval = []
            for i, (h, hist, h_, c_, rule, token, copy) in enumerate(zip(
                    hs, history, h_n, c_n, rule_pred, token_pred, copy_pred)):
                state = State(
                    h.state.query, h.state.query_tensor, hist, h_, c_)

                rule = rule.view(-1)
                token = token.view(-1)
                copy = copy.view(-1)

                def get_rule_prob():
                    n_rules = rule.numel()
                    # TODO
                    idx_to_rule = \
                        encoder.action_sequence_encoder._rule_encoder.vocab
                    rule_np = rule.detach().cpu().numpy()
                    retval = {}
                    for j in range(1, n_rules):  # 0 is unknown rule
                        p = rule_np[j]
                        if p < eps:
                            retval[idx_to_rule[j]] = np.log(eps)
                        else:
                            retval[idx_to_rule[j]] = np.log(p)
                    return retval

                def get_token_prob():
                    probs = {}
                    n_tokens = token.numel()
                    n_words = len(h.state.query)
                    token_np = token.detach().cpu().numpy()
                    for j in range(1, n_tokens):  # 0 is UnknownToken
                        p = token_np[j]
                        t = \
                            encoder.action_sequence_encoder. \
                            _token_encoder.decode(
                                torch.LongTensor([j]))
                        probs[t] = (probs.get(t) or 0) + p
                    copy_np = copy.detach().cpu().numpy()
                    for j in range(n_words):
                        p = copy_np[j]
                        t = h.state.query[j]
                        probs[t] = (probs.get(t) or 0) + p

                    log_prob = {}
                    for t, p in probs.items():
                        if p < eps:
                            log_prob[t] = np.log(eps)
                        else:
                            log_prob[t] = np.log(p)
                    return log_prob
                prob = LazyLogProbability(get_rule_prob, get_token_prob)
                retval.append((state, prob))
            return retval

        super(BeamSearchSynthesizer, self).__init__(
            beam_size, initialize, batch_update, is_subtype, options,
            max_steps)
