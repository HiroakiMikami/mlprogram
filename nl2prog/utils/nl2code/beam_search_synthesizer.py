import torch
import numpy as np
from typing import List, Callable, Optional, Tuple, Any
from dataclasses import dataclass
from nl2prog.nn.nl2code import ActionSequenceReader, Decoder, Predictor
from nl2prog.language.action import ActionOptions
from nl2prog.language.evaluator import Evaluator
from nl2prog.encoders import ActionSequenceEncoder
from nl2prog.utils \
    import BeamSearchSynthesizer as BaseBeamSearchSynthesizer, \
    IsSubtype, LazyLogProbability
from nl2prog.nn.utils.rnn import pad_sequence, PaddedSequenceWithMask


@dataclass
class State:
    query: List[str]
    query_tensor: torch.FloatTensor
    history: torch.FloatTensor
    h_n: torch.FloatTensor
    c_n: torch.FloatTensor


class BeamSearchSynthesizer(BaseBeamSearchSynthesizer):
    def __init__(self, beam_size: int,
                 transform_input: Callable[[Any], Tuple[List[str], Any]],
                 transform_evaluator: Callable[[Evaluator, List[str]],
                                               Optional[Tuple[torch.Tensor,
                                                              torch.Tensor]]],
                 collate_input, collate_action_sequence, collate_query,
                 nl_reader: Callable[[PaddedSequenceWithMask],
                                     Tuple[PaddedSequenceWithMask,
                                           Optional[torch.Tensor]]],
                 ast_reader: ActionSequenceReader, decoder: Decoder,
                 predictor: Predictor,
                 action_sequence_encoder: ActionSequenceEncoder,
                 is_subtype: IsSubtype,
                 options: ActionOptions = ActionOptions(True, True),
                 eps: float = 1e-8,
                 max_steps: Optional[int] = None):
        """
        Parameters
        ----------
        beam_size: int
            The number of candidates
        nl_reader:
            The encoder module
        ast_reader:
        decoder:
        predictor: Predictor
            The module to predict the probabilities of actions
        action_seqeunce_encoder: ActionSequenceEncoder
        is_subtype: IsSubType
            The function to check the type relations between 2 node types.
            This returns true if the argument 0 is subtype of the argument 1.
        options: ActionOptions
        eps: float
        max_steps: Optional[int]
        """
        super(BeamSearchSynthesizer, self).__init__(
            beam_size, is_subtype, options, max_steps)
        self.device = list(predictor.parameters())[0].device
        self.transform_input = transform_input
        self.transform_evaluator = transform_evaluator
        self.collate_input = collate_input
        self.collate_action_sequence = collate_action_sequence
        self.collate_query = collate_query
        self.nl_reader = nl_reader
        self.ast_reader = ast_reader
        self.decoder = decoder
        self.predictor = predictor
        self.action_sequence_encoder = action_sequence_encoder
        self.eps = eps
        self.hidden_size = predictor.hidden_size

    def initialize(self, query: str):
        query_for_synth, query_tensor = self.transform_input(query)
        query_tensor = self.collate_input([query_tensor])
        query_tensor, _ = self.nl_reader(query_tensor)
        query_tensor = query_tensor.data
        L = query_tensor.shape[0]
        query_tensor = query_tensor.view(L, -1)

        # Create initial hypothesis
        h_0 = torch.zeros(self.hidden_size,
                          device=self.device)  # (hidden_size)
        c_0 = torch.zeros(self.hidden_size,
                          device=self.device)  # (hidden_size)
        hist_0 = torch.zeros(1, self.hidden_size,
                             device=self.device)  # (1, hidden_size)

        return State(query_for_synth, query_tensor, hist_0, h_0, c_0)

    def batch_update(self, hs):
        # Create batch of hypothesis
        query_seq = []
        action_sequences = []
        queries = []
        hist = []
        h_n = []
        c_n = []
        for h in hs:
            query_seq.append(h.state.query_tensor)
            action_sequence, query = \
                self.transform_evaluator(h.evaluator, h.state.query)
            action_sequences.append(action_sequence)
            queries.append(query)
            hist.append(h.state.history.to(self.device))
            h_n.append(h.state.h_n.to(self.device))
            c_n.append(h.state.c_n.to(self.device))
        query_seq = \
            pad_sequence(query_seq)  # (L_q, len(hs), query_state_size)
        action_sequences = self.collate_action_sequence(action_sequences)
        queries = self.collate_query(queries)
        hist = torch.stack(hist, dim=1)  # (L_a, len(hs), state_size)
        h_n = torch.stack(h_n, dim=0)  # (len(hs), state_size)
        c_n = torch.stack(c_n, dim=0)  # (len(hs), state_size)

        with torch.no_grad():
            ast_feature = self.ast_reader(action_sequences)
            feature, state = self.decoder(queries, query_seq, None,
                                          ast_feature,
                                          (hist, h_n, c_n))
            results = self.predictor(query_seq, feature)
        # (len(hs), n_rules)
        rule_pred = results[0].data.cpu().reshape(len(hs), -1)
        # (len(hs), n_tokens)
        token_pred = results[1].data.cpu().reshape(len(hs), -1)
        # (len(hs), query_length)
        copy_pred = \
            results[2].data.cpu().reshape(len(hs), -1)
        history, h_n, c_n = state
        state_size = history.shape[2]
        history = torch.split(history, 1,
                              dim=1)  # (L_a + 1, 1, state_size)
        history = [x.reshape(-1, state_size)
                   for x in history]  # (L_a + 1, state_size)
        h_n = torch.split(h_n, 1, dim=0)  # (1, state_size)
        h_n = [x.view(-1) for x in h_n]  # (state_size)
        c_n = torch.split(c_n, 1, dim=0)  # (,1 state_size)
        c_n = [x.view(-1) for x in c_n]  # (state_size)

        retval = []
        for i, (h, hist_, h_, c_) in enumerate(zip(hs, history, h_n, c_n)):
            state = State(h.state.query, h.state.query_tensor, hist_, h_, c_)

            class Functions:
                def __init__(self, i, action_sequence_encoder, eps):
                    self.i = i
                    self.action_sequence_encoder = action_sequence_encoder
                    self.eps = eps

                def get_rule_prob(self):
                    # TODO
                    idx_to_rule = \
                        self.action_sequence_encoder._rule_encoder.vocab
                    retval = {}
                    # 0 is unknown rule
                    for j in range(1, rule_pred.shape[1]):
                        p = rule_pred[self.i, j].item()
                        if p < self.eps:
                            retval[idx_to_rule[j]] = np.log(self.eps)
                        else:
                            retval[idx_to_rule[j]] = np.log(p)
                    return retval

                def get_token_prob(self):
                    probs = {}
                    n_words = len(h.state.query)
                    # 0 is UnknownToken
                    for j in range(1, token_pred.shape[1]):
                        p = token_pred[self.i, j].item()
                        t = \
                            self.action_sequence_encoder._token_encoder.decode(
                                torch.LongTensor([j]))
                        if t in probs:
                            probs[t] = probs.get(t) + p
                        else:
                            probs[t] = p
                    for j in range(n_words):
                        p = copy_pred[self.i, j].item()
                        t = h.state.query[j]
                        if t in probs:
                            probs[t] = probs.get(t) + p
                        else:
                            probs[t] = p

                    log_prob = {}
                    for t, p in probs.items():
                        if p < self.eps:
                            log_prob[t] = np.log(self.eps)
                        else:
                            log_prob[t] = np.log(p)
                    return log_prob
            funcs = Functions(i, self.action_sequence_encoder, self.eps)
            prob = LazyLogProbability(funcs.get_rule_prob,
                                      funcs.get_token_prob)
            retval.append((state, prob))
        return retval
