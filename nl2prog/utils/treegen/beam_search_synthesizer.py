import torch
from torchnlp.encoders import LabelEncoder
import numpy as np
from typing import List, Callable, Optional
from dataclasses import dataclass
from nl2prog.nn.treegen import ASTReader, Decoder, Predictor
from nl2prog.language.action import ActionOptions
from nl2prog.encoders import ActionSequenceEncoder
from nl2prog.utils \
    import BeamSearchSynthesizer as BaseBeamSearchSynthesizer, \
    IsSubtype, LazyLogProbability, Query
from nl2prog.nn.utils.rnn import pad_sequence, PaddedSequenceWithMask


@dataclass
class State:
    query: List[str]
    nl_feature: torch.FloatTensor


class BeamSearchSynthesizer(BaseBeamSearchSynthesizer):
    def __init__(self, beam_size: int,
                 tokenizer: Callable[[str], Query],
                 nl_reader: Callable[[PaddedSequenceWithMask, torch.Tensor],
                                     PaddedSequenceWithMask],
                 ast_reader: ASTReader, decoder: Decoder,
                 predictor: Predictor, word_encoder: LabelEncoder,
                 char_encoder: LabelEncoder,
                 action_sequence_encoder: ActionSequenceEncoder,
                 max_word_num: int, max_arity: int,
                 is_subtype: IsSubtype, options=ActionOptions(False, False),
                 eps: float = 1e-8,
                 max_steps: Optional[int] = None):
        """
        Parameters
        ----------
        beam_size: int
            The number of candidates
        tokenize: Callable[[str], Query]
        nl_reader: Callble[[PaddedSequenceWithMask, torch.Tensor],
                           PaddedSequenceWithMask]
            The encoder module
        ast_reader: ASTReader
        decoder: Decoder
        predictor: Predictor
            The module to predict the probabilities of actions
        word_encoder: LabelEncoder
        char_encoder: LabelEncoder
        action_seqneuce_encoder: ActionSequenceEncoder
        max_word_num: int
        max_arity: int
        is_subtype: IsSubType
            The function to check the type relations between 2 node types.
            This returns true if the argument 0 is subtype of the argument 1.
        options: ActionOptions
        eps: float
        max_steps: Optional[int]
        """
        device = list(predictor.parameters())[0].device

        def initialize(query: str):
            query = tokenizer(query)
            word_query = \
                word_encoder.batch_encode(query.query_for_dnn)
            word_query = word_query.to(device)
            word_query = pad_sequence([word_query])
            char_query = torch.ones(len(query.query_for_dnn),
                                    max_word_num).to(device) * -1
            for i, word in enumerate(query.query_for_dnn):
                char_query[i, :] = \
                    char_encoder.batch_encode(word)[:max_word_num]
            # TODO embedding
            nl_feature = nl_reader(word_query, char_query).data
            L = nl_feature.shape[0]
            nl_feature = nl_feature.view(L, -1)

            # Create initial hypothesis
            return State(query.query_for_synth, nl_feature)

        def batch_update(hs):
            # Create batch of hypothesis
            query_seq = []
            action = []
            prev_action = []
            hist = []
            h_n = []
            c_n = []
            for h in hs:
                # TODO create tensors
                query_seq.append(h.state.nl_feature)
                # (L_a + 1, 4)
                a = action_sequence_encoder.encode_action(
                    h.evaluator, h.state.query)
                # (L_a + 1, 3)
                p = action_sequence_encoder.encode_parent(h.evaluator)
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
            query_seq = \
                pad_sequence(query_seq)  # (L_q, len(hs), query_state_size)
            action = pad_sequence(action)  # (1, len(hs), 3)
            prev_action = pad_sequence(prev_action)  # (1, len(hs), 3)

            with torch.no_grad():
                # TODO use ast_reader and decoder
                results = predictor(query_seq, action, prev_action,
                                    hist, (h_n, c_n))
            # (len(hs), n_rules)
            rule_pred = results[0].data.cpu().reshape(len(hs), -1)
            # (len(hs), n_tokens)
            token_pred = results[1].data.cpu().reshape(len(hs), -1)
            # (len(hs), query_length)
            copy_pred = results[2].data.cpu().reshape(len(hs), -1)

            retval = []
            for i, h in enumerate(hs):
                class Functions:
                    def __init__(self, i):
                        self.i = i

                    def get_rule_prob(self):
                        # TODO
                        idx_to_rule = \
                            action_sequence_encoder._rule_encoder.vocab
                        retval = {}
                        # 0 is unknown rule
                        for j in range(1, rule_pred.shape[1]):
                            p = rule_pred[self.i, j].item()
                            if p < eps:
                                retval[idx_to_rule[j]] = np.log(eps)
                            else:
                                retval[idx_to_rule[j]] = np.log(p)
                        return retval

                    def get_token_prob(self):
                        probs = {}
                        n_words = len(h.state.query)
                        for j in range(1,
                                       token_pred.shape[1]
                                       ):  # 0 is UnknownToken
                            p = token_pred[self.i, j].item()
                            t = \
                                action_sequence_encoder._token_encoder.decode(
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
                            if p < eps:
                                log_prob[t] = np.log(eps)
                            else:
                                log_prob[t] = np.log(p)
                        return log_prob
                funcs = Functions(i)
                prob = LazyLogProbability(funcs.get_rule_prob,
                                          funcs.get_token_prob)
                retval.append((h.state, prob))
            return retval

        super(BeamSearchSynthesizer, self).__init__(
            beam_size, initialize, batch_update, is_subtype, options,
            max_steps)
