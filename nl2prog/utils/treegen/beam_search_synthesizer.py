import torch
from torchnlp.encoders import LabelEncoder
import numpy as np
from typing import List, Callable, Optional
from dataclasses import dataclass
from nl2prog.nn.treegen \
    import ActionSequenceReader, Decoder, Predictor, NLReader
from nl2prog.language.action import ActionOptions
from nl2prog.encoders import ActionSequenceEncoder
from nl2prog.utils \
    import BeamSearchSynthesizer as BaseBeamSearchSynthesizer, \
    IsSubtype, LazyLogProbability, Query
from nl2prog.nn.utils.rnn import pad_sequence


@dataclass
class State:
    query: List[str]
    nl_feature: torch.FloatTensor


class BeamSearchSynthesizer(BaseBeamSearchSynthesizer):
    def __init__(self, beam_size: int,
                 tokenizer: Callable[[str], Query],
                 nl_reader: NLReader, ast_reader: ActionSequenceReader,
                 decoder: Decoder,
                 predictor: Predictor, word_encoder: LabelEncoder,
                 char_encoder: LabelEncoder,
                 action_sequence_encoder: ActionSequenceEncoder,
                 max_word_num: int, max_arity: int,
                 is_subtype: IsSubtype,
                 options: ActionOptions = ActionOptions(False, False),
                 eps: float = 1e-8,
                 max_steps: Optional[int] = None):
        """
        Parameters
        ----------
        beam_size: int
            The number of candidates
        tokenizer: Callable[[str], Query]
        nl_reader:
            The encoder module
        ast_reader: ASTReader
        decoder: Decoder
        predictor: Predictor
            The module to predict the probabilities of actions
        word_encoder: LabelEncoder
        char_encoder: LabelEncoder
        action_sequence_encoder: ActionSequenceEncoder
        max_word_num: int
        max_arity: int
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
        self.tokenizer = tokenizer
        self.nl_reader = nl_reader
        self.ast_reader = ast_reader
        self.decoder = decoder
        self.predictor = predictor
        self.word_encoder = word_encoder
        self.char_encoder = char_encoder
        self.action_sequence_encoder = action_sequence_encoder
        self.max_word_num = max_word_num
        self.max_arity = max_arity
        self.eps = eps

    def initialize(self, query: str):
        query = self.tokenizer(query)
        word_query = \
            self.word_encoder.batch_encode(query.query_for_dnn)
        word_query = word_query.to(self.device)
        word_query = pad_sequence([word_query])
        char_query = torch.ones(len(query.query_for_dnn),
                                self.max_word_num).to(self.device).long() * -1
        for i, word in enumerate(query.query_for_dnn):
            chars = self.char_encoder.batch_encode(word)
            length = min(self.max_word_num, len(chars))
            char_query[i, :length] = \
                self.char_encoder.batch_encode(word)[:length]
        char_query = pad_sequence([char_query])
        nl_feature, _ = self.nl_reader((word_query, char_query))
        nl_feature = nl_feature.data
        L = nl_feature.shape[0]
        nl_feature = nl_feature.view(L, -1)

        # Create initial hypothesis
        return State(query.query_for_synth, nl_feature)

    def batch_update(self, hs):
        # Create batch of hypothesis
        query_seq = []
        action_seq = []
        rule_action_seq = []
        depth = []
        matrix = []
        for h in hs:
            query_seq.append(h.state.nl_feature)
            a = self.action_sequence_encoder.encode_action(
                h.evaluator, h.state.query)
            action_seq.append(a[:-1, 1:])
            rule_action_seq.append(
                self.action_sequence_encoder.encode_each_action(
                    h.evaluator, h.state.query, self.max_arity))
            d, m = self.action_sequence_encoder.encode_tree(h.evaluator)
            depth.append(d)
            matrix.append(m)
        query_seq = pad_sequence(query_seq, padding_value=-1)
        query_seq.data = query_seq.data.to(self.device)
        query_seq.mask = query_seq.mask.to(self.device)
        action = pad_sequence(action_seq, padding_value=-1)
        action.data = action.data.to(self.device)
        action.mask = action.mask.to(self.device)
        rule_action = pad_sequence(rule_action_seq, padding_value=-1)
        rule_action.data = rule_action.data.to(self.device)
        rule_action.mask = rule_action.mask.to(self.device)
        depth = torch.stack(depth).reshape(len(hs), -1).permute(1, 0)
        depth = depth.to(self.device)
        matrix = torch.stack(matrix)
        matrix = matrix.to(self.device)

        with torch.no_grad():
            ast_feature = self.ast_reader((action, rule_action, depth, matrix))
            feature, _ = \
                self.decoder(action, query_seq, None, ast_feature, None)
            results = self.predictor(query_seq, feature)
        # (len(hs), n_rules)
        rule_pred = results[0].data[-1, :, :].cpu().reshape(len(hs), -1)
        # (len(hs), n_tokens)
        token_pred = results[1].data[-1, :, :].cpu().reshape(len(hs), -1)
        # (len(hs), query_length)
        copy_pred = results[2].data[-1, :, :].cpu().reshape(len(hs), -1)

        retval = []
        for i, h in enumerate(hs):
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
            retval.append((h.state, prob))
        return retval
