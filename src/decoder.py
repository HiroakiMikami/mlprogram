import nnabla as nn
import nnabla.functions as F
import nnabla.logger as logger
import numpy as np
from typing import NamedTuple, Union, Dict, Any

import src.model as model
from .utils import TopKElement
from .grammar import *
from .annotation import Annotation, EncoderInput
from .python.grammar import is_builtin_node_type


class Hypothesis(NamedTuple):
    score: float
    sequence: Sequence
    orig_sequence: Any
    previous_action_embed: Union[None, np.array]
    state: np.array
    cell: np.array
    hist: np.array


class Decoder:
    def __init__(self, beam_size: int, max_query_length: int,
                 max_action_length: int, word_to_id: Dict[str, id],
                 grammar: Grammar, embedding_size, node_type_embedding_size,
                 lstm_state_size, hidden_state_size, dropout):
        self._beam_size = beam_size
        self._lstm_state_size = lstm_state_size
        self._max_query_length = max_query_length
        self._max_action_length = max_action_length
        self._grammar = grammar

        # encoder
        self._query = nn.Variable((1, max_query_length), need_grad=False)
        query_embed, query_embed_mask = model.encoder(
            self._query,
            len(word_to_id),
            embedding_size,
            lstm_state_size,
            dropout=dropout,
            train=False)
        self._query_embed = F.broadcast(
            query_embed, (beam_size, max_query_length, lstm_state_size))
        self._query_embed_mask = F.broadcast(query_embed_mask,
                                             (beam_size, max_query_length))

        self._action = nn.Variable((self._beam_size, 2, 3))
        self._action_type = nn.Variable((self._beam_size, 2, 3))
        self._parent_rule = nn.Variable((self._beam_size, 2))
        self._parent_index = nn.Variable((self._beam_size, 2))
        self._node_type = nn.Variable((self._beam_size, 2))
        self._previous_action_embed = nn.Variable((self._beam_size, 1,
                                                   embedding_size))
        self._initial_state = nn.Variable((self._beam_size, lstm_state_size))
        self._initial_cell = nn.Variable((self._beam_size, lstm_state_size))

        class DecoderNN(NamedTuple):
            hist: nn.Variable
            action_embed: nn.Variable
            rule_prob: nn.Variable
            gen_prob: nn.Variable
            token_prob: nn.Variable
            copy_prob: nn.Variable
            decoder_states: nn.Variable
            decoder_cell: nn.Variable
            new_hist: nn.Variable

        self._decoders = []
        for t in range(self._max_action_length):
            hist = nn.Variable((self._beam_size, t + 1, lstm_state_size),
                               need_grad=False)
            action_embed, decoder_states, decoder_cell, ctx_vector, decoder_mask, new_hist = model.decoder(
                self._action.get_unlinked_variable(),
                self._action_type.get_unlinked_variable(),
                self._node_type.get_unlinked_variable(),
                self._parent_rule.get_unlinked_variable(),
                self._parent_index.get_unlinked_variable(),
                self._query_embed.get_unlinked_variable(),
                self._query_embed_mask.get_unlinked_variable(),
                len(grammar.rule_to_id),
                len(grammar.token_to_id),
                len(grammar.node_type_to_id),
                embedding_size,
                node_type_embedding_size,
                lstm_state_size,
                hidden_state_size,
                previous_action_embed=self._previous_action_embed.
                get_unlinked_variable(),
                initial_state=self._initial_state.get_unlinked_variable(),
                initial_cell=self._initial_cell.get_unlinked_variable(),
                hist=hist,
                dropout=dropout,
                train=False)
            rule_prob, gen_prob, token_prob, copy_prob = model.pred(
                decoder_states, ctx_vector,
                self._query_embed.get_unlinked_variable(),
                self._query_embed_mask.get_unlinked_variable(),
                len(grammar.rule_to_id), len(grammar.token_to_id),
                embedding_size, hidden_state_size)
            self._decoders.append(
                DecoderNN(hist, action_embed, rule_prob, gen_prob, token_prob,
                          copy_prob, decoder_states, decoder_cell, new_hist))

    def decode(self, annotation: Annotation, query: EncoderInput):
        lstm_state_size = self._lstm_state_size
        grammar = self._grammar
        length = min(self._max_query_length, len(annotation.query))

        # word in tokens
        w = {}
        known_words_indexes = set()
        for i, word in enumerate(annotation.query[:length]):
            if word in annotation.mappings:
                word = annotation.mappings[word]
            if word in grammar.token_to_id:
                known_words_indexes.add(i)
                if not (word in w):
                    w[word] = []
                w[word].append(i)

        # Encode query
        import time
        z = time.time()
        self._query.data.zero()
        self._query.d[0, :length] = query.query[:length]
        F.sink(self._query_embed, self._query_embed_mask).forward()

        # Root hypothesis
        completed = []
        Q = [
            Hypothesis(0, [], [], None, np.zeros((lstm_state_size)),
                       np.zeros((lstm_state_size)),
                       np.zeros((1, lstm_state_size)))
        ]

        for t in range(self._max_action_length):
            if self._beam_size - len(completed) <= 0:
                break

            elem_size = self._beam_size - len(completed)
            Q_ = TopKElement(elem_size)

            def create_new_hypothesis(b, p, s, o):
                def f():
                    return Hypothesis(
                        p, s, o,
                        self._decoders[t].action_embed.d[b, -1::, :].copy(),
                        self._decoders[t].decoder_states.d[b, 0, :].copy(),
                        self._decoders[t].decoder_cell.d[b, 0, :].copy(),
                        self._decoders[t].new_hist.d[b, 0:-1, :].copy())

                return f

            self._action.data.zero()
            self._action_type.data.zero()
            self._parent_rule.data.zero()
            self._parent_index.data.zero()
            self._node_type.data.zero()
            self._previous_action_embed.data.zero()
            self._initial_state.data.zero()
            self._initial_cell.data.zero()
            self._decoders[t].hist.data.zero()

            node_types = [None] * self._beam_size
            valid = [True] * self._beam_size
            import time
            z = time.time()
            for b, h in enumerate(Q):
                s = h.sequence.copy()
                decoder_input = to_decoder_input(s, annotation, grammar)
                if decoder_input is None:
                    valid[b] = False
                    continue
                if decoder_input[0] is None:
                    completed.append(h)
                    valid[b] = False
                    continue
                next_node_type, pindex = decoder_input[0]
                node_types[b] = next_node_type

                # Previous action
                if not (h.previous_action_embed is None):
                    if h.orig_sequence[-1][0] == "ApplyRule":
                        # ApplyRule
                        self._action_type.d[b, 0, 0] = 1
                        self._action.d[b, 0, 0] = h.orig_sequence[-1][1]
                    elif h.orig_sequence[-1][0] == "CopyToken":
                        # CopyToken
                        self._action_type.d[b, 0, 2] = 1
                        self._action.d[b, 0, 2] = h.orig_sequence[-1][1]
                    else:
                        # GenToken
                        self._action_type.d[b, 0, 1] = 1
                        self._action.d[b, 0, 1] = grammar.token_to_id[
                            h.orig_sequence[-1][1]]
                    self._parent_rule.d[b, 0] = decoder_input[1].parent_action[
                        -1]
                    self._parent_index.d[
                        b, 0] = decoder_input[1].parent_index[-1] + 1
                    self._node_type.d[b, 0] = decoder_input[1].node_type[-1]
                    self._previous_action_embed.d[
                        b, :] = h.previous_action_embed

                # Current action
                if is_builtin_node_type(node_types[b]):
                    # GenToken
                    self._action_type.d[b, 1, 2] = 1
                else:
                    # ApplyRule
                    self._action_type.d[b, 1, 0] = 1
                if pindex >= 0:
                    self._parent_rule.d[b, 1] = decoder_input[1].action[pindex,
                                                                        0]
                self._parent_index.d[b, 1] = pindex + 1
                self._node_type.d[b, 1] = grammar.node_type_to_id[
                    node_types[b]]

                self._initial_state.d[b, :] = h.state
                self._initial_cell.d[b, :] = h.cell
                self._decoders[t].hist.d[b, :] = h.hist
            z = time.time()
            F.sink(self._decoders[t].action_embed,
                   self._decoders[t].decoder_states,
                   self._decoders[t].decoder_cell, self._decoders[t].new_hist,
                   self._decoders[t].rule_prob, self._decoders[t].gen_prob,
                   self._decoders[t].token_prob,
                   self._decoders[t].copy_prob).forward()

            for b, (n, h) in enumerate(zip(node_types, Q)):
                if not valid[b]:
                    continue

                if is_builtin_node_type(n):
                    token_prob_arr = self._decoders[t].token_prob.d[b, 1, :]
                    token_prob_arr *= self._decoders[t].gen_prob.d[b, 1, 0]
                    token_prob_arr = token_prob_arr.tolist()
                    copy_prob_arr = self._decoders[t].copy_prob.d[b, 1, :min(
                        len(annotation.query), self._max_query_length)]
                    copy_prob_arr *= self._decoders[t].gen_prob.d[b, 1, 1]
                    copy_prob_arr.tolist()

                    # Add prob
                    for word, indexes in w.items():
                        for i in indexes:
                            token_prob_arr[
                                grammar.token_to_id[word]] += copy_prob_arr[i]

                    # GenToken
                    # gen
                    token_prob_arr = list(enumerate(token_prob_arr))
                    token_prob_arr = sorted(
                        token_prob_arr, key=lambda x: x[1], reverse=True)
                    for i, p in token_prob_arr[:elem_size]:
                        token = grammar.id_to_token[i]
                        p = np.log(p)
                        s2 = h.sequence.copy()
                        s2.append(token)
                        os2 = h.orig_sequence.copy()
                        os2.append(["GenToken", token])
                        Q_.push(h.score + p,
                                create_new_hypothesis(b, h.score + p, s2, os2))
                    # copy
                    copy_prob_arr = list(enumerate(copy_prob_arr))
                    copy_prob_arr = list(
                        filter(lambda x: not (x[0] in known_words_indexes),
                               copy_prob_arr))
                    copy_prob_arr = sorted(
                        copy_prob_arr, key=lambda x: x[1], reverse=True)
                    for i, p in copy_prob_arr[:elem_size]:
                        token = annotation.query[i]
                        if token in annotation.mappings:
                            token = annotation.mappings[token]

                        p = np.log(p)
                        s2 = h.sequence.copy()
                        s2.append(token)
                        os2 = h.orig_sequence.copy()
                        os2.append(["CopyToken", i])
                        Q_.push(h.score + p,
                                create_new_hypothesis(b, h.score + p, s2, os2))

                else:
                    # ApplyRule
                    for i in range(len(grammar.rule_to_id)):
                        rule = grammar.id_to_rule[i]

                        if rule.parent == n:
                            p = np.log(self._decoders[t].rule_prob.d[b, 1, i])
                        else:
                            continue
                        s2 = h.sequence.copy()
                        s2.append(rule)
                        os2 = h.orig_sequence.copy()
                        os2.append(["ApplyRule", i])
                        Q_.push(h.score + p,
                                create_new_hypothesis(b, h.score + p, s2, os2))
            # sort by score
            Q = list(map(lambda x: x[2], Q_.queue))  # Get top-k hypothesis
        # sort by score
        Q = list(sorted(completed, key=lambda h: h.score))
        if len(Q) == 0:
            return None
        return Q[-1]
