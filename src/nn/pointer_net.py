import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.initializer as I

from .layers.core import dense


def pointer_net(query_embed, query_embed_mask, decoder_states, hidden_dim):
    """
    query_embed: (batch_size, max_query_length, E1)
    decoder_states: (batch_size, max_action_length, E2)
    """
    with nn.parameter_scope("pointer_net"):
        batch_size, max_query_length, _ = query_embed.shape
        _, max_action_length, _ = decoder_states.shape
        with nn.parameter_scope("layer1_input"):
            query_embed_trans = dense(
                query_embed, hidden_dim, base_axis=2, activation=lambda x: x)
        with nn.parameter_scope("layer1_h"):
            h_trans = dense(
                decoder_states,
                hidden_dim,
                base_axis=2,
                activation=lambda x: x)

        query_embed_trans = F.reshape(
            query_embed_trans, (batch_size, 1, max_query_length, hidden_dim))
        query_embed_trans = F.broadcast(
            query_embed_trans,
            (batch_size, max_action_length, max_query_length, hidden_dim))
        h_trans = F.reshape(h_trans,
                            (batch_size, max_action_length, 1, hidden_dim))
        h_trans = F.broadcast(
            h_trans,
            (batch_size, max_action_length, max_query_length, hidden_dim))
        dense1_trans = F.tanh(query_embed_trans + h_trans)

        with nn.parameter_scope("layer2"):
            # scores: (batch_size, max_action_length, max_query_length, 1)
            scores = dense(
                dense1_trans, 1, base_axis=3, activation=lambda x: x)
        # scores: (batch_size, max_action_length, max_query_length)
        scores = F.reshape(scores,
                           (batch_size, max_action_length, max_query_length))
        scores = F.exp(scores - F.max(scores, axis=2, keepdims=True))
        mask = F.reshape(query_embed_mask, (batch_size, 1, max_query_length))
        mask = F.broadcast(mask,
                           (batch_size, max_action_length, max_query_length))
        scores = scores * mask
        scores = scores / F.sum(scores, axis=2, keepdims=True)

        return scores
