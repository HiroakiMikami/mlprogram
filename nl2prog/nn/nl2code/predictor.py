import torch
import torch.nn as nn
from typing import Tuple

from nl2prog.nn.nl2code import Decoder
from nl2prog.nn.pointer_net import PointerNet
from nl2prog.nn.embedding import EmbeddingWithMask, EmbeddingInverse
from nl2prog.nn.utils.rnn import PaddedSequenceWithMask


class Predictor(nn.Module):
    def __init__(self, num_rules: int, num_tokens: int, num_node_types: int,
                 node_type_embedding_size: int,
                 embedding_size: int, query_size: int, hidden_size: int,
                 att_hidden_size: int, dropout: float):
        """
        Constructor

        Parameters
        ----------
        num_rules: int
            The number of rules
        num_tokens: int
            The number of tokens
        num_node_types: int
            The number of node types
        node_type_embedding_size: int
            Size of each node-type embedding vector
        embedding_size: int
            Size of each embedding vector
        query_size: int
            Size of each query vector
        hidden_size: int
            Size of each hidden state
        att_hidden_size: int
            The number of features in the hidden state for attention
        dropout: float
            The probability of dropout
        """
        super(Predictor, self).__init__()
        self.hidden_size = hidden_size
        self._num_rules = num_rules
        self._num_tokens = num_tokens
        self._num_node_types = num_node_types
        self._rule_embed = EmbeddingWithMask(
            num_rules, embedding_size, num_rules)
        self._rule_embed_inv = \
            EmbeddingInverse(self._rule_embed.num_embeddings)
        self._token_embed = EmbeddingWithMask(
            num_tokens, embedding_size, num_tokens)
        self._token_embed_inv = \
            EmbeddingInverse(self._token_embed.num_embeddings)
        self._node_type_embed = EmbeddingWithMask(
            num_node_types, node_type_embedding_size, num_node_types)
        self._decoder = Decoder(query_size,
                                2 * embedding_size + node_type_embedding_size,
                                hidden_size, att_hidden_size, dropout)
        self._l_rule = nn.Linear(hidden_size, embedding_size)
        self._l_token = nn.Linear(hidden_size + query_size, embedding_size)
        self._l_generate = nn.Linear(hidden_size, 2)
        self._pointer_net = PointerNet(
            hidden_size + query_size, query_size, att_hidden_size)
        nn.init.normal_(self._rule_embed.weight, std=0.01)
        nn.init.normal_(self._token_embed.weight, std=0.01)
        nn.init.normal_(self._node_type_embed.weight, std=0.01)
        nn.init.xavier_uniform_(self._l_rule.weight)
        nn.init.zeros_(self._l_rule.bias)
        nn.init.xavier_uniform_(self._l_token.weight)
        nn.init.zeros_(self._l_token.bias)
        nn.init.xavier_uniform_(self._l_generate.weight)
        nn.init.zeros_(self._l_generate.bias)

    def forward(self,
                query: PaddedSequenceWithMask,
                action: PaddedSequenceWithMask,
                previous_action: PaddedSequenceWithMask,
                history: torch.FloatTensor,
                state: Tuple[torch.FloatTensor, torch.FloatTensor]
                ) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                           torch.FloatTensor, torch.FloatTensor,
                           Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """
        Parameters
        ----------
        query: rnn.PackedSequenceWithMask
            The query embedding vector
            The shape of the query embedding vector is (L_q, B, query_size).
        action: rnn.PackedSequenceWithMask
            The input sequence of action. Each action is represented by
            the tuple of (ID of the node types, ID of the parent-action's rule,
            the index of the parent action).
            The padding value should be -1.
        previous_action: rnn.PackedSequenceWithMask
            The input sequence of previous action. Each action is represented
            by the tuple of (ID of the applied rule, ID of the inserted token,
            the index of the word copied from the query).
            The padding value should be -1.
        history: torch.FloatTensor
            The list of LSTM states. The shape is (L_h, B, hidden_size)
        (h_0, c_0): Tuple[torch.FloatTensor, torch.FloatTensor]
            The tuple of the LSTM initial states. The shape of each tensor is
            (B, hidden_size)

        Returns
        -------
        rule_pred: rnn.PackedSequenceWithMask
            The probabilities of apply-rule
        token_pred: PaddedSequenceWithMask
            The probabilities of gen-token
        copy_pred: PaddedSequenceWithMask
            The probabilities of copy-token
        history: torch.FloatTensor
            The list of LSTM states. The shape is (L_h + 1, B, hidden_size)
        (h_n, c_n) : Tuple[torch.FloatTensor, torch.FloatTensor]
            The tuple of the next states. The shape of each tensor is
            (B, hidden_size)
        """
        L_a, B, _ = action.data.shape
        L_q, _, _ = query.data.shape
        _, _, h = history.shape

        node_types, parent_rule, parent_index = torch.split(
            action.data, 1, dim=2)  # (L_a, B, 1)
        prev_rules, prev_tokens, prev_words = torch.split(
            previous_action.data, 1, dim=2)  # (L_a, B, 1)

        # Change the padding value
        node_types = node_types + \
            (node_types == -1).long() * (self._num_node_types + 1)
        node_types = node_types.reshape([L_a, B])
        parent_rule = parent_rule + \
            (parent_rule == -1).long() * (self._num_rules + 1)
        parent_rule = parent_rule.reshape([L_a, B])
        prev_rules = prev_rules + \
            (prev_rules == -1).long() * (self._num_rules + 1)
        prev_rules = prev_rules.reshape([L_a, B])
        prev_tokens = prev_tokens + \
            (prev_tokens == -1).long() * (self._num_tokens + 1)
        prev_tokens = prev_tokens.reshape([L_a, B])

        # Embed previous actions
        prev_action_embed = self._rule_embed(prev_rules) + \
            self._token_embed(prev_tokens)  # (L_a, B, embedding_size)
        # Embed action
        node_type_embed = self._node_type_embed(
            node_types)  # (L_a, B, node_type_embedding_size)
        parent_rule_embed = self._rule_embed(
            parent_rule)  # (L_a, B, embedding_size)

        # Decode embeddings
        decoder_input = torch.cat(
            [prev_action_embed, node_type_embed, parent_rule_embed],
            dim=2)  # (L_a, B, input_size)
        output, contexts, history, (h_n, c_n) = \
            self._decoder(query,
                          PaddedSequenceWithMask(decoder_input, action.mask),
                          PaddedSequenceWithMask(parent_index, action.mask),
                          history,
                          state)  # (L_a, B, *)
        # (L_a, B, hidden_size + query_size)
        dc = torch.cat([output.data, contexts.data], dim=2)

        # Calculate probabilities
        # (L_a, B, embedding_size)
        rule_pred = torch.tanh(self._l_rule(output.data))
        rule_pred = self._rule_embed_inv(
            rule_pred,
            self._rule_embed)  # (L_a, B, num_rules + 1)
        rule_pred = torch.softmax(
            rule_pred[:, :, :-1], dim=2)  # (L_a, B, num_rules)

        token_pred = torch.tanh(self._l_token(dc))  # (L_a, B, embedding_size)
        token_pred = self._token_embed_inv(
            token_pred,
            self._token_embed)  # (L_a, B, num_tokens + 1)
        token_pred = torch.softmax(
            token_pred[:, :, :-1], dim=2)  # (L_a, B, num_tokens)

        copy_pred = self._pointer_net(dc, query)  # (L_a, B, query_length)
        copy_pred = torch.exp(copy_pred)
        copy_pred *= query.mask.permute(1, 0).view(1, B, L_q).float()

        generate_pred = torch.softmax(
            self._l_generate(output.data), dim=2)  # (L_a, B, 2)
        token, copy = torch.split(generate_pred, 1, dim=2)  # (L_a, B, 1)

        token_pred = token * token_pred  # (L_a, B, num_tokens)
        copy_pred = copy * copy_pred  # (L_a, B, query_length)

        return (PaddedSequenceWithMask(rule_pred, output.mask),
                PaddedSequenceWithMask(token_pred, output.mask),
                PaddedSequenceWithMask(copy_pred, output.mask), history,
                (h_n, c_n))
