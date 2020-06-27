import torch
import torch.nn as nn
from typing import Dict, Any, cast

from mlprogram.nn.nl2code import ActionSequenceReader
from mlprogram.nn import PointerNet
from mlprogram.nn.embedding import EmbeddingInverse
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class Predictor(nn.Module):
    def __init__(self, reader: ActionSequenceReader, embedding_size: int,
                 query_size: int, hidden_size: int, att_hidden_size: int):
        """
        Constructor

        Parameters
        ----------
        reader:
        embedding_size: int
            Size of each embedding vector
        query_size: int
            Size of each query vector
        hidden_size: int
            Size of each hidden state
        att_hidden_size: int
            The number of features in the hidden state for attention
        """
        super(Predictor, self).__init__()
        self.hidden_size = hidden_size
        self._reader = reader
        self._rule_embed_inv = \
            EmbeddingInverse(self._reader._rule_embed.num_embeddings)
        self._token_embed_inv = \
            EmbeddingInverse(self._reader._token_embed.num_embeddings)
        self._l_rule = nn.Linear(hidden_size, embedding_size)
        self._l_token = nn.Linear(hidden_size + query_size, embedding_size)
        self._l_generate = nn.Linear(hidden_size, 2)
        self._pointer_net = PointerNet(
            hidden_size + query_size, query_size, att_hidden_size)
        nn.init.xavier_uniform_(self._l_rule.weight)
        nn.init.zeros_(self._l_rule.bias)
        nn.init.xavier_uniform_(self._l_token.weight)
        nn.init.zeros_(self._l_token.bias)
        nn.init.xavier_uniform_(self._l_generate.weight)
        nn.init.zeros_(self._l_generate.bias)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parameters
        ----------
        nl_query_features: PaddedSequenceWithMask
            (L_nl, N, nl_feature_size) where L_nl is the sequence length,
            N is the batch size.
        action_features: PaddedSequenceWithMask
                Packed sequence containing the output hidden states.
        action_contexts: PaddedSequenceWithMask
                Packed sequence containing the context vectors.

        Returns
        -------
        rule_probs: PaddedSequenceWithMask
            (L_ast, N, rule_size) where L_ast is the sequence length,
            N is the batch_size.
        token_probs: PaddedSequenceWithMask
           (L_ast, N, token_size) where L_ast is the sequence length,
            N is the batch_size.
        copy_probs: PaddedSequenceWithMask
            (L_ast, N, L_nl) where L_ast is the sequence length,
            N is the batch_size.
        """
        nl_query_features = cast(PaddedSequenceWithMask,
                                 inputs["nl_query_features"])
        action_features = cast(PaddedSequenceWithMask,
                               inputs["action_features"])
        action_contexts = cast(PaddedSequenceWithMask,
                               inputs["action_contexts"])
        L_q, B, _ = nl_query_features.data.shape

        # Decode embeddings
        # (L_a, B, hidden_size + query_size)
        dc = torch.cat([action_features.data, action_contexts.data], dim=2)

        # Calculate probabilities
        # (L_a, B, embedding_size)
        rule_pred = torch.tanh(self._l_rule(action_features.data))
        rule_pred = self._rule_embed_inv(
            rule_pred,
            self._reader._rule_embed)  # (L_a, B, num_rules + 1)
        rule_pred = torch.softmax(
            rule_pred[:, :, :-1], dim=2)  # (L_a, B, num_rules)

        token_pred = torch.tanh(self._l_token(dc))  # (L_a, B, embedding_size)
        token_pred = self._token_embed_inv(
            token_pred,
            self._reader._token_embed)  # (L_a, B, num_tokens + 1)
        token_pred = torch.softmax(
            token_pred[:, :, :-1], dim=2)  # (L_a, B, num_tokens)

        # (L_a, B, query_length)
        copy_pred = self._pointer_net(dc, nl_query_features)
        copy_pred = torch.exp(copy_pred)
        copy_pred = copy_pred * \
            nl_query_features.mask.permute(1, 0).view(1, B, L_q)\
            .to(copy_pred.dtype)

        generate_pred = torch.softmax(
            self._l_generate(action_features.data), dim=2)  # (L_a, B, 2)
        token, copy = torch.split(generate_pred, 1, dim=2)  # (L_a, B, 1)

        token_pred = token * token_pred  # (L_a, B, num_tokens)
        copy_pred = copy * copy_pred  # (L_a, B, query_length)

        if self.training:
            inputs["rule_probs"] = \
                PaddedSequenceWithMask(rule_pred, action_features.mask)
            inputs["token_probs"] = \
                PaddedSequenceWithMask(token_pred, action_features.mask)
            inputs["copy_probs"] = \
                PaddedSequenceWithMask(copy_pred, action_features.mask)
        else:
            inputs["rule_probs"] = rule_pred[-1, :, :]
            inputs["token_probs"] = token_pred[-1, :, :]
            inputs["copy_probs"] = copy_pred[-1, :, :]
        return inputs
