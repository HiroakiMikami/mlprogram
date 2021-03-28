import torch
from torch import nn

from mlprogram.nn.embedding import EmbeddingWithMask
from mlprogram.nn.utils.rnn import PaddedSequenceWithMask


class PreviousActionsEmbedding(nn.Module):
    def __init__(self, n_rule: int, n_token: int, embedding_size: int):
        super().__init__()
        self.n_token = n_token
        self.rule_embed = EmbeddingWithMask(n_rule, embedding_size, -1)
        self.token_embed = EmbeddingWithMask(n_token + 1, embedding_size, -1)

    def forward(self,
                previous_actions: PaddedSequenceWithMask) -> PaddedSequenceWithMask:
        """
        Parameters
        ----------
        previous_actions: PaddedSequenceWithMask
            The shape is (L, N, 3). where L is the sequence length and
            N is the batch size. The padding value should be -1.
            [:, :, 0] represent the rule IDs, [:, :, 1] represent the token
            IDs, [:, :, 2] represent the indexes of the queries.
            The padding value should be -1

        Returns
        -------
        seq_embed: PaddedSequenceWithMask
            The shape is (L, N, embedding_dim). where L is the sequence length
            and N is the batch size.
        """
        L, N = previous_actions.data.shape[:2]

        rule_seq = previous_actions.data[:, :, 0]

        token_seq = previous_actions.data[:, :, 1]
        """
        # TODO this decreases the performance of CSG pbe significantly
        reference_seq = (token_seq == -1) * (previous_actions.data[:, :, 2] != -1)
        # reference_seq => self.token_num
        token_seq = token_seq + reference_seq * (self.n_token + 1)
        """

        embedding = self.rule_embed(rule_seq) + self.token_embed(token_seq)
        return PaddedSequenceWithMask(embedding, previous_actions.mask)


class ActionsEmbedding(nn.Module):
    def __init__(self,
                 n_rule: int, n_token: int, n_node_type: int,
                 node_type_embedding_size: int, embedding_size: int):
        """
        Constructor

        Parameters
        ----------
        n_rule: int
            The number of rules
        n_token: int
            The number of tokens
        n_node_type: int
            The number of node types
        node_type_embedding_size: int
            Size of each node-type embedding vector
        embedding_size: int
            Size of each embedding vector
        """
        super().__init__()
        self.output_size = embedding_size * 2 + node_type_embedding_size
        self.previous_actions_embed = PreviousActionsEmbedding(n_rule, n_token,
                                                               embedding_size)
        self.node_type_embed = EmbeddingWithMask(n_node_type, node_type_embedding_size,
                                                 -1)
        nn.init.normal_(self.previous_actions_embed.rule_embed.weight, std=0.01)
        nn.init.normal_(self.previous_actions_embed.token_embed.weight, std=0.01)
        nn.init.normal_(self.node_type_embed.weight, std=0.01)

    def forward(self,
                actions: PaddedSequenceWithMask,
                previous_actions: PaddedSequenceWithMask,
                ) -> PaddedSequenceWithMask:
        """
        Parameters
        ----------
        actions: rnn.PackedSequenceWithMask
            The input sequence of action. Each action is represented by
            the tuple of (ID of the node types, ID of the parent-action's
            rule, the index of the parent action).
            The padding value should be -1.
        previous_actions: rnn.PackedSequenceWithMask
            The input sequence of previous action. Each action is
            represented by the tuple of (ID of the applied rule, ID of
            the inserted token, the index of the word copied from
            the reference).
            The padding value should be -1.

        Returns
        -------
        action_features: PaddedSequenceWithMask
            Packed sequence containing the output hidden states.
        """
        L_a, B, _ = actions.data.shape

        node_types, parent_rule, parent_index = torch.split(
            actions.data, 1, dim=2)  # (L_a, B, 1)
        node_types = node_types.reshape([L_a, B])
        parent_rule = parent_rule.reshape([L_a, B])

        # Embed previous actions
        prev_action_embed = self.previous_actions_embed(previous_actions).data
        # Embed action
        node_type_embed = self.node_type_embed(node_types)
        parent_rule_embed = self.previous_actions_embed.rule_embed(parent_rule)

        # Decode embeddings
        feature = torch.cat(
            [prev_action_embed, node_type_embed, parent_rule_embed],
            dim=2
        )  # (L_a, B, input_size)
        return PaddedSequenceWithMask(feature, actions.mask)
