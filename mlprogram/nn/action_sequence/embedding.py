from torch import nn

from mlprogram.nn import EmbeddingWithMask
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
