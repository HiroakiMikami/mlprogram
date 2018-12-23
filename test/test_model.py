import unittest
from math import isnan
from src.model import encoder, decoder, pred, loss
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np


class TestModel(unittest.TestCase):
    def test_encoder(self):
        words = nn.Variable((1, 3))
        words.d = 0
        words.d[0, 0] = 2
        words.d[0, 1] = 1
        words.d[0, 2] = 0

        with nn.parameter_scope("encoder"), nn.auto_forward():
            embed, mask = encoder(words, 3, 128, 256)
        self.assertEqual(embed.shape, (1, 3, 256))
        self.assertEqual(mask.shape, (1, 3))
        self.assertTrue(np.all(mask.d == [[1, 1, 0]]))

    def test_decoder(self):
        target_action = nn.Variable((1, 2, 3))
        target_action.d = 0
        target_action_type = nn.Variable((1, 2, 3))
        target_action_type.d = 0
        target_action_type.d[0, 0, 2] = 1
        target_node_type = nn.Variable((1, 2))
        target_node_type.d = 0
        target_parent_rule = nn.Variable((1, 2))
        target_parent_rule.d = 0
        target_parent_rule = nn.Variable((1, 2))
        target_parent_rule.d = 0
        target_parent_rule.d[0, 1] = -1
        target_parent_index = nn.Variable((1, 2))
        target_parent_index.d = 1
        query_embed = nn.Variable((1, 3, 1))
        query_embed.d = 1
        query_embed_mask = nn.Variable((1, 3))
        query_embed_mask.d = [[1, 1, 0]]

        with nn.parameter_scope("decoder"), nn.auto_forward():
            _, hs, cs, ctx, mask, hist = decoder(
                target_action, target_action_type, target_node_type,
                target_parent_rule, target_parent_index, query_embed,
                query_embed_mask, 2, 2, 2, 128, 64, 256, 50)
        self.assertEqual(hs.shape, (1, 2, 256))
        self.assertEqual(cs.shape, (1, 2, 256))
        self.assertEqual(ctx.shape, (1, 2, 1))
        self.assertEqual(mask.shape, (1, 2))
        self.assertEqual(hist.shape, (1, 3, 256))
        self.assertTrue(np.all(mask.d == [[1, 0]]))

    def test_pred(self):
        decoder_hidden_states = nn.Variable((1, 2, 3))
        decoder_hidden_states.d = 0
        ctx_vector = nn.Variable((1, 2, 3))
        ctx_vector.d = 0
        query_embed = nn.Variable((1, 3, 1))
        query_embed.d = 1
        query_embed_mask = nn.Variable((1, 3))
        query_embed_mask.d = [[1, 1, 0]]
        with nn.parameter_scope("pred"), nn.auto_forward():
            rule, token, gen, copy = pred(decoder_hidden_states, ctx_vector,
                                          query_embed, query_embed_mask, 2, 2,
                                          128, 128)
        self.assertEqual(rule.shape, (1, 2, 2))
        self.assertEqual(token.shape, (1, 2, 2))
        self.assertEqual(gen.shape, (1, 2, 2))
        self.assertEqual(copy.shape, (1, 2, 3))

    def test_loss(self):
        target_action = nn.Variable((1, 2, 3))
        target_action.d = 0
        target_action_type = nn.Variable((1, 2, 3))
        target_action_type.d = 0
        target_action_type.d[0, 0, 2] = 1
        target_action_mask = nn.Variable((1, 2))
        target_action_mask.d = [[1, 0]]

        gen_prob = nn.Variable((1, 2, 2))
        gen_prob.d = 0.5
        rule_prob = nn.Variable((1, 2, 2))
        rule_prob.d = 1
        token_prob = nn.Variable((1, 2, 2))
        token_prob.d = 1
        copy_prob = nn.Variable((1, 2, 3))
        copy_prob.d = 1

        with nn.parameter_scope("loss"), nn.auto_forward():
            l = loss(target_action, target_action_type, target_action_mask,
                     rule_prob, gen_prob, token_prob, copy_prob)
        self.assertEqual(l.shape, ())


if __name__ == "__main__":
    unittest.main()
