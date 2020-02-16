import unittest
from dummy_dataset import is_subtype, train_dataset, test_dataset
from typing import List
from tqdm import trange

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchnlp.encoders import LabelEncoder

import fairseq.optim as optim

from nl2prog.encoders import ActionSequenceEncoder
from nl2prog.utils import Query, synthesize as _synthesize, evaluate
from nl2prog.utils.treegen import BeamSearchSynthesizer
from nl2prog.language.action \
    import ast_to_action_sequence as to_seq, ActionOptions
from nl2prog.utils.data \
    import get_samples, to_eval_dataset, get_words, get_characters
from nl2prog.utils.data.treegen import to_train_dataset, collate_train_dataset
from nl2prog.nn import Loss, Accuracy as Acc
from nl2prog.nn.treegen import TrainModel
from nl2prog.nn.utils import rnn as nrnn
from nl2prog.metrics import Accuracy


def tokenize_query(str: str) -> Query:
    return Query(str.split(" "), str.split(" "))


def tokenize_token(token: str) -> List[str]:
    return [token]


def tokenize_token_2(token: str) -> List[str]:
    if token == "print":
        return [token]
    return list(token)


class TestTreeGen(unittest.TestCase):
    def evaluate(self, model, options, dataset):
        test_dataset = to_eval_dataset(dataset)
        encoder, model = model
        synthesizer = BeamSearchSynthesizer(
            5, tokenize_query, model.query_embedding, model.rule_embedding,
            model.nl_reader, model.ast_reader, model.decoder, model.predictor,
            encoder[0], encoder[1], encoder[2], 32, 2, is_subtype,
            options=options, max_steps=20)

        def synthesize(query: str):
            return _synthesize(query, synthesizer)

        accuracy = Accuracy(lambda x: x, lambda x: x)
        metrics = {"accuracy": accuracy}
        result = evaluate(test_dataset, synthesize,
                          top_n=[1], metrics=metrics)
        return result

    def train(self, options, dataset, tokenize_token):
        def to_action_sequence(ast):
            return to_seq(ast, options=options, tokenizer=tokenize_token)

        words = get_words(dataset, tokenize_query)
        chars = get_characters(dataset, tokenize_query)
        samples = get_samples(dataset, tokenize_token, to_action_sequence)
        qencoder = LabelEncoder(words, 2)
        cencoder = LabelEncoder(chars, 0)
        aencoder = ActionSequenceEncoder(samples, 2, options=options)

        train_dataset = to_train_dataset(
            dataset, tokenize_query, tokenize_token, to_action_sequence,
            qencoder, cencoder, aencoder, 32, 2, options)
        model = TrainModel(qencoder, cencoder, aencoder, 32, 2, 1, 6, 5, 5,
                           256, 1024, 0.0)
        optimizer = optim.adafactor.Adafactor(model.parameters())
        loss_function = Loss()
        acc_function = Acc()

        for _ in trange(100):
            loader = DataLoader(train_dataset, 1, shuffle=True,
                                collate_fn=collate_train_dataset)
            avg_acc = 0
            for data, ground_truth in loader:
                word_query = data[0]
                char_query = data[1]
                prev_action = data[2]
                rule_prev_action = data[3]
                depth = data[4]
                matrix = data[5]
                word_query = nrnn.pad_sequence(word_query, padding_value=-1)
                char_query = nrnn.pad_sequence(char_query, padding_value=-1)
                prev_action = nrnn.pad_sequence(prev_action, padding_value=-1)
                rule_prev_action = \
                    nrnn.pad_sequence(rule_prev_action, padding_value=-1)
                depth = \
                    nrnn.pad_sequence(depth).data.reshape(1, -1).permute(1, 0)
                ground_truth = \
                    nrnn.pad_sequence(ground_truth, padding_value=-1)
                L = prev_action.data.shape[0]
                matrix = [F.pad(m, (0, L - m.shape[0], 0, L - m.shape[1]))
                          for m in matrix]
                matrix = nrnn.pad_sequence(matrix).data.permute(1, 0, 2)

                rule_prob, token_prob, copy_prob = model(
                    word_query, char_query, prev_action, rule_prev_action,
                    depth, matrix)
                loss = loss_function(rule_prob, token_prob, copy_prob,
                                     ground_truth)
                acc = acc_function(rule_prob, token_prob, copy_prob,
                                   ground_truth)
                model.zero_grad()
                loss.backward()
                optimizer.step()
                avg_acc += acc.item() / len(loader)
            if avg_acc == 1.0:
                break
        return (qencoder, cencoder, aencoder), model

    def test_default_settings(self):
        torch.manual_seed(0)
        options = ActionOptions(False, False)
        model = self.train(options, train_dataset, tokenize_token)
        results = self.evaluate(model, options, test_dataset)
        self.assertAlmostEqual(1.0, results.metrics[1]["accuracy"])

    def test_split_nonterminals(self):
        torch.manual_seed(0)
        options = ActionOptions(False, True)
        model = self.train(options, train_dataset, tokenize_token)
        results = self.evaluate(model, options, test_dataset)
        self.assertAlmostEqual(1.0, results.metrics[1]["accuracy"])

        model = self.train(options, train_dataset, tokenize_token_2)
        results = self.evaluate(model, options, test_dataset)
        self.assertAlmostEqual(1.0, results.metrics[1]["accuracy"])

    def test_retain_variadic_args(self):
        torch.manual_seed(0)
        options = ActionOptions(True, False)
        model = self.train(options, train_dataset, tokenize_token)
        results = self.evaluate(model, options, test_dataset)
        self.assertAlmostEqual(1.0, results.metrics[1]["accuracy"])


if __name__ == "__main__":
    unittest.main()
