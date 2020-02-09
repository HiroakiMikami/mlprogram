import unittest
from dummy_dataset import is_subtype, train_dataset, test_dataset
from typing import List
from tqdm import trange

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import rnn

from nl2prog.encoders import Encoder
from nl2prog.utils import Query, synthesize as _synthesize, evaluate
from nl2prog.utils.nl2code import BeamSearchSynthesizer
from nl2prog.language.action \
    import ast_to_action_sequence as to_seq, ActionOptions
from nl2prog.utils.data import get_samples, to_eval_dataset
from nl2prog.utils.data.nl2code import to_train_dataset, collate_train_dataset
from nl2prog.nn.nl2code import TrainModel, Loss, Accuracy as Acc
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


class TestNL2Code(unittest.TestCase):
    def evaluate(self, model, options, dataset):
        test_dataset = to_eval_dataset(dataset)
        encoder, model = model
        synthesizer = BeamSearchSynthesizer(5, tokenize_query,
                                            model.encoder, model.predictor,
                                            encoder, is_subtype,
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

        samples = get_samples(dataset, tokenize_query, tokenize_token,
                              to_action_sequence)
        encoder = Encoder(samples, 2, 2, options=options)

        train_dataset = to_train_dataset(
            dataset, tokenize_query, tokenize_token, to_action_sequence,
            encoder, options)
        model = TrainModel(encoder, 256, 64, 256, 64, 0.0)
        optimizer = optim.Adam(model.parameters())
        loss_function = Loss()
        acc_function = Acc()

        for _ in trange(100):
            loader = DataLoader(train_dataset, 1, shuffle=True,
                                collate_fn=collate_train_dataset)
            avg_acc = 0
            for query, action, prev_action in loader:
                query = rnn.pack_sequence(query, enforce_sorted=False)
                action = rnn.pack_sequence(action, enforce_sorted=False)
                prev_action_train = [x[:-1] for x in prev_action]
                action_ground_truth = [x[1:] for x in prev_action]
                prev_action_train = \
                    rnn.pack_sequence(prev_action_train, enforce_sorted=False)
                action_ground_truth = \
                    rnn.pack_sequence(action_ground_truth,
                                      enforce_sorted=False)
                query = nrnn.pad_packed_sequence(query, padding_value=-1)
                action = nrnn.pad_packed_sequence(action, padding_value=-1)
                prev_action_train = \
                    nrnn.pad_packed_sequence(
                        prev_action_train, padding_value=-1)
                action_ground_truth = \
                    nrnn.pad_packed_sequence(action_ground_truth,
                                             padding_value=-1)

                rule_prob, token_prob, copy_prob, _, _ = model(
                    query, action, prev_action_train)
                loss = loss_function(rule_prob, token_prob,
                                     copy_prob, action_ground_truth)
                acc = acc_function(rule_prob, token_prob,
                                   copy_prob, action_ground_truth)
                model.zero_grad()
                loss.backward()
                optimizer.step()
                avg_acc += acc.item() / len(loader)
            if avg_acc == 1.0:
                break
        return encoder, model

    def test_default_settings(self):
        torch.manual_seed(0)
        options = ActionOptions(True, True)
        model = self.train(options, train_dataset, tokenize_token)
        results = self.evaluate(model, options, test_dataset)
        self.assertAlmostEqual(1.0, results.metrics[1]["accuracy"])

        model = self.train(options, train_dataset, tokenize_token_2)
        results = self.evaluate(model, options, test_dataset)
        self.assertAlmostEqual(1.0, results.metrics[1]["accuracy"])

    def test_not_split_nonterminals(self):
        torch.manual_seed(0)
        options = ActionOptions(True, False)
        model = self.train(options, train_dataset, tokenize_token)
        results = self.evaluate(model, options, test_dataset)
        self.assertAlmostEqual(1.0, results.metrics[1]["accuracy"])

    def test_not_retain_variadic_args(self):
        torch.manual_seed(0)
        options = ActionOptions(False, True)
        model = self.train(options, train_dataset, tokenize_token)
        results = self.evaluate(model, options, test_dataset)
        self.assertAlmostEqual(1.0, results.metrics[1]["accuracy"])

        model = self.train(options, train_dataset, tokenize_token_2)
        results = self.evaluate(model, options, test_dataset)
        self.assertAlmostEqual(1.0, results.metrics[1]["accuracy"])


if __name__ == "__main__":
    unittest.main()
