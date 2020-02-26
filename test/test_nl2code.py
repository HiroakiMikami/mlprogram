import unittest
from dummy_dataset import is_subtype, train_dataset, test_dataset
from typing import List
from tqdm import trange

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from torchnlp.encoders import LabelEncoder

from nl2prog.encoders import ActionSequenceEncoder
from nl2prog.utils import Query, synthesize as _synthesize, evaluate
from nl2prog.utils.nl2code import BeamSearchSynthesizer
from nl2prog.language.action \
    import ast_to_action_sequence as to_seq, ActionOptions
from nl2prog.utils.data import get_samples, to_eval_dataset, get_words
from nl2prog.utils.data.nl2code import to_train_dataset, collate_train_dataset
from nl2prog.nn import Loss, Accuracy as Acc
from nl2prog.nn.nl2code import TrainModel
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
                                            encoder[0], encoder[1], is_subtype,
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
        samples = get_samples(dataset, tokenize_token, to_action_sequence)
        qencoder = LabelEncoder(words, 2)
        aencoder = ActionSequenceEncoder(samples, 2, options=options)

        train_dataset = to_train_dataset(
            dataset, tokenize_query, tokenize_token, to_action_sequence,
            qencoder, aencoder, options)
        model = TrainModel(qencoder, aencoder, 256, 64, 256, 64, 0.0)
        optimizer = optim.Adam(model.parameters())
        loss_function = Loss()
        acc_function = Acc()

        for _ in trange(100):
            loader = DataLoader(train_dataset, 1, shuffle=True,
                                collate_fn=collate_train_dataset)
            avg_acc = 0
            for (query, action, prev_action), ground_truth in loader:
                query = nrnn.pad_sequence(query, padding_value=-1)
                action = nrnn.pad_sequence(action, padding_value=-1)
                prev_action = \
                    nrnn.pad_sequence(prev_action, padding_value=-1)
                ground_truth = \
                    nrnn.pad_sequence(ground_truth, padding_value=-1)

                rule_prob, token_prob, copy_prob = model(
                    query, action, prev_action)
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
        return (qencoder, aencoder), model

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
