import unittest
from dummy_dataset import is_subtype, train_dataset, test_dataset
from typing import List
from tqdm import trange

import torch
from torch.utils.data import DataLoader

from torchnlp.encoders import LabelEncoder

import fairseq.optim as optim

from nl2prog.encoders import ActionSequenceEncoder
from nl2prog.utils \
    import Query, synthesize as _synthesize, evaluate, \
    CommonBeamSearchSynthesizer
from nl2prog.language.action \
    import ast_to_action_sequence as to_seq, ActionOptions
from nl2prog.utils.data \
    import get_samples, to_eval_dataset, get_words, get_characters, \
    Collate, CollateGroundTruth, collate_none, split_none, CollateNlFeature
from nl2prog.utils.data.treegen \
    import CollateQuery, CollateActionSequence, CollateInput
from nl2prog.utils.transform \
    import TransformDataset, TransformGroundTruth, TransformCode
from nl2prog.utils.transform.treegen import TransformQuery, TransformEvaluator
from nl2prog.nn import Loss, Accuracy as Acc
from nl2prog.nn.treegen import TrainModel
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
        transform_input = TransformQuery(tokenize_query, encoder[0],
                                         encoder[1], 32)
        transform_evaluator = TransformEvaluator(encoder[2], 2, train=False)
        synthesizer = CommonBeamSearchSynthesizer(
            5, transform_input, transform_evaluator,
            CollateInput(torch.device("cpu")),
            CollateActionSequence(torch.device("cpu")),
            CollateQuery(torch.device("cpu")), collate_none,
            CollateNlFeature(torch.device("cpu")),
            collate_none, split_none,
            model.input_reader, model.action_sequence_reader, model.decoder,
            model.predictor, encoder[2], is_subtype,
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

        tquery = TransformQuery(tokenize_query, qencoder, cencoder, 32)
        tcode = TransformCode(to_action_sequence, options)
        teval = TransformEvaluator(aencoder, 2)
        tgt = TransformGroundTruth(aencoder)
        transform = TransformDataset(tquery, tcode, teval, tgt)
        train_dataset = transform(dataset)

        model = TrainModel(qencoder, cencoder, aencoder, 32, 2, 1, 6, 5, 5,
                           256, 1024, 0.0)
        optimizer = optim.adafactor.Adafactor(model.parameters())
        loss_function = Loss()
        acc_function = Acc()

        for _ in trange(100):
            loader = DataLoader(train_dataset, 1, shuffle=True,
                                collate_fn=Collate(
                                    CollateInput(torch.device("cpu")),
                                    CollateActionSequence(torch.device("cpu")),
                                    CollateQuery(torch.device("cpu")),
                                    CollateGroundTruth(torch.device("cpu"))))
            avg_acc = 0
            for input, action_sequence, query, ground_truth in loader:
                rule_prob, token_prob, copy_prob = model(
                    input, action_sequence, query)
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
