from collections import OrderedDict

import torch

import examples.deepfix.baseline_base as base
import mlprogram
from mlprogram.builtins import Apply, Div, Flatten, Pick
from mlprogram.launch import global_options

global_options["n_test_sample"] = None

global_options["device_type"] = "cuda"
device = torch.device(global_options.device_type, 0)

optimizer = torch.optim.Adam(base.model.parameters())
collate_fn = mlprogram.functools.Sequence(OrderedDict(
    to_episode=mlprogram.functools.Map(func=base.to_episode),
    flatten=Flatten(),
    transform=mlprogram.functools.Map(func=base.transform),
    collate=base.collate.collate,
))
loss_fn = torch.nn.Sequential(OrderedDict(
    loss=Apply(
        module=mlprogram.nn.action_sequence.Loss(reduction="sum"),
        in_keys=[
            "rule_probs",
            "token_probs",
            "reference_probs",
            "ground_truth_actions",
        ],
        out_key="action_sequence_loss",
    ),
    normalize=Apply(
        in_keys=[["action_sequence_loss", "lhs"]],
        out_key="action_sequence_loss",
        module=mlprogram.nn.Function(Div()),
        constants={"rhs": global_options.batch_size},
    ),
    pick=mlprogram.nn.Function(f=Pick(key="action_sequence_loss")),
))


def main():
    mlprogram.tasks.train_supervised(
        output_dir=global_options.train_artifact_dir,
        dataset=base.train_dataset,
        model=base.model,
        optimizer=optimizer,
        loss=loss_fn,
        evaluate=mlprogram.tasks.EvaluateSynthesizer(
            dataset=base.test_dataset,
            synthesizer=base.synthesizer,
            metrics={},
            top_n=[],
            n_samples=global_options.n_test_sample,
        ),
        metric="generation_rate",
        collate=collate_fn,
        batch_size=global_options.batch_size,
        length=global_options.n_epoch,
        evaluation_interval=global_options.eval_interval,
        snapshot_interval=global_options.snapshot_interval,
        device=device,
    )
