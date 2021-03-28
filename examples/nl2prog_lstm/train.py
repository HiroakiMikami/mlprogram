from collections import OrderedDict

import torch

import mlprogram
from examples.nl2prog_lstm.base import setup as base_setup
from mlprogram.builtins import Apply, Constant, Pick
from mlprogram.launch import global_options


def run(
    *,
    dataset,
    parser,
    extract_reference,
    is_subtype,
    metric_top_n,
    metrics,
    metric,
    metric_threshold,
    setup_synthesizer,
):
    base = base_setup(
        dataset=dataset,
        parser=parser,
        extract_reference=extract_reference,
        is_subtype=is_subtype,
    )
    synthesizer = setup_synthesizer(base["synthesizer"])

    global_options["device_type"] = "cpu"
    device = torch.device("cpu", 0)

    global_options["n_test_sample"] = None

    transform = mlprogram.functools.Sequence(OrderedDict(
        set_train=Apply(module=Constant(value=True), in_keys=[], out_key="train"),
        transform_input=base["transform_input"],
        transform_code=Apply(
            module=mlprogram.transforms.action_sequence.GroundTruthToActionSequence(
                parser=parser,
            ),
            in_keys=["ground_truth"],
            out_key="action_sequence",
        ),
        transform_action_sequence=base["transform_action_sequence"],
        transform_ground_truth=Apply(
            module=mlprogram.transforms.action_sequence.EncodeActionSequence(
                action_sequence_encoder=base["action_sequence_encoder"],
            ),
            in_keys=["action_sequence", "reference"],
            out_key="ground_truth_actions",
        ),
    ),
    )
    optimizer = torch.optim.Adam(base["model"].parameters())
    mlprogram.tasks.train_supervised(
        output_dir=global_options.train_artifact_dir,
        dataset=base["train_dataset"],
        model=base["model"],
        optimizer=optimizer,
        loss=torch.nn.Sequential(OrderedDict(
            loss=Apply(
                module=mlprogram.nn.action_sequence.Loss(),
                in_keys=[
                    "rule_probs",
                    "token_probs",
                    "reference_probs",
                    "ground_truth_actions",
                ],
                out_key="action_sequence_loss",
            ),
            pick=mlprogram.nn.Function(
                f=Pick(key="action_sequence_loss"),
            ),
        )),
        evaluate=mlprogram.tasks.EvaluateSynthesizer(
            dataset=base["test_dataset"],
            synthesizer=synthesizer,
            metrics=metrics,
            top_n=metric_top_n,
            n_sample=global_options.n_test_sample,
        ),
        metric=metric,
        threshold=metric_threshold,
        collate=mlprogram.functools.Compose(OrderedDict(
            transform=mlprogram.functools.Map(func=transform),
            collate=base["collate"].collate
        )),
        batch_size=global_options.batch_size,
        length=global_options.n_epoch,
        evaluation_interval=global_options.eval_interval,
        snapshot_interval=global_options.snapshot_interval,
        device=device,
    )
