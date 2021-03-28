import torch

import mlprogram
from examples.nl2prog_lstm.base import setup as base_setup
from mlprogram.launch import global_options


def run(
    *,
    dataset,
    parser,
    extract_reference,
    is_subtype,
    metric_top_n,
    metrics,
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

    global_options["n_validate_sample"] = None

    mlprogram.tasks.evaluate(
        input_dir=global_options.train_artifact_dir,
        output_dir=global_options.evaluate_artifact_dir,
        valid_dataset=base["valid_dataset"],
        model=base["model"],
        synthesizer=synthesizer,
        metrics=metrics,
        top_n=metric_top_n,
        device=device,
        n_sample=global_options.n_validate_sample,
    )
