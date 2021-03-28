import torch

import examples.deepfix.baseline_base as base
import mlprogram
from mlprogram.launch import global_options

global_options["n_validate_sample"] = None

global_options["device_type"] = "cpu"
device = torch.device(global_options.device_type, 0)


mlprogram.tasks.evaluate(
    input_dir=global_options.train_artifact_dir,
    output_dir=global_options.evaluate_artifact_dir,
    valid_dataset=base.valid_dataset,
    model=base.model,
    synthesizer=base.synthesizer,
    metrics={},
    top_n=[],
    device=device,
    n_samples=global_options.n_validate_sample,
)
