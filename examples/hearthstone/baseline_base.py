from examples.hearthstone import base  # noqa
from mlprogram.launch import global_options
from mlprogram.tasks.train import Epoch

# options for dataset
global_options["word_threshold"] = 3
global_options["token_threshold"] = 0

# options for model
global_options["node_type_embedding_size"] = 64
global_options["embedding_size"] = 128
global_options["hidden_size"] = 256
global_options["attr_hidden_size"] = 50
global_options["dropout"] = 0.2

# options for training
global_options["batch_size"] = 1
global_options["n_epoch"] = Epoch(20)
global_options["eval_interval"] = Epoch(10)
global_options["snapshot_interval"] = Epoch(1)
global_options["metric_threshold"] = 1.0
global_options["metric"] = "bleu@1"

# options for evaluation
global_options["beam_size"] = 15
global_options["max_step_size"] = 350
global_options["metric_top_n"] = [1]

global_options["train_artifact_dir"] = "artifacts/train"
global_options["evaluate_artifact_dir"] = "artifacts/evaluate"
