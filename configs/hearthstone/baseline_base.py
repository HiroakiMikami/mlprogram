imports = ["../nl2prog_baseline/base.py", "base.py"]

dataset_params = {
    "word_threshold": 3,
    "token_threshold": 0,
}
model_params = {
    "node_type_embedding_size": 64,
    "embedding_size": 128,
    "hidden_size": 256,
    "attr_hidden_size": 50,
    "dropout": 0.2,
}
train_params = {
    "batch_size": 1,
    "n_epoch": 20,
    "eval_interval": 10,
    "snapshot_interval": 1,
    "metric_threshold": 1.0,
    "metric": "bleu@1",
}
inference_params = {
    "beam_size": 15,
    "max_step_size": 350,
    "inference_timeout_sec": 10,  # TODO
}
params = {
    "metric_top_n": [1],
}
train_artifact_dir = "artifacts/train"
evaluate_artifact_dir = "artifacts/evaluate"
