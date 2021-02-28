imports = ["../nl2prog_baseline/base.py", "base.py"]
dataset_params = {
    "word_threshold": 5,
    "token_threshold": 5,
}
model_params = {
    "node_type_embedding_size": 64,
    "embedding_size": 128,
    "hidden_size": 256,
    "attr_hidden_size": 50,
    "dropout": 0.2,
}
train_params = {
    "batch_size": 10,
    "n_epoch": 50,
    "eval_interval": 10,
    "snapshot_interval": 1,
    "metric_threshold": 1.0,
    "metric": "bleu@1",
}
inference_params = {
    "beam_size": 15,
    "max_step_size": 100,
}
params = {
    "metric_top_n": [1],
}
synthesizer = base_synthesizer
