imports = ["baseline_base.py", "benchmark.py"]
benchmark = "short"
option = select(options=benchmark_dict, key="short")
device = torch.device(
    type_str="cpu",
    index=0,
)
batch_size = 32
optimizer = torch.optim.Optimizer(
    optimizer_cls=torch.optim.Adam(),
    model=model,
)
collate_fn = mlprogram.functools.Sequence(
    funcs=collections.OrderedDict(
        items=[
            [
                "transform",
                mlprogram.functools.Map(
                    func=transform,
                ),
            ],
            ["collate", collate.collate],
        ],
    ),
)
loss_fn = torch.nn.Sequential(
    modules=collections.OrderedDict(
        items=[
            [
                "loss",
                Apply(
                    module=mlprogram.nn.action_sequence.Loss(
                        reduction="mean",
                    ),
                    in_keys=[
                        "rule_probs",
                        "token_probs",
                        "reference_probs",
                        "ground_truth_actions",
                    ],
                    out_key="action_sequence_loss",
                ),
            ],
            [
                "pick",
                mlprogram.nn.Function(
                    f=Pick(
                        key="action_sequence_loss",
                    ),
                ),
            ],
        ],
    ),
)
main = mlprogram.entrypoint.train_supervised(
    output_dir=train_artifact_dir,
    dataset=train_dataset,
    model=model,
    optimizer=optimizer,
    loss=loss_fn,
    evaluate=mlprogram.entrypoint.EvaluateSynthesizer(
        dataset=test_dataset,
        synthesizer=synthesizer,
        metrics={},
        top_n=[],
    ),
    metric="generation_rate",
    threshold=1.0,
    maximize=True,
    collate=collate_fn,
    batch_size=batch_size,
    length=mlprogram.entrypoint.train.Iteration(
        n=train_option.n_train_iteration,
    ),
    evaluation_interval=mlprogram.entrypoint.train.Iteration(
        n=train_option.interval_iter,
    ),
    snapshot_interval=mlprogram.entrypoint.train.Iteration(
        n=train_option.interval_iter,
    ),
    device=device,
)
