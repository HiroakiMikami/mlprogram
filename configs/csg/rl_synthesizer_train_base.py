imports = ["rl_synthesizer_base.py"]
device = torch.device(
    type_str="cuda",
    index=0,
)
batch_size = 32
output_dir = "output/output"
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
                action_sequence_loss_fn,
            ],
            [
                "pick",
                mlprogram.nn.Function(
                    f=Pick(
                        key="loss",
                    ),
                ),
            ],
            [
                "aggregate",
                Apply(
                    in_keys=["loss"],
                    out_key="loss",
                    module=mlprogram.nn.AggregatedLoss(),
                ),
            ],
            [
                "normalize",
                Apply(
                    in_keys=[["loss", "lhs"]],
                    out_key="loss",
                    module=mlprogram.nn.Function(
                        f=Div(),
                    ),
                    constants={"rhs": batch_size},
                ),
            ],
        ],
    ),
)
main = mlprogram.entrypoint.train_supervised(
    workspace_dir="output/workspace_training",
    output_dir=output_dir,
    dataset=train_dataset,
    model=model,
    optimizer=optimizer,
    loss=loss_fn,
    evaluate=mlprogram.entrypoint.EvaluateSynthesizer(
        dataset=test_dataset,
        synthesizer=synthesizer,
        metrics={},
        top_n=[],
        n_process=None,
    ),
    metric="generation_rate",
    threshold=1.0,
    maximize=True,
    collate=collate_fn,
    batch_size=batch_size,
    length=mlprogram.entrypoint.train.Iteration(
        n=option.n_train_iteration,
    ),
    evaluation_interval=mlprogram.entrypoint.train.Iteration(
        n=option.interval_iter,
    ),
    snapshot_interval=mlprogram.entrypoint.train.Iteration(
        n=option.interval_iter,
    ),
    device=device,
)
