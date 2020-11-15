imports = ["without_repl_base.py"]
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
                mlprogram.nn.action_sequence.Loss(
                    reduction="mean",
                ),
            ],
            [
                "pick",
                mlprogram.nn.Function(
                    f=Pick(
                        key="output@action_sequence_loss",
                    ),
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
        n_process=2,
    ),
    metric="generation_rate",
    threshold=1.0,
    maximize=True,
    collate=collate_fn,
    batch_size=batch_size,
    length=mlprogram.entrypoint.train.Iteration(
        n=option.n_pretrain_iteration,
    ),
    evaluation_interval=mlprogram.entrypoint.train.Iteration(
        n=option.interval_iter,
    ),
    snapshot_interval=mlprogram.entrypoint.train.Iteration(
        n=option.interval_iter,
    ),
    device=device,
)
