imports = ["baseline_base.py"]
device = torch.device(
    type_str="cuda",
    index=0,
)
output_dir = "output/output"
optimizer = torch.optim.Optimizer(
    optimizer_cls=torch.optim.Adam(),
    model=model,
)
collate_fn = mlprogram.functools.Sequence(
    funcs=collections.OrderedDict(
        items=[
            [
                "to_episode",
                mlprogram.functools.Map(
                    func=to_episode,
                ),
            ],
            ["flatten", Flatten()],
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
batch_size = params.batch_size
loss_fn = torch.nn.Sequential(
    modules=collections.OrderedDict(
        items=[
            [
                "loss",
                Apply(
                    module=mlprogram.nn.action_sequence.Loss(
                        reduction="sum",
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
                "normalize",
                Apply(
                    in_keys=[["action_sequence_loss", "lhs"]],
                    out_key="action_sequence_loss",
                    module=mlprogram.nn.Function(
                        f=Div(),
                    ),
                    constants={"rhs": batch_size},
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
    workspace_dir="output/workspace",
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
    ),
    metric="generation_rate",
    collate=collate_fn,
    batch_size=batch_size,
    length=mlprogram.entrypoint.train.Epoch(
        n=params.n_epoch,
    ),
    evaluation_interval=mlprogram.entrypoint.train.Epoch(
        n=params.eval_interval,
    ),
    snapshot_interval=mlprogram.entrypoint.train.Epoch(
        n=params.snapshot_interval,
    ),
    device=device,
)
