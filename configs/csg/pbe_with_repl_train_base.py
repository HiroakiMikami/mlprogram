imports = ["pbe_with_repl_base.py"]
device = torch.device(
    type_str="cpu",
    index=0,
)
batch_size = 1
n_rollout = 16
output_dir = "output/output"
optimizer = torch.optim.Optimizer(
    optimizer_cls=torch.optim.Adam(),
    model=model,
)
loss_fn = torch.nn.Sequential(
    modules=collections.OrderedDict(
        items=[
            [
                "policy",
                torch.nn.Sequential(
                    modules=collections.OrderedDict(
                        items=[
                            [
                                "loss",
                                mlprogram.nn.action_sequence.Loss(
                                    reduction="none",
                                ),
                            ],
                            [
                                "weight_by_reward",
                                Apply(
                                    in_keys=[
                                        ["reward", "lhs"],
                                        ["action_sequence_loss", "rhs"],
                                    ],
                                    out_key="action_sequence_loss",
                                    module=mlprogram.nn.Function(
                                        f=Mul(),
                                    ),
                                ),
                            ],
                        ],
                    ),
                ),
            ],
            [
                "value",
                torch.nn.Sequential(
                    modules=collections.OrderedDict(
                        items=[
                            [
                                "reshape_reward",
                                Apply(
                                    in_keys=[["reward", "x"]],
                                    out_key="value_loss_target",
                                    module=torch.Reshape(
                                        sizes=[-1, 1],
                                    ),
                                ),
                            ],
                            [
                                "BCE",
                                Apply(
                                    in_keys=[
                                        ["value", "input"],
                                        ["value_loss_target", "target"],
                                    ],
                                    out_key="value_loss",
                                    module=torch.nn.BCELoss(
                                        reduction="sum",
                                    ),
                                ),
                            ],
                        ],
                    ),
                ),
            ],
            [
                "aggregate",
                Apply(
                    in_keys=["action_sequence_loss", "value_loss"],
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
            [
                "pick",
                mlprogram.nn.Function(
                    f=Pick(
                        key="loss",
                    ),
                ),
            ],
        ],
    ),
)
main = mlprogram.entrypoint.train_REINFORCE(
    input_dir=output_dir,
    workspace_dir="output/workspace",
    output_dir=output_dir,
    dataset=train_dataset,
    synthesizer=train_synthesizer,
    model=model,
    optimizer=optimizer,
    loss=loss_fn,
    evaluate=mlprogram.entrypoint.EvaluateSynthesizer(
        dataset=test_dataset,
        synthesizer=train_synthesizer,
        metrics={},
        top_n=[],
    ),
    metric="generation_rate",
    threshold=1.0,
    reward=mlprogram.metrics.transform(
        metric=mlprogram.metrics.TestCaseResult(
            interpreter=interpreter,
            metric=mlprogram.metrics.Iou(),
        ),
        transform=Threshold(
            threshold=0.9,
            dtype="float",
        ),
    ),
    collate=collate_fn,
    batch_size=batch_size,
    n_rollout=n_rollout,
    length=mlprogram.entrypoint.train.Iteration(
        n=option.n_train_iteration,
    ),
    evaluation_interval=mlprogram.entrypoint.train.Iteration(
        n=option.interval_iter,
    ),
    snapshot_interval=mlprogram.entrypoint.train.Iteration(
        n=1000,
    ),
    use_pretrained_model=True,
    device=device,
)
