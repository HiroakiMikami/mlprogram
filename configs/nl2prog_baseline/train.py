imports = ["base.py"]
device = torch.device(type_str="cpu", index=0)
transform = mlprogram.functools.Sequence(
    funcs=collections.OrderedDict(
        items=[
            [
                "set_train",
                Apply(module=Constant(value=True), in_keys=[], out_key="train"),
            ],
            ["transform_input", transform_input],
            [
                "transform_code",
                Apply(
                    module=mlprogram.transforms.action_sequence.GroundTruthToActionSequence(
                        parser=parser,
                    ),
                    in_keys=["ground_truth"],
                    out_key="action_sequence",
                ),
            ],
            ["transform_action_sequence", transform_action_sequence],
            [
                "transform_ground_truth",
                Apply(
                    module=mlprogram.transforms.action_sequence.EncodeActionSequence(
                        action_sequence_encoder=encoder.action_sequence_encoder,
                    ),
                    in_keys=["action_sequence", "reference"],
                    out_key="ground_truth_actions",
                ),
            ],
        ],
    ),
)
optimizer = torch.optim.Optimizer(
    optimizer_cls=torch.optim.Adam(),
    model=model,
)
main = mlprogram.entrypoint.train_supervised(
    output_dir=train_artifact_dir,
    dataset=train_dataset,
    model=model,
    optimizer=optimizer,
    loss=torch.nn.Sequential(
        modules=collections.OrderedDict(
            items=[
                [
                    "loss",
                    Apply(
                        module=mlprogram.nn.action_sequence.Loss(),
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
                        f=Pick(key="action_sequence_loss"),
                    ),
                ],
            ],
        ),
    ),
    evaluate=mlprogram.entrypoint.EvaluateSynthesizer(
        dataset=test_dataset,
        synthesizer=synthesizer,
        metrics=metrics,
        top_n=params.metric_top_n,
    ),
    metric=train_params.metric,
    threshold=train_params.metric_threshold,
    collate=mlprogram.functools.Compose(
        funcs=collections.OrderedDict(
            items=[
                [
                    "transform",
                    mlprogram.functools.Map(func=transform),
                ],
                ["collate", collate.collate],
            ],
        ),
    ),
    batch_size=train_params.batch_size,
    length=mlprogram.entrypoint.train.Epoch(
        n=train_params.n_epoch,
    ),
    evaluation_interval=mlprogram.entrypoint.train.Epoch(
        n=train_params.eval_interval,
    ),
    snapshot_interval=mlprogram.entrypoint.train.Epoch(
        n=train_params.snapshot_interval,
    ),
    device=device,
)
