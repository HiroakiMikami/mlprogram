imports = ["base.py"]
device = torch.device(
    type_str="cuda",
    index=0,
)
transform = mlprogram.functools.Sequence(
    funcs=collections.OrderedDict(
        items=[
            ["transform_input", transform_input],
            [
                "transform_code",
                mlprogram.transforms.action_sequence.GroundTruthToActionSequence(
                    parser=parser,
                ),
            ],
            ["transform_action_sequence", transform_action_sequence],
            [
                "transform_ground_truth",
                mlprogram.transforms.action_sequence.EncodeActionSequence(
                    action_sequence_encoder=encoder.action_sequence_encoder,
                ),
            ],
        ],
    ),
)
optimizer = torch.optim.Optimizer(
    optimizer_cls=fairseq.optim.Adafactor(),
    model=model,
)
main = mlprogram.entrypoint.train_supervised(
    workspace_dir="output/workspace",
    output_dir=output_dir,
    dataset=train_dataset,
    model=model,
    optimizer=optimizer,
    loss=torch.nn.Sequential(
        modules=collections.OrderedDict(
            items=[
                ["loss", mlprogram.nn.action_sequence.Loss()],
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
    ),
    evaluate=mlprogram.entrypoint.EvaluateSynthesizer(
        dataset=test_dataset,
        synthesizer=synthesizer,
        metrics=metrics,
        top_n=params.metric_top_n,
        n_process=params.n_evaluate_process,
    ),
    metric="bleu@1",
    threshold=params.metric_threshold,
    maximize=True,
    collate=mlprogram.functools.Compose(
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
    ),
    batch_size=params.batch_size,
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
