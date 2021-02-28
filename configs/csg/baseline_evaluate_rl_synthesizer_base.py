imports = ["baseline_base.py"]
output_dir = "output/output"

option = {"n_rollout": 100}

device = torch.device(
    type_str="cpu",
    index=0,
)
# rl synthesizer
rl_optimizer = torch.optim.Optimizer(
    optimizer_cls=torch.optim.Adam(),
    model=model,
    lr=1e-5,
)
action_sequence_loss_fn = Apply(
    module=mlprogram.nn.action_sequence.Loss(reduction="none"),
    in_keys=[
        "rule_probs",
        "token_probs",
        "reference_probs",
        "ground_truth_actions",
    ],
    out_key="loss",
)
rl_loss_fn = torch.nn.Sequential(
    modules=collections.OrderedDict(
        items=[
            ["loss", action_sequence_loss_fn],
            [
                "weight_by_reward",
                Apply(
                    in_keys=[
                        ["reward", "lhs"],
                        ["loss", "rhs"],
                    ],
                    out_key="loss",
                    module=mlprogram.nn.Function(f=Mul()),
                ),
            ],
            [
                "entropy_loss",
                Apply(
                    module=mlprogram.nn.action_sequence.EntropyLoss(reduction="none"),
                    in_keys=[
                        "rule_probs",
                        "token_probs",
                        "reference_probs",
                    ],
                    out_key="entropy_loss",
                ),
            ],
            [
                "neg",
                Apply(
                    in_keys=[["entropy_loss", "lhs"]],
                    out_key="entropy_loss",
                    module=mlprogram.nn.Function(f=Mul()),
                    constants={"rhs": -0.05},
                ),
            ],
            [
                "aggregate",
                Apply(
                    in_keys=["loss", "entropy_loss"],
                    out_key="loss",
                    module=mlprogram.nn.AggregatedLoss(),
                ),
            ],
            ["pick", mlprogram.nn.Function(f=Pick(key="loss"))],
        ],
    ),
)
rl_reward_fn = mlprogram.metrics.use_environment(
    metric=mlprogram.metrics.TestCaseResult(
        interpreter=interpreter,
        metric=mlprogram.metrics.use_environment(
            metric=mlprogram.metrics.Iou(),
            in_keys=["expected", "actual"],
            value_key="actual",
        ),
    ),
    in_keys=["test_cases", "actual"],
    value_key="actual",
)

# overwrite configs
collate = mlprogram.utils.data.Collate(
    test_case_tensor=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=False,
        dim=0,
        padding_value=0,
    ),
    input_feature=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=False,
        dim=0,
        padding_value=0,
    ),
    reference_features=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=True,
        dim=0,
        padding_value=0,
    ),
    previous_actions=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=True,
        dim=0,
        padding_value=-1,
    ),
    hidden_state=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=False,
        dim=0,
        padding_value=0,
    ),
    state=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=False,
        dim=0,
        padding_value=0,
    ),
    ground_truth_actions=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=True,
        dim=0,
        padding_value=-1,
    ),
    reward=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=False,
        dim=0,
        padding_value=0,
    ),
)
collate_fn = mlprogram.functools.Sequence(
    funcs=collections.OrderedDict(
        items=[
            [
                "add_test_cases",
                mlprogram.functools.Map(
                    func=Apply(
                        module=mlprogram.languages.csg.transforms.AddTestCases(
                            interpreter=interpreter,
                        ),
                        in_keys=["ground_truth"],
                        out_key="test_cases",
                        is_out_supervision=False,
                    ),
                ),
            ],
            [
                "transform",
                mlprogram.functools.Map(func=transform),
            ],
            ["collate", collate.collate],
        ],
    ),
)
base_synthesizer = mlprogram.synthesizers.REINFORCESynthesizer(
    synthesizer=mlprogram.synthesizers.SMC(
        max_step_size=mul(
            x=5,
            y=mul(
                x=5,
                y=dataset_option.evaluate_max_object,
            ),
        ),
        initial_particle_size=1,
        max_try_num=1,
        sampler=mlprogram.samplers.FilteredSampler(
            sampler=sampler,
            score=mlprogram.metrics.use_environment(
                metric=mlprogram.metrics.TestCaseResult(
                    interpreter=interpreter,
                    metric=mlprogram.metrics.use_environment(
                        metric=mlprogram.metrics.Iou(),
                        in_keys=["expected", "actual"],
                        value_key="actual",
                    ),
                ),
                in_keys=["test_cases", "actual"],
                value_key="actual",
            ),
            threshold=0.9,
        ),
        to_key=Pick(key="action_sequence"),
    ),
    model=model,
    optimizer=rl_optimizer,
    loss_fn=rl_loss_fn,
    reward=rl_reward_fn,
    collate=collate_fn,
    n_rollout=option.n_rollout,
    device=device,
    baseline_momentum=0.9,  # disable baseline
    max_try_num=10,
)
main = mlprogram.entrypoint.evaluate(
    workspace_dir="output/workspace",
    input_dir=output_dir,
    output_dir=output_dir,
    valid_dataset=valid_dataset,
    model=model,
    synthesizer=synthesizer,
    metrics={},
    top_n=[],
    device=device,
)
