imports = ["base.py"]

options = {
    "small": {
        "n_pretrain_iteration": 35000,
        "train_max_object": 3,
        "evaluate_max_object": 6,
        "size": 4,
        "resolution": 4,
        "n_evaluate_dataset": 30,
        "timeout_sec": 5,
        "interval_iter": 5000,
    },
    "large": {
        "n_pretrain_iteration": 437500,
        "train_max_object": 13,
        "evaluate_max_object": 30,
        "size": 16,
        "resolution": 1,
        "n_evaluate_dataset": 30,
        "timeout_sec": 120,
        "interval_iter": 50000,
    },
}
reference = False
model = torch.share_memory_(
    model=torch.nn.Sequential(
        modules=collections.OrderedDict(
            items=[
                [
                    "encoder",
                    torch.nn.Sequential(
                        modules=collections.OrderedDict(
                            items=[
                                [
                                    "encoder",
                                    Apply(
                                        in_keys=[["test_case_tensor", "x"]],
                                        out_key="input_feature",
                                        module=mlprogram.nn.CNN2d(
                                            in_channel=1,
                                            out_channel=16,
                                            hidden_channel=32,
                                            n_conv_per_block=2,
                                            n_block=2,
                                            pool=2,
                                        ),
                                    ),
                                ],
                                [
                                    "reduction",
                                    Apply(
                                        in_keys=[["input_feature", "input"]],
                                        out_key="input_feature",
                                        module=torch.Mean(
                                            dim=1,
                                        ),
                                    ),
                                ],
                            ],
                        ),
                    ),
                ],
                [
                    "decoder",
                    torch.nn.Sequential(
                        modules=collections.OrderedDict(
                            items=[
                                [
                                    "decoder",
                                    Apply(
                                        module=mlprogram.nn.action_sequence.LSTMDecoder(
                                            n_rule=encoder._rule_encoder.vocab_size,
                                            n_token=encoder._token_encoder.vocab_size,
                                            input_feature_size=mul(
                                                x=16,
                                                y=mul(
                                                    x=n_feature_pixel,
                                                    y=n_feature_pixel,
                                                ),
                                            ),
                                            action_feature_size=256,
                                            output_feature_size=512,
                                            dropout=0.1,
                                        ),
                                        in_keys=[
                                            "input_feature",
                                            "previous_actions",
                                            "hidden_state",
                                            "state",
                                        ],
                                        out_key=[
                                            "action_features",
                                            "hidden_state",
                                            "state",
                                        ],
                                    ),
                                ],
                                [
                                    "predictor",
                                    Apply(
                                        module=mlprogram.nn.action_sequence.Predictor(
                                            feature_size=512,
                                            reference_feature_size=1,
                                            rule_size=encoder._rule_encoder.vocab_size,
                                            token_size=encoder._token_encoder.vocab_size,
                                            hidden_size=512,
                                        ),
                                        in_keys=[
                                            "reference_features",
                                            "action_features",
                                        ],
                                        out_key=[
                                            "rule_probs",
                                            "token_probs",
                                            "reference_probs",
                                        ],
                                    ),
                                ],
                            ],
                        ),
                    ),
                ],
            ],
        ),
    ),
)
collate = mlprogram.utils.data.Collate(
    device=device,
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
)
transform_input = mlprogram.functools.Sequence(
    funcs=collections.OrderedDict(
        items=[
            [
                "add_reference",
                Apply(
                    module=mlprogram.transforms.action_sequence.AddEmptyReference(),
                    in_keys=[],
                    out_key=["reference", "reference_features"],
                ),
            ],
            [
                "transform_canvas",
                Apply(
                    module=mlprogram.languages.csg.transforms.TransformInputs(),
                    in_keys=["test_cases"],
                    out_key="test_case_tensor",
                ),
            ],
        ],
    ),
)
transform_action_sequence = mlprogram.functools.Sequence(
    funcs=collections.OrderedDict(
        items=[
            [
                "add_previous_actions",
                Apply(
                    module=mlprogram.transforms.action_sequence.AddPreviousActions(
                        action_sequence_encoder=encoder,
                        n_dependent=1,
                    ),
                    in_keys=["action_sequence", "reference", "train"],
                    out_key="previous_actions",
                ),
            ],
            [
                "add_state",
                mlprogram.transforms.action_sequence.AddState(key="state"),
            ],
            [
                "add_hidden_state",
                mlprogram.transforms.action_sequence.AddState(key="hidden_state"),
            ],
        ],
    ),
)
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
                        action_sequence_encoder=encoder,
                    ),
                    in_keys=["action_sequence", "reference"],
                    out_key="ground_truth_actions",
                ),
            ],
        ],
    ),
)
sampler = mlprogram.samplers.transform(
    sampler=mlprogram.samplers.ActionSequenceSampler(
        encoder=encoder,
        is_subtype=mlprogram.languages.csg.IsSubtype(),
        transform_input=transform_input,
        transform_action_sequence=mlprogram.functools.Sequence(
            funcs=collections.OrderedDict(
                items=[
                    [
                        "set_train",
                        Apply(
                            module=Constant(value=False), in_keys=[], out_key="train"
                        ),
                    ],
                    ["transform", transform_action_sequence],
                ]
            )
        ),
        collate=collate,
        module=model,
    ),
    transform=parser.unparse,
)
synthesizer = mlprogram.synthesizers.FilteredSynthesizer(
    synthesizer=mlprogram.synthesizers.SynthesizerWithTimeout(
        synthesizer=mlprogram.synthesizers.SMC(
            max_step_size=mul(
                x=5,
                y=mul(
                    x=5,
                    y=option.evaluate_max_object,
                ),
            ),
            initial_particle_size=100,
            max_try_num=50,
            sampler=sampler,
            to_key=Pick(
                key="action_sequence",
            ),
        ),
        timeout_sec=option.timeout_sec,
    ),
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
)
