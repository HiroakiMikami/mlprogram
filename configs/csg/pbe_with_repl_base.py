imports = ["base.py"]

options = {
    "small": {
        "n_pretrain_iteration": 5000,
        "n_train_iteration": 15000,
        "train_max_object": 3,
        "evaluate_max_object": 6,
        "size": 4,
        "resolution": 4,
        "n_evaluate_dataset": 30,
        "timeout_sec": 5,
        "interval_iter": 3000,
    },
    "large": {
        "n_pretrain_iteration": 156250,
        "n_train_iteration": 281250,
        "train_max_object": 13,
        "evaluate_max_object": 30,
        "size": 16,
        "resolution": 1,
        "n_evaluate_dataset": 30,
        "timeout_sec": 120,
        "interval_iter": 50000,
    },
}
reference = True
model = torch.share_memory_(
    model=torch.nn.Sequential(
        modules=collections.OrderedDict(
            items=[
                [
                    "encode_input",
                    Apply(
                        in_keys=[["test_case_tensor", "x"]],
                        out_key="test_case_feature",
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
                    "encoder",
                    Apply(
                        module=mlprogram.nn.pbe_with_repl.Encoder(
                            module=mlprogram.nn.CNN2d(
                                in_channel=2,
                                out_channel=16,
                                hidden_channel=32,
                                n_conv_per_block=2,
                                n_block=2,
                                pool=2,
                            ),
                        ),
                        in_keys=[
                            "test_case_tensor",
                            "variables_tensor",
                            "test_case_feature",
                        ],
                        out_key=["reference_features", "input_feature"],
                    ),
                ],
                [
                    "decoder",
                    torch.nn.Sequential(
                        modules=collections.OrderedDict(
                            items=[
                                [
                                    "action_embedding",
                                    Apply(
                                        module=mlprogram.nn.action_sequence.PreviousActionsEmbedding(
                                            n_rule=encoder._rule_encoder.vocab_size,
                                            n_token=encoder._token_encoder.vocab_size,
                                            embedding_size=256,
                                        ),
                                        in_keys=["previous_actions"],
                                        out_key="action_features",
                                    ),
                                ],
                                [
                                    "decoder",
                                    Apply(
                                        module=mlprogram.nn.action_sequence.LSTMDecoder(
                                            inject_input=mlprogram.nn.action_sequence.CatInput(),
                                            input_feature_size=mul(
                                                x=32,
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
                                            "action_features",
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
                                            reference_feature_size=mul(
                                                x=16,
                                                y=mul(
                                                    x=n_feature_pixel,
                                                    y=n_feature_pixel,
                                                ),
                                            ),
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
                [
                    "value",
                    Apply(
                        in_keys=[["input_feature", "x"]],
                        out_key="value",
                        module=mlprogram.nn.MLP(
                            in_channel=mul(
                                x=2,
                                y=mul(
                                    x=16,
                                    y=mul(
                                        x=n_feature_pixel,
                                        y=n_feature_pixel,
                                    ),
                                ),
                            ),
                            out_channel=1,
                            hidden_channel=512,
                            n_linear=2,
                            activation=torch.nn.Sigmoid(),
                        ),
                    ),
                ],
            ],
        ),
    ),
)
collate = mlprogram.utils.data.Collate(
    test_case_tensor=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=False,
        dim=0,
        padding_value=0,
    ),
    test_case_feature=mlprogram.utils.data.CollateOptions(
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
    variables_tensor=mlprogram.utils.data.CollateOptions(
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
transform_input = mlprogram.functools.Sequence(
    funcs=collections.OrderedDict(
        items=[
            [
                "transform_inputs",
                Apply(
                    module=mlprogram.languages.csg.transforms.TransformInputs(),
                    in_keys=["test_cases"],
                    out_key="test_case_tensor",
                ),
            ],
            [
                "transform_variables",
                Apply(
                    module=mlprogram.languages.csg.transforms.TransformVariables(),
                    in_keys=["variables", "test_case_tensor"],
                    out_key="variables_tensor",
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
to_episode = mlprogram.transforms.pbe.ToEpisode(
    interpreter=interpreter,
    expander=mlprogram.languages.csg.Expander(),
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

subsampler = mlprogram.samplers.transform(
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
subsynthesizer = mlprogram.synthesizers.SMC(
    max_step_size=5,
    max_try_num=1,
    initial_particle_size=1,
    sampler=subsampler,
    to_key=Pick(
        key="action_sequence",
    ),
)
sampler = mlprogram.samplers.SequentialProgramSampler(
    synthesizer=subsynthesizer,
    transform_input=Apply(
        module=mlprogram.languages.csg.transforms.TransformInputs(),
        in_keys=["test_cases"],
        out_key="test_case_tensor",
    ),
    collate=collate,
    encoder=model.encode_input,
    interpreter=interpreter,
    expander=mlprogram.languages.csg.Expander(),
)
train_synthesizer = mlprogram.synthesizers.SMC(
    max_step_size=mul(
        x=option.train_max_object,
        y=3,
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
    to_key=Pick(
        key="interpreter_state",
    ),
)
evaluate_synthesizer = mlprogram.synthesizers.FilteredSynthesizer(
    synthesizer=mlprogram.synthesizers.SynthesizerWithTimeout(
        synthesizer=mlprogram.synthesizers.SMC(
            max_step_size=mul(
                x=option.evaluate_max_object,
                y=5,
            ),
            initial_particle_size=100,
            sampler=mlprogram.samplers.SamplerWithValueNetwork(
                sampler=sampler,
                transform=transform_input,
                collate=collate,
                value_network=torch.nn.Sequential(
                    modules=collections.OrderedDict(
                        items=[
                            ["encoder", model.encoder],
                            ["value", model.value],
                            [
                                "pick",
                                mlprogram.nn.Function(
                                    f=Pick(
                                        key="value",
                                    ),
                                ),
                            ],
                        ],
                    ),
                ),
                batch_size=1,
            ),
            to_key=Pick(
                key="interpreter_state",
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
