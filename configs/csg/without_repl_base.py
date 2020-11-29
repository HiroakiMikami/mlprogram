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
                                    mlprogram.nn.Apply(
                                        in_keys=[["state@test_case_tensor", "x"]],
                                        out_key="state@input_feature",
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
                                    mlprogram.nn.Apply(
                                        in_keys=[["state@input_feature", "input"]],
                                        out_key="state@input_feature",
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
                                    "action_sequence_reader",
                                    mlprogram.nn.action_sequence.ActionSequenceReader(
                                        n_rule=encoder._rule_encoder.vocab_size,
                                        n_token=encoder._token_encoder.vocab_size,
                                        hidden_size=256,
                                    ),
                                ],
                                [
                                    "decoder",
                                    mlprogram.nn.action_sequence.RnnDecoder(
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
                                ],
                                [
                                    "predictor",
                                    mlprogram.nn.action_sequence.Predictor(
                                        feature_size=512,
                                        reference_feature_size=1,
                                        rule_size=encoder._rule_encoder.vocab_size,
                                        token_size=encoder._token_encoder.vocab_size,
                                        hidden_size=512,
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
                mlprogram.transforms.action_sequence.AddEmptyReference(),
            ],
            ["transform_canvas", mlprogram.languages.csg.transforms.TransformCanvas()],
        ],
    ),
)
transform_action_sequence = mlprogram.functools.Sequence(
    funcs=collections.OrderedDict(
        items=[
            [
                "add_previous_actions",
                mlprogram.transforms.action_sequence.AddPreviousActions(
                    action_sequence_encoder=encoder,
                    n_dependent=1,
                ),
            ],
            [
                "add_state",
                mlprogram.transforms.action_sequence.AddStateForRnnDecoder(),
            ],
        ],
    ),
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
                    action_sequence_encoder=encoder,
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
        transform_action_sequence=transform_action_sequence,
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
                key="state@action_sequence",
            ),
        ),
        timeout_sec=option.timeout_sec,
    ),
    score=mlprogram.metrics.TestCaseResult(
        interpreter=interpreter,
        metric=mlprogram.metrics.Iou(),
    ),
    threshold=0.9,
)
