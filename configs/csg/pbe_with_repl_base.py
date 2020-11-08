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
device = torch.device(
    type_str="cuda",
    index=0,
)
parser = mlprogram.languages.csg.Parser()
n_pixel = mul(
    x=option.size,
    y=option.resolution,
)
n_feature_pixel = intdiv(
    lhs=n_pixel,
    rhs=2,
)
dataset = mlprogram.languages.csg.Dataset(
    canvas_size=option.size,
    min_object=1,
    max_object=option.train_max_object,
    length_stride=1,
    degree_stride=15,
    reference=True,
)
encoder = with_file_cache(
    path=os.path.join(
        args=[output_dir, "encoder.pt"],
    ),
    config=mlprogram.encoders.ActionSequenceEncoder(
        samples=mlprogram.languages.csg.get_samples(
            dataset=dataset,
            parser=parser,
        ),
        token_threshold=0,
    ),
)
train_dataset = mlprogram.utils.data.transform(
    dataset=dataset,
    transform=mlprogram.languages.csg.transform.AddTestCases(
        interpreter=interpreter,
    ),
)
test_dataset = mlprogram.utils.data.transform(
    dataset=mlprogram.utils.data.to_map_style_dataset(
        dataset=mlprogram.languages.csg.Dataset(
            canvas_size=option.size,
            min_object=option.train_max_object,
            max_object=option.evaluate_max_object,
            length_stride=1,
            degree_stride=15,
            reference=True,
            seed=10000,
        ),
        n=option.n_evaluate_dataset,
    ),
    transform=mlprogram.languages.csg.transform.AddTestCases(
        interpreter=interpreter,
    ),
)
valid_dataset = mlprogram.utils.data.transform(
    dataset=mlprogram.utils.data.to_map_style_dataset(
        dataset=mlprogram.languages.csg.Dataset(
            canvas_size=option.size,
            min_object=option.train_max_object,
            max_object=option.evaluate_max_object,
            length_stride=1,
            degree_stride=15,
            reference=True,
            seed=20000,
        ),
        n=option.n_evaluate_dataset,
    ),
    transform=mlprogram.languages.csg.transform.AddTestCases(
        interpreter=interpreter,
    ),
)
interpreter = mlprogram.languages.csg.Interpreter(
    width=option.size,
    height=option.size,
    resolution=option.resolution,
    delete_used_reference=True,
)
model = torch.share_memory_(
    model=torch.nn.Sequential(
        modules=collections.OrderedDict(
            items=[
                [
                    "encode_input",
                    mlprogram.nn.Apply(
                        in_keys=[["state@test_case_tensor", "x"]],
                        out_key="state@test_case_feature",
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
                    mlprogram.nn.pbe_with_repl.Encoder(
                        module=mlprogram.nn.CNN2d(
                            in_channel=2,
                            out_channel=16,
                            hidden_channel=32,
                            n_conv_per_block=2,
                            n_block=2,
                            pool=2,
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
                                ],
                                [
                                    "predictor",
                                    mlprogram.nn.action_sequence.Predictor(
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
                                ],
                            ],
                        ),
                    ),
                ],
                [
                    "value",
                    mlprogram.nn.Apply(
                        in_keys=[["state@input_feature", "x"]],
                        out_key="state@value",
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
                        value_type="tensor",
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
            ["transform_canvas", mlprogram.languages.csg.transform.TransformCanvas()]
        ],
    ),
)
transform_action_sequence = mlprogram.functools.Sequence(
    funcs=collections.OrderedDict(
        items=[
            [
                "add_previous_actions",
                mlprogram.utils.transform.action_sequence.AddPreviousActions(
                    action_sequence_encoder=encoder,
                    n_dependent=1,
                ),
            ],
            [
                "add_state",
                mlprogram.utils.transform.action_sequence.AddStateForRnnDecoder(),
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
                mlprogram.utils.transform.action_sequence.GroundTruthToActionSequence(
                    parser=parser,
                ),
            ],
            ["transform_action_sequence", transform_action_sequence],
            [
                "transform_ground_truth",
                mlprogram.utils.transform.action_sequence.EncodeActionSequence(
                    action_sequence_encoder=encoder,
                ),
            ],
        ],
    ),
)
subsampler = mlprogram.samplers.transform(
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
subsynthesizer = mlprogram.synthesizers.SMC(
    max_step_size=5,
    max_try_num=1,
    initial_particle_size=1,
    sampler=subsampler,
    to_key=Pick(
        key="state@action_sequence",
    ),
)
sampler = mlprogram.samplers.SequentialProgramSampler(
    synthesizer=subsynthesizer,
    transform_input=mlprogram.languages.csg.transform.TransformCanvas(),
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
        score=mlprogram.metrics.TestCaseResult(
            interpreter=interpreter,
            metric=mlprogram.metrics.Iou(),
        ),
        threshold=0.9,
    ),
    to_key=Pick(
        key="state@interpreter_state",
    ),
)
evaluate_synthesizer = mlprogram.synthesizers.SynthesizerWithTimeout(
    synthesizer=mlprogram.synthesizers.SMC(
        max_step_size=mul(
            x=option.evaluate_max_object,
            y=5,
        ),
        initial_particle_size=100,
        sampler=mlprogram.samplers.SamplerWithValueNetwork(
            sampler=sampler,
            transform=mlprogram.functools.Compose(
                funcs=collections.OrderedDict(
                    items=[
                        [
                            "transform_canvas",
                            mlprogram.languages.csg.transform.TransformCanvas(),
                        ]
                    ],
                ),
            ),
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
                                    key="state@value",
                                ),
                            ),
                        ],
                    ],
                ),
            ),
            batch_size=1,
        ),
        to_key=Pick(
            key="state@interpreter_state",
        ),
    ),
    timeout_sec=option.timeout_sec,
)
to_episode = mlprogram.utils.transform.pbe.ToEpisode(
    interpreter=interpreter,
    expander=mlprogram.languages.csg.Expander(),
)
