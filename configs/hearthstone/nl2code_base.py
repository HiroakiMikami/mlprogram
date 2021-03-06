imports = ["base.py"]

params = {
    "word_threshold": 3,
    "token_threshold": 0,
    "node_type_embedding_size": 64,
    "embedding_size": 128,
    "hidden_size": 256,
    "attr_hidden_size": 50,
    "dropout": 0.2,
    "batch_size": 1,
    "n_epoch": 50,
    "eval_interval": 10,
    "snapshot_interval": 1,
    "beam_size": 15,
    "max_step_size": 350,
    "metric_top_n": [1],
    "metric_threshold": 1.0,
    "metric": "bleu@1",
}

normalize_dataset = Apply(
    module=mlprogram.transforms.NormalizeGroundTruth(
        normalize=mlprogram.functools.Sequence(
            funcs=collections.OrderedDict(
                items=[["parse", parser.parse], ["unparse", parser.unparse]],
            ),
        )
    ),
    in_keys=["ground_truth"],
    out_key="ground_truth",
)
train_dataset = dataset.train
test_dataset = mlprogram.utils.data.transform(
    dataset=dataset.test,
    transform=normalize_dataset,
)
valid_dataset = mlprogram.utils.data.transform(
    dataset=dataset.valid,
    transform=normalize_dataset,
)
encoder = {
    "word_encoder": with_file_cache(
        path=os.path.join(
            args=[output_dir, "word_encoder.pt"],
        ),
        config=torchnlp.encoders.LabelEncoder(
            sample=mlprogram.utils.data.get_words(
                dataset=train_dataset,
                extract_reference=extract_reference,
            ),
            min_occurrences=params.word_threshold,
        ),
    ),
    "action_sequence_encoder": with_file_cache(
        path=os.path.join(
            args=[output_dir, "action_sequence_encoder.pt"],
        ),
        config=mlprogram.encoders.ActionSequenceEncoder(
            samples=mlprogram.utils.data.get_samples(
                dataset=train_dataset,
                parser=parser,
            ),
            token_threshold=params.token_threshold,
        ),
    ),
}
embedding = mlprogram.nn.action_sequence.ActionsEmbedding(
    n_rule=encoder.action_sequence_encoder._rule_encoder.vocab_size,
    n_token=encoder.action_sequence_encoder._token_encoder.vocab_size,
    n_node_type=encoder.action_sequence_encoder._node_type_encoder.vocab_size,
    node_type_embedding_size=params.node_type_embedding_size,
    embedding_size=params.embedding_size,
)
decoder = mlprogram.nn.nl2code.Decoder(
    input_size=embedding.output_size,
    query_size=params.hidden_size,
    hidden_size=params.hidden_size,
    att_hidden_size=params.attr_hidden_size,
    dropout=params.dropout,
)
model = torch.nn.Sequential(
    modules=collections.OrderedDict(
        items=[
            [
                "encoder",
                torch.nn.Sequential(
                    modules=collections.OrderedDict(
                        items=[
                            [
                                "embedding",
                                Apply(
                                    module=mlprogram.nn.EmbeddingWithMask(
                                        n_id=encoder.word_encoder.vocab_size,
                                        embedding_size=params.embedding_size,
                                        ignore_id=-1,
                                    ),
                                    in_keys=[["word_nl_query", "x"]],
                                    out_key="word_nl_feature",
                                ),
                            ],
                            [
                                "lstm",
                                Apply(
                                    module=mlprogram.nn.BidirectionalLSTM(
                                        input_size=params.embedding_size,
                                        hidden_size=params.hidden_size,
                                        dropout=params.dropout,
                                    ),
                                    in_keys=[["word_nl_feature", "x"]],
                                    out_key="reference_features",
                                ),
                            ],
                        ]
                    )
                ),
            ],
            [
                "decoder",
                torch.nn.Sequential(
                    modules=collections.OrderedDict(
                        items=[
                            [
                                "embedding",
                                Apply(
                                    module=embedding,
                                    in_keys=[
                                        "actions",
                                        "previous_actions",
                                    ],
                                    out_key="action_features",
                                ),
                            ],
                            [
                                "decoder",
                                Apply(
                                    module=decoder,
                                    in_keys=[
                                        ["reference_features", "nl_query_features"],
                                        "actions",
                                        "action_features",
                                        "history",
                                        "hidden_state",
                                        "state",
                                    ],
                                    out_key=[
                                        "action_features",
                                        "action_contexts",
                                        "history",
                                        "hidden_state",
                                        "state",
                                    ],
                                ),
                            ],
                            [
                                "predictor",
                                Apply(
                                    module=mlprogram.nn.nl2code.Predictor(
                                        embedding=embedding,
                                        embedding_size=params.embedding_size,
                                        query_size=params.hidden_size,
                                        hidden_size=params.hidden_size,
                                        att_hidden_size=params.attr_hidden_size,
                                    ),
                                    in_keys=[
                                        "reference_features",
                                        "action_features",
                                        "action_contexts",
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
)
collate = mlprogram.utils.data.Collate(
    word_nl_query=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=True,
        dim=0,
        padding_value=-1,
    ),
    nl_query_features=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=True,
        dim=0,
        padding_value=-1,
    ),
    reference_features=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=True,
        dim=0,
        padding_value=-1,
    ),
    actions=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=True,
        dim=0,
        padding_value=-1,
    ),
    previous_actions=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=True,
        dim=0,
        padding_value=-1,
    ),
    previous_action_rules=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=True,
        dim=0,
        padding_value=-1,
    ),
    history=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=False,
        dim=1,
        padding_value=0,
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
transform_input = mlprogram.functools.Compose(
    funcs=collections.OrderedDict(
        items=[
            [
                "extract_reference",
                Apply(
                    module=mlprogram.nn.Function(f=extract_reference),
                    in_keys=[["text_query", "query"]],
                    out_key="reference",
                ),
            ],
            [
                "encode_word_query",
                Apply(
                    module=mlprogram.transforms.text.EncodeWordQuery(
                        word_encoder=encoder.word_encoder,
                    ),
                    in_keys=["reference"],
                    out_key="word_nl_query",
                ),
            ],
        ],
    ),
)
transform_action_sequence = mlprogram.functools.Compose(
    funcs=collections.OrderedDict(
        items=[
            [
                "add_previous_action",
                Apply(
                    module=mlprogram.transforms.action_sequence.AddPreviousActions(
                        action_sequence_encoder=encoder.action_sequence_encoder,
                        n_dependent=1,
                    ),
                    in_keys=["action_sequence", "reference", "train"],
                    out_key="previous_actions",
                ),
            ],
            [
                "add_action",
                Apply(
                    module=mlprogram.transforms.action_sequence.AddActions(
                        action_sequence_encoder=encoder.action_sequence_encoder,
                        n_dependent=1,
                    ),
                    in_keys=["action_sequence", "reference", "train"],
                    out_key="actions",
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
            [
                "add_history",
                mlprogram.transforms.action_sequence.AddState(key="history"),
            ],
        ],
    ),
)
base_synthesizer = mlprogram.synthesizers.BeamSearch(
    beam_size=params.beam_size,
    max_step_size=params.max_step_size,
    sampler=mlprogram.samplers.transform(
        sampler=mlprogram.samplers.ActionSequenceSampler(
            encoder=encoder.action_sequence_encoder,
            is_subtype=is_subtype,
            transform_input=transform_input,
            transform_action_sequence=mlprogram.functools.Sequence(
                funcs=collections.OrderedDict(
                    items=[
                        [
                            "set_train",
                            Apply(
                                module=Constant(value=False),
                                in_keys=[],
                                out_key="train",
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
    ),
)
