imports = ["base.py"]
params = {
    "word_threshold": 3,
    "token_threshold": 0,
    "max_word_length": 128,
    "max_arity": 128,
    "max_tree_depth": 128,
    "char_embedding_size": 256,
    "rule_embedding_size": 256,
    "hidden_size": 256,
    "decoder_hidden_size": 1024,
    "tree_conv_kernel_size": 3,
    "n_head": 1,
    "n_block": 6,
    "dropout": 0.15,
    "batch_size": 1,
    "n_epoch": 25,
    "eval_interval": 10,
    "snapshot_interval": 1,
    "beam_size": 15,
    "max_step_size": 250,
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
    "char_encoder": with_file_cache(
        path=os.path.join(
            args=[output_dir, "char_encoder.pt"],
        ),
        config=torchnlp.encoders.LabelEncoder(
            sample=mlprogram.utils.data.get_characters(
                dataset=train_dataset,
                extract_reference=extract_reference,
            ),
            min_occurrences=0,
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
                                    module=mlprogram.nn.treegen.NlEmbedding(
                                        n_token=encoder.word_encoder.vocab_size,
                                        n_char=encoder.char_encoder.vocab_size,
                                        max_token_length=params.max_word_length,
                                        char_embedding_size=params.char_embedding_size,
                                        embedding_size=params.hidden_size,
                                    ),
                                    in_keys=["word_nl_query", "char_nl_query"],
                                    out_key=["word_nl_feature", "char_nl_feature"],
                                ),
                            ],
                            [
                                "encoder",
                                Apply(
                                    module=mlprogram.nn.treegen.Encoder(
                                        char_embedding_size=params.char_embedding_size,
                                        hidden_size=params.hidden_size,
                                        n_head=params.n_head,
                                        dropout=params.dropout,
                                        n_block=params.n_block,
                                    ),
                                    in_keys=["word_nl_feature", "char_nl_feature"],
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
                                "query_embedding",
                                Apply(
                                    module=mlprogram.nn.treegen.QueryEmbedding(
                                        n_rule=encoder.action_sequence_encoder._rule_encoder.vocab_size,
                                        max_depth=params.max_tree_depth,
                                        embedding_size=params.hidden_size,
                                    ),
                                    in_keys=["action_queries"],
                                    out_key="action_query_features",
                                ),
                            ],
                            [
                                "action_embedding",
                                Apply(
                                    module=mlprogram.nn.treegen.ActionEmbedding(
                                        n_rule=encoder.action_sequence_encoder._rule_encoder.vocab_size,
                                        n_token=encoder.action_sequence_encoder._token_encoder.vocab_size,
                                        n_node_type=encoder.action_sequence_encoder._node_type_encoder.vocab_size,
                                        max_arity=params.max_arity,
                                        rule_embedding_size=params.rule_embedding_size,
                                        embedding_size=params.hidden_size,
                                    ),
                                    in_keys=[
                                        "previous_actions",
                                        "previous_action_rules",
                                    ],
                                    out_key=[
                                        "action_features",
                                        "action_rule_features",
                                    ],
                                ),
                            ],
                            [
                                "decoder",
                                Apply(
                                    module=mlprogram.nn.treegen.Decoder(
                                        rule_embedding_size=params.rule_embedding_size,
                                        encoder_hidden_size=params.hidden_size,
                                        decoder_hidden_size=params.decoder_hidden_size,
                                        out_size=params.hidden_size,
                                        tree_conv_kernel_size=params.tree_conv_kernel_size,
                                        n_head=params.n_head,
                                        dropout=params.dropout,
                                        n_encoder_block=params.n_block,
                                        n_decoder_block=params.n_block,
                                    ),
                                    in_keys=[
                                        ["reference_features", "nl_query_features"],
                                        "action_query_features",
                                        "action_features",
                                        "action_rule_features",
                                        "depthes",
                                        "adjacency_matrix",
                                    ],
                                    out_key="action_features",
                                ),
                            ],
                            [
                                "predictor",
                                Apply(
                                    module=mlprogram.nn.action_sequence.Predictor(
                                        feature_size=params.hidden_size,
                                        reference_feature_size=params.hidden_size,
                                        rule_size=encoder.action_sequence_encoder._rule_encoder.vocab_size,
                                        token_size=encoder.action_sequence_encoder._token_encoder.vocab_size,
                                        hidden_size=params.hidden_size,
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
)
collate_as_sequence = mlprogram.utils.data.CollateOptions(
    use_pad_sequence=True,
    dim=0,
    padding_value=-1,
)
collate = mlprogram.utils.data.Collate(
    word_nl_query=collate_as_sequence,
    char_nl_query=collate_as_sequence,
    nl_query_features=collate_as_sequence,
    reference_features=collate_as_sequence,
    actions=collate_as_sequence,
    previous_actions=collate_as_sequence,
    previous_action_rules=collate_as_sequence,
    depthes=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=False,
        dim=1,
        padding_value=0,
    ),
    adjacency_matrix=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=False,
        dim=0,
        padding_value=0,
    ),
    action_queries=collate_as_sequence,
    ground_truth_actions=collate_as_sequence,
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
            [
                "encode_char",
                Apply(
                    module=mlprogram.transforms.text.EncodeCharacterQuery(
                        char_encoder=encoder.char_encoder,
                        max_word_length=params.max_word_length,
                    ),
                    in_keys=["reference"],
                    out_key="char_nl_query",
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
                    ),
                    in_keys=["action_sequence", "reference", "train"],
                    out_key="previous_actions",
                ),
            ],
            [
                "add_previous_action_rule",
                Apply(
                    module=mlprogram.transforms.action_sequence.AddPreviousActionRules(
                        action_sequence_encoder=encoder.action_sequence_encoder,
                        max_arity=params.max_arity,
                    ),
                    in_keys=["action_sequence", "reference", "train"],
                    out_key="previous_action_rules",
                ),
            ],
            [
                "add_action_sequence_as_tree",
                Apply(
                    module=mlprogram.transforms.action_sequence.AddActionSequenceAsTree(
                        action_sequence_encoder=encoder.action_sequence_encoder,
                    ),
                    in_keys=["action_sequence", "reference", "train"],
                    out_key=["adjacency_matrix", "depthes"],
                ),
            ],
            [
                "add_query",
                Apply(
                    module=mlprogram.transforms.action_sequence.AddQueryForTreeGenDecoder(
                        action_sequence_encoder=encoder.action_sequence_encoder,
                        max_depth=params.max_tree_depth,
                    ),
                    in_keys=["action_sequence", "reference", "train"],
                    out_key="action_queries",
                ),
            ],
        ],
    ),
)
synthesizer = mlprogram.synthesizers.BeamSearch(
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
