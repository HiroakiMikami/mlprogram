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
model = torch.share_memory_(
    model=torch.nn.Sequential(
        modules=collections.OrderedDict(
            items=[
                [
                    "encoder",
                    Apply(
                        module=mlprogram.nn.treegen.NLReader(
                            token_num=encoder.word_encoder.vocab_size,
                            char_num=encoder.char_encoder.vocab_size,
                            max_token_len=params.max_word_length,
                            char_embed_size=params.char_embedding_size,
                            hidden_size=params.hidden_size,
                            n_heads=params.n_head,
                            dropout=params.dropout,
                            n_blocks=params.n_block,
                        ),
                        in_keys=["word_nl_query", "char_nl_query"],
                        out_key="reference_features",
                    ),
                ],
                [
                    "decoder",
                    torch.nn.Sequential(
                        modules=collections.OrderedDict(
                            items=[
                                [
                                    "action_sequence_reader",
                                    Apply(
                                        module=mlprogram.nn.treegen.ActionSequenceReader(
                                            rule_num=encoder.action_sequence_encoder._rule_encoder.vocab_size,
                                            token_num=encoder.action_sequence_encoder._token_encoder.vocab_size,
                                            node_type_num=encoder.action_sequence_encoder._node_type_encoder.vocab_size,
                                            max_arity=params.max_arity,
                                            rule_embed_size=params.rule_embedding_size,
                                            hidden_size=params.hidden_size,
                                            tree_conv_kernel_size=params.tree_conv_kernel_size,
                                            n_heads=params.n_head,
                                            dropout=params.dropout,
                                            n_blocks=params.n_block,
                                        ),
                                        in_keys=[
                                            "previous_actions",
                                            "previous_action_rules",
                                            "depthes",
                                            "adjacency_matrix",
                                        ],
                                        out_key="action_features",
                                    ),
                                ],
                                [
                                    "decoder",
                                    Apply(
                                        module=mlprogram.nn.treegen.Decoder(
                                            rule_num=encoder.action_sequence_encoder._rule_encoder.vocab_size,
                                            max_depth=params.max_tree_depth,
                                            feature_size=params.hidden_size,
                                            hidden_size=params / decoder_hidden_size,
                                            out_size=params.hidden_size,
                                            n_heads=params.n_head,
                                            dropout=params.dropout,
                                            n_blocks=params.n_block,
                                        ),
                                        in_keys=[
                                            ["reference_features", "nl_query_features"],
                                            "action_queries",
                                            "action_features",
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
    ),
)
collate = mlprogram.utils.data.Collate(
    device=device,
    word_nl_query=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=True,
        dim=0,
        padding_value=-1,
    ),
    char_nl_query=mlprogram.utils.data.CollateOptions(
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
    action_queries=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=True,
        dim=0,
        padding_value=-1,
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
