dataset_params = {
    "word_threshold": 3,
    "token_threshold": 100,
}
model_params = {
    "node_type_embedding_size": 64,
    "embedding_size": 128,
    "hidden_size": 256,
    "attr_hidden_size": 50,
    "dropout": 0.2,
}
train_params = {
    "batch_size": 8,
    "n_epoch": 50,
    "eval_interval": 10,
    "snapshot_interval": 1,
}
params = {
    "seed": 0,
    "max_mutation": 5,
    "mutation_seed": 1,
    "timeout_sec": 10,
}
train_artifact_dir = "artifacts/train"
evaluate_artifact_dir = "artifacts/evaluate"
lexer = mlprogram.languages.LexerWithLineNumber(
    lexer=mlprogram.datasets.deepfix.Lexer()
)
analyzer = mlprogram.languages.c.Analyzer()
parser = mlprogram.languages.linediff.Parser(lexer=lexer)
is_subtype = mlprogram.languages.linediff.IsSubtype()
interpreter = mlprogram.languages.linediff.Interpreter()
expander = mlprogram.languages.linediff.Expander()

dataset = mlprogram.utils.data.split_by_n_error(
    dataset=mlprogram.datasets.deepfix.download(), analyzer=analyzer, n_process=4
)
typo_mutator = mlprogram.languages.c.TypoMutator(
    max_mutation=params.max_mutation, seed=params.mutation_seed
)
splitted = mlprogram.utils.data.random_split(
    dataset=dataset.no_error, ratio={"train": 0.8, "test": 0.2}, seed=params.seed
)
mutator = Apply(
    in_keys=["code"],
    out_key=["code", "test_cases", "ground_truth"],
    module=mlprogram.nn.Function(f=typo_mutator.mutate),  # convert to nn.Module
)
train_dataset = mlprogram.utils.data.transform(
    dataset=splitted.train,
    transform=mutator,
)
test_dataset = mlprogram.utils.data.transform(
    dataset=splitted.test,
    transform=mutator,
)
valid_dataset = mlprogram.utils.data.transform(
    dataset=dataset.with_error,
    transform=mlprogram.languages.linediff.AddTestCases(),
)

encoder = {
    "word_encoder": with_file_cache(
        path=os.path.join(
            args=[train_artifact_dir, "word_encoder.pt"],
        ),
        config=torchnlp.encoders.LabelEncoder(
            # TODO use not seeded dataset
            sample=mlprogram.utils.data.get_words(
                dataset=splitted.train,
                extract_reference=lexer.tokenize,
                query_key="code",
            ),
            min_occurrences=dataset_params.word_threshold,
        ),
    ),
    "action_sequence_encoder": with_file_cache(
        path=os.path.join(
            args=[train_artifact_dir, "action_sequence_encoder.pt"],
        ),
        config=mlprogram.encoders.ActionSequenceEncoder(
            samples=mlprogram.utils.data.get_samples(
                dataset=train_dataset,
                parser=parser,
            ),
            token_threshold=dataset_params.token_threshold,
        ),
    ),
}
embedding = mlprogram.nn.action_sequence.ActionsEmbedding(
    n_rule=encoder.action_sequence_encoder._rule_encoder.vocab_size,
    n_token=encoder.action_sequence_encoder._token_encoder.vocab_size,
    n_node_type=encoder.action_sequence_encoder._node_type_encoder.vocab_size,
    node_type_embedding_size=model_params.node_type_embedding_size,
    embedding_size=model_params.embedding_size,
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
                                        embedding_size=model_params.embedding_size,
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
                                        input_size=model_params.embedding_size,
                                        hidden_size=model_params.hidden_size,
                                        dropout=model_params.dropout,
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
                                    module=mlprogram.nn.action_sequence.LSTMTreeDecoder(
                                        inject_input=mlprogram.nn.action_sequence.AttentionInput(
                                            attn_hidden_size=model_params.attr_hidden_size
                                        ),
                                        input_feature_size=model_params.hidden_size,
                                        action_feature_size=embedding.output_size,
                                        output_feature_size=model_params.hidden_size,
                                        dropout=model_params.dropout,
                                    ),
                                    in_keys=[
                                        ["reference_features", "input_feature"],
                                        "actions",
                                        "action_features",
                                        "history",
                                        "hidden_state",
                                        "state",
                                    ],
                                    out_key=[
                                        "action_features",
                                        "history",
                                        "hidden_state",
                                        "state",
                                    ],
                                ),
                            ],
                            [
                                "predictor",
                                Apply(
                                    module=mlprogram.nn.action_sequence.Predictor(
                                        feature_size=model_params.hidden_size,
                                        reference_feature_size=model_params.hidden_size,
                                        hidden_size=model_params.attr_hidden_size,
                                        rule_size=encoder.action_sequence_encoder._rule_encoder.vocab_size,
                                        token_size=encoder.action_sequence_encoder._token_encoder.vocab_size,
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
to_episode = mlprogram.languages.linediff.ToEpisode(
    interpreter=interpreter,
    expander=expander,
)
transform_input = mlprogram.functools.Compose(
    funcs=collections.OrderedDict(
        items=[
            [
                "update_input",
                mlprogram.languages.linediff.UpdateInput(),
            ],
            [
                "extract_reference",
                Apply(
                    module=mlprogram.nn.Function(f=lexer.tokenize),
                    in_keys=[["code", "text"]],
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
subsampler = mlprogram.samplers.transform(
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
    max_step_size=50,
    max_try_num=1,
    initial_particle_size=1,
    sampler=subsampler,
    to_key=Pick(
        key="action_sequence",
    ),
)
sampler = mlprogram.samplers.SequentialProgramSampler(
    synthesizer=subsynthesizer,
    transform_input=mlprogram.nn.Function(f=mlprogram.functools.Identity()),
    collate=collate,
    encoder=mlprogram.nn.Function(f=mlprogram.functools.Identity()),
    interpreter=interpreter,
    expander=expander,
)
synthesizer = mlprogram.synthesizers.FilteredSynthesizer(
    synthesizer=mlprogram.synthesizers.SynthesizerWithTimeout(
        synthesizer=mlprogram.synthesizers.SMC(
            max_step_size=mul(
                x=params.max_mutation,
                y=3,
            ),
            initial_particle_size=1,
            max_try_num=1,
            sampler=sampler,
            to_key=Pick(
                key="interpreter_state",
            ),
        ),
        timeout_sec=params.timeout_sec,
    ),
    score=mlprogram.metrics.use_environment(
        metric=mlprogram.metrics.ErrorCorrectRate(
            interpreter=interpreter,
            analyzer=analyzer,
        ),
        in_keys=["test_cases", "actual"],
        value_key="actual",
    ),
    threshold=1.0,
)
