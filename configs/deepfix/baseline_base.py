params = {
    "word_threshold": 3,
    "node_type_embedding_size": 64,
    "embedding_size": 128,
    "hidden_size": 256,
    "attr_hidden_size": 50,
    "dropout": 0.2,
    "batch_size": 8,
    "n_epoch": 50,
    "eval_interval": 10,
    "snapshot_interval": 1,
    "particle_size": 15,
    "n_evaluate_process": 8,
    "seed": 0,
    "max_mutation": 5,
    "mutation_seed": 1,
    "timeout_sec": 10,
}
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
# TODO mutator is not nn.Module
mutator = mlprogram.nn.Apply(
    in_keys=[["input@code", "code"]],
    out_key=["input@text_query", "input@test_cases", "supervision@ground_truth"],
    module=typo_mutator.mutate,
)
train_dataset = mlprogram.utils.data.transform(
    dataset=splitted.train,
    transform=mutator,
)
test_dataset = mlprogram.utils.data.transform(
    dataset=splitted.test,
    transform=mutator,
)
valid_dataset = dataset.with_error

encoder = {
    "word_encoder": with_file_cache(
        path=os.path.join(
            args=[output_dir, "word_encoder.pt"],
        ),
        config=torchnlp.encoders.LabelEncoder(
            # TODO use not seeded dataset
            sample=mlprogram.utils.data.get_words(
                dataset=train_dataset,
                extract_reference=lexer.tokenize,
            ),
            min_occurrences=params.word_threshold,
        ),
    ),
    "action_sequence_encoder": with_file_cache(
        path=os.path.join(
            args=[output_dir, "action_sequence_encoder.pt"],
        ),
        config=mlprogram.encoders.ActionSequenceEncoder(
            samples=mlprogram.languages.linediff.get_samples(
                parser=parser,
            ),
            token_threshold=0,
        ),
    ),
}
action_sequence_reader = mlprogram.nn.nl2code.ActionSequenceReader(
    num_rules=encoder.action_sequence_encoder._rule_encoder.vocab_size,
    num_tokens=encoder.action_sequence_encoder._token_encoder.vocab_size,
    num_node_types=encoder.action_sequence_encoder._node_type_encoder.vocab_size,
    node_type_embedding_size=params.node_type_embedding_size,
    embedding_size=params.embedding_size,
)
model = torch.share_memory_(
    model=torch.nn.Sequential(
        modules=collections.OrderedDict(
            items=[
                [
                    "encoder",
                    mlprogram.nn.nl2code.NLReader(
                        num_words=encoder.word_encoder.vocab_size,
                        embedding_dim=params.embedding_size,
                        hidden_size=params.hidden_size,
                        dropout=params.dropout,
                    ),
                ],
                [
                    "decoder",
                    torch.nn.Sequential(
                        modules=collections.OrderedDict(
                            items=[
                                ["action_sequence_reader", action_sequence_reader],
                                [
                                    "decoder",
                                    mlprogram.nn.nl2code.Decoder(
                                        query_size=params.hidden_size,
                                        input_size=add(
                                            x=mul(
                                                x=2,
                                                y=params.embedding_size,
                                            ),
                                            y=params.node_type_embedding_size,
                                        ),
                                        hidden_size=params.hidden_size,
                                        att_hidden_size=params.attr_hidden_size,
                                        dropout=params.dropout,
                                    ),
                                ],
                                [
                                    "predictor",
                                    mlprogram.nn.action_sequence.Predictor(
                                        feature_size=params.hidden_size,
                                        reference_feature_size=params.hidden_size,
                                        hidden_size=params.hidden_size,
                                        rule_size=encoder.action_sequence_encoder._rule_encoder.vocab_size,
                                        token_size=encoder.action_sequence_encoder._token_encoder.vocab_size,
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
                mlprogram.utils.transform.text.ExtractReference(
                    extract_reference=lexer.tokenize,
                ),
            ],
            [
                "encode_word_query",
                mlprogram.utils.transform.text.EncodeWordQuery(
                    word_encoder=encoder.word_encoder,
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
                mlprogram.utils.transform.action_sequence.AddPreviousActions(
                    action_sequence_encoder=encoder.action_sequence_encoder,
                    n_dependent=1,
                ),
            ],
            [
                "add_action",
                mlprogram.utils.transform.action_sequence.AddActions(
                    action_sequence_encoder=encoder.action_sequence_encoder,
                    n_dependent=1,
                ),
            ],
            [
                "add_state",
                mlprogram.utils.transform.action_sequence.AddStateForRnnDecoder(),
            ],
            [
                "add_history",
                mlprogram.utils.transform.action_sequence.AddHistoryState(),
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
                    action_sequence_encoder=encoder.action_sequence_encoder,
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
        transform_action_sequence=transform_action_sequence,
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
        key="state@action_sequence",
    ),
)
sampler = mlprogram.samplers.SequentialProgramSampler(
    synthesizer=subsynthesizer,
    transform_input=mlprogram.languages.linediff.AddTestCases(),
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
                key="state@interpreter_state",
            ),
        ),
        timeout_sec=params.timeout_sec,
    ),
    score=mlprogram.metrics.ErrorCorrectRate(
        interpreter=interpreter,
        analyzer=analyzer,
    ),
    threshold=1.0,
)
