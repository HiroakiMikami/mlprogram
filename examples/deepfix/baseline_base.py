import os
from collections import OrderedDict

import torch
import torchnlp

import mlprogram
import mlprogram.datasets.deepfix as deepfix
import mlprogram.languages as languages
import mlprogram.languages.c as c
import mlprogram.languages.linediff as linediff
from mlprogram import nn
from mlprogram import transforms as T
from mlprogram.builtins import Apply, Constant, Pick
from mlprogram.functools import file_cache
from mlprogram.launch import global_options
from mlprogram.tasks.train import Epoch

# options for datasets
global_options["word_threshold"] = 3
global_options["token_threshold"] = 100
global_options["max_mutation"] = 5

# options for models
global_options["node_type_embedding_size"] = 64
global_options["embedding_size"] = 128
global_options["hidden_size"] = 256
global_options["attr_hidden_size"] = 50
global_options["dropout"] = 0.2

# options for training
global_options["batch_size"] = 8
global_options["n_epoch"] = Epoch(50)
global_options["eval_interval"] = Epoch(10)
global_options["snapshot_interval"] = Epoch(1)

# options for evaluation
global_options["timeout_sec"] = 10

# option for seed
global_options["seed"] = 0
global_options["mutation_seed"] = 1

# options for directory
global_options["train_artifact_dir"] = os.path.join("artifacts", "train")
global_options["evaluate_artifact_dir"] = os.path.join("artifacts", "evaluate")

# define components
lexer = languages.LexerWithLineNumber(deepfix.Lexer())
analyzer = c.Analyzer()
parser = linediff.Parser(lexer=lexer)
is_subtype = linediff.IsSubtype()
interpreter = linediff.Interpreter()
expander = linediff.Expander()
typo_mutator = c.TypoMutator(
    max_mutation=global_options.max_mutation, seed=global_options.mutation_seed
)
mutator = Apply(
    in_keys=["code"],
    out_key=["code", "test_cases", "ground_truth"],
    module=nn.Function(f=typo_mutator.mutate),  # convert to nn.Module
)

# define dataset
dataset = mlprogram.utils.data.split_by_n_error(
    dataset=deepfix.download(),
    analyzer=analyzer,
    n_process=4,
)
splitted = mlprogram.utils.data.random_split(
    dataset=dataset["no_error"],
    ratio={"train": 0.8, "test": 0.2},
    seed=global_options.seed,
)
train_dataset = mlprogram.utils.data.transform(
    dataset=splitted["train"], transform=mutator
)
test_dataset = mlprogram.utils.data.transform(
    dataset=splitted["test"], transform=mutator
)
valid_dataset = mlprogram.utils.data.transform(
    dataset=dataset["with_error"], transform=linediff.AddTestCases()
)


# define encoders
@file_cache(os.path.join(global_options.train_artifact_dir, "word_encoder.pt"))
def mk_word_encoder():
    return torchnlp.encoders.LabelEncoder(
        # TODO use not seeded dataset
        sample=mlprogram.utils.data.get_words(
            dataset=splitted["train"],
            extract_reference=lexer.tokenize,
            query_key="code",
        ),
        min_occurrences=global_options.word_threshold,
    )


@file_cache(
    os.path.join(global_options.train_artifact_dir, "action_sequence_encoder.pt")
)
def mk_action_sequence_encoder():
    return mlprogram.encoders.ActionSequenceEncoder(
        samples=mlprogram.utils.data.get_samples(dataset=train_dataset, parser=parser),
        token_threshold=global_options.token_threshold,
    )


word_encoder = mk_word_encoder()
action_sequence_encoder = mk_action_sequence_encoder()

# define module
embedding = nn.action_sequence.ActionsEmbedding(
    n_rule=action_sequence_encoder._rule_encoder.vocab_size,
    n_token=action_sequence_encoder._token_encoder.vocab_size,
    n_node_type=action_sequence_encoder._node_type_encoder.vocab_size,
    node_type_embedding_size=global_options.node_type_embedding_size,
    embedding_size=global_options.embedding_size,
)
encoder = torch.nn.Sequential(OrderedDict(
    embedding=Apply(
        module=nn.EmbeddingWithMask(
            n_id=word_encoder.vocab_size,
            embedding_size=global_options.embedding_size,
            ignore_id=-1,
        ),
        in_keys=[["word_nl_query", "x"]],
        out_key="word_nl_feature",
    ),
    lstm=Apply(
        module=nn.BidirectionalLSTM(
            input_size=global_options.embedding_size,
            hidden_size=global_options.hidden_size,
            dropout=global_options.dropout,
        ),
        in_keys=[["word_nl_feature", "x"]],
        out_key="reference_features",
    ),
))
decoder = torch.nn.Sequential(OrderedDict(
    embedding=Apply(
        module=embedding,
        in_keys=["actions", "previous_actions"],
        out_key="action_features",
    ),
    decoder=Apply(
        module=nn.action_sequence.LSTMTreeDecoder(
            inject_input=nn.action_sequence.AttentionInput(
                attn_hidden_size=global_options.attr_hidden_size
            ),
            input_feature_size=global_options.hidden_size,
            action_feature_size=embedding.output_size,
            output_feature_size=global_options.hidden_size,
            dropout=global_options.dropout,
        ),
        in_keys=[
            ["reference_features", "input_feature"],
            "actions",
            "action_features",
            "history",
            "hidden_state",
            "state",
        ],
        out_key=["action_features", "history", "hidden_state", "state"],
    ),
    predictor=Apply(
        module=nn.action_sequence.Predictor(
            feature_size=global_options.hidden_size,
            reference_feature_size=global_options.hidden_size,
            hidden_size=global_options.attr_hidden_size,
            rule_size=action_sequence_encoder._rule_encoder.vocab_size,
            token_size=action_sequence_encoder._token_encoder.vocab_size,
        ),
        in_keys=["reference_features", "action_features"],
        out_key=["rule_probs", "token_probs", "reference_probs"],
    ),
))

model = torch.nn.Sequential(OrderedDict(encoder=encoder, decoder=decoder))
_sequence = mlprogram.utils.data.CollateOptions(
    use_pad_sequence=True, dim=0, padding_value=-1
)
_tensor = mlprogram.utils.data.CollateOptions(
    use_pad_sequence=False, dim=0, padding_value=0
)
collate = mlprogram.utils.data.Collate(
    word_nl_query=_sequence,
    nl_query_features=_sequence,
    reference_features=_sequence,
    actions=_sequence,
    previous_actions=_sequence,
    previous_action_rules=_sequence,
    history=mlprogram.utils.data.CollateOptions(
        use_pad_sequence=False, dim=1, padding_value=0
    ),
    hidden_state=_tensor,
    state=_tensor,
    ground_truth_actions=_sequence,
)
to_episode = linediff.ToEpisode(interpreter=interpreter, expander=expander)
transform_input = mlprogram.functools.Compose(OrderedDict(
    update_input=linediff.UpdateInput(),
    extract_reference=Apply(
        module=nn.Function(f=lexer.tokenize),
        in_keys=[["code", "text"]],
        out_key="reference",
    ),
    encode_word_query=Apply(
        module=T.text.EncodeWordQuery(word_encoder=word_encoder),
        in_keys=["reference"],
        out_key="word_nl_query",
    ),
))
transform_action_sequence = mlprogram.functools.Compose(OrderedDict(
    add_previous_action=Apply(
        module=T.action_sequence.AddPreviousActions(
            action_sequence_encoder=action_sequence_encoder,
            n_dependent=1,
        ),
        in_keys=["action_sequence", "reference", "train"],
        out_key="previous_actions",
    ),
    add_action=Apply(
        module=T.action_sequence.AddActions(
            action_sequence_encoder=action_sequence_encoder,
            n_dependent=1,
        ),
        in_keys=["action_sequence", "reference", "train"],
        out_key="actions",
    ),
    add_state=T.action_sequence.AddState(key="state"),
    add_hidden_state=T.action_sequence.AddState(key="hidden_state"),
    add_history=T.action_sequence.AddState(key="history"),
))
transform = mlprogram.functools.Sequence(OrderedDict(
    set_train=Apply(module=Constant(value=True), in_keys=[], out_key="train"),
    transform_input=transform_input,
    transform_code=Apply(
        module=T.action_sequence.GroundTruthToActionSequence(parser=parser),
        in_keys=["ground_truth"],
        out_key="action_sequence",
    ),
    transform_action_sequence=transform_action_sequence,
    transform_ground_truth=Apply(
        module=T.action_sequence.EncodeActionSequence(
            action_sequence_encoder=action_sequence_encoder,
        ),
        in_keys=["action_sequence", "reference"],
        out_key="ground_truth_actions",
    ),
))
subsampler = mlprogram.samplers.transform(
    sampler=mlprogram.samplers.ActionSequenceSampler(
        encoder=action_sequence_encoder,
        is_subtype=is_subtype,
        transform_input=transform_input,
        transform_action_sequence=mlprogram.functools.Sequence(OrderedDict(
            set_train=Apply(module=Constant(value=False), in_keys=[], out_key="train"),
            transform=transform_action_sequence,
        )),
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
    to_key=Pick(key="action_sequence"),
)
sampler = mlprogram.samplers.SequentialProgramSampler(
    synthesizer=subsynthesizer,
    transform_input=nn.Function(f=mlprogram.functools.Identity()),
    collate=collate,
    encoder=nn.Function(f=mlprogram.functools.Identity()),
    interpreter=interpreter,
    expander=expander,
)
synthesizer = mlprogram.synthesizers.FilteredSynthesizer(
    synthesizer=mlprogram.synthesizers.SynthesizerWithTimeout(
        synthesizer=mlprogram.synthesizers.SMC(
            max_step_size=global_options.max_mutation * 3,
            initial_particle_size=1,
            max_try_num=1,
            sampler=sampler,
            to_key=Pick(key="interpreter_state"),
        ),
        timeout_sec=global_options.timeout_sec,
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
