import os
from collections import OrderedDict

import torch
import torchnlp

import mlprogram
import mlprogram.transforms as T
from mlprogram.builtins import Apply, Constant
from mlprogram.functools import with_file_cache
from mlprogram.launch import global_options


def setup(*, dataset, parser, extract_reference, is_subtype):
    normalize_dataset = Apply(
        module=T.NormalizeGroundTruth(
            normalize=mlprogram.functools.Sequence(
                OrderedDict(parse=parser.parse, unparse=parser.unparse)
            )
        ),
        in_keys=["ground_truth"],
        out_key="ground_truth",
    )
    train_dataset = dataset["train"]
    test_dataset = mlprogram.utils.data.transform(
        dataset=dataset["test"], transform=normalize_dataset
    )
    valid_dataset = mlprogram.utils.data.transform(
        dataset=dataset["valid"], transform=normalize_dataset
    )

    # define encoders
    word_encoder = with_file_cache(
        os.path.join(global_options.train_artifact_dir, "word_encoder.pt"),
        torchnlp.encoders.LabelEncoder,
        sample=mlprogram.utils.data.get_words(
            dataset=train_dataset,
            extract_reference=extract_reference,
        ),
        min_occurrences=global_options.word_threshold,
    )
    action_sequence_encoder = with_file_cache(
        os.path.join(global_options.train_artifact_dir, "action_sequence_encoder.pt"),
        mlprogram.encoders.ActionSequenceEncoder,
        samples=mlprogram.utils.data.get_samples(
            dataset=train_dataset, parser=parser
        ),
        token_threshold=global_options.token_threshold,
    )

    embedding = mlprogram.nn.action_sequence.ActionsEmbedding(
        n_rule=action_sequence_encoder._rule_encoder.vocab_size,
        n_token=action_sequence_encoder._token_encoder.vocab_size,
        n_node_type=action_sequence_encoder._node_type_encoder.vocab_size,
        node_type_embedding_size=global_options.node_type_embedding_size,
        embedding_size=global_options.embedding_size,
    )
    model = torch.nn.Sequential(OrderedDict(
        encoder=torch.nn.Sequential(OrderedDict(
            embedding=Apply(
                module=mlprogram.nn.EmbeddingWithMask(
                    n_id=word_encoder.vocab_size,
                    embedding_size=global_options.embedding_size,
                    ignore_id=-1,
                ),
                in_keys=[["word_nl_query", "x"]],
                out_key="word_nl_feature",
            ),
            lstm=Apply(
                module=mlprogram.nn.BidirectionalLSTM(
                    input_size=global_options.embedding_size,
                    hidden_size=global_options.hidden_size,
                    dropout=global_options.dropout,
                ),
                in_keys=[["word_nl_feature", "x"]],
                out_key="reference_features",
            ),
        )),
        decoder=torch.nn.Sequential(OrderedDict(
            embedding=Apply(
                module=embedding,
                in_keys=["actions", "previous_actions"],
                out_key="action_features",
            ),
            decoder=Apply(
                module=mlprogram.nn.action_sequence.LSTMTreeDecoder(
                    inject_input=mlprogram.nn.action_sequence.AttentionInput(
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
                module=mlprogram.nn.action_sequence.Predictor(
                    feature_size=global_options.hidden_size,
                    reference_feature_size=global_options.hidden_size,
                    hidden_size=global_options.attr_hidden_size,
                    rule_size=action_sequence_encoder._rule_encoder.vocab_size,
                    token_size=action_sequence_encoder._token_encoder.vocab_size,
                ),
                in_keys=["reference_features", "action_features"],
                out_key=["rule_probs", "token_probs", "reference_probs"],
            ),
        )),
    ))
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
    transform_input = mlprogram.functools.Compose(OrderedDict(
        extract_reference=Apply(
            module=mlprogram.nn.Function(f=extract_reference),
            in_keys=[["text_query", "query"]],
            out_key="reference",
        ),
        encode_word_query=Apply(
            module=T.text.EncodeWordQuery(word_encoder=word_encoder),
            in_keys=["reference"],
            out_key="word_nl_query",
        ),
    ),
    )
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
    synthesizer = mlprogram.synthesizers.BeamSearch(
        beam_size=global_options.beam_size,
        max_step_size=global_options.max_step_size,
        sampler=mlprogram.samplers.transform(
            sampler=mlprogram.samplers.ActionSequenceSampler(
                encoder=action_sequence_encoder,
                is_subtype=is_subtype,
                transform_input=transform_input,
                transform_action_sequence=mlprogram.functools.Sequence(OrderedDict(
                    set_train=Apply(
                        module=Constant(value=False),
                        in_keys=[],
                        out_key="train",
                    ),
                    transform=transform_action_sequence,
                )),
                collate=collate,
                module=model,
            ),
            transform=parser.unparse,
        ),
    )

    return {
        "collate": collate,
        "transform_input": transform_input,
        "transform_action_sequence": transform_action_sequence,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "valid_dataset": valid_dataset,
        "word_encoder": word_encoder,
        "action_sequence_encoder": action_sequence_encoder,
        "model": model,
        "synthesizer": synthesizer,
    }
