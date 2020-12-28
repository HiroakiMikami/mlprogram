import collections
import os

import mlprogram.actions
import mlprogram.builtins
import mlprogram.datasets.deepfix
import mlprogram.datasets.django
import mlprogram.datasets.hearthstone
import mlprogram.datasets.nl2bash
import mlprogram.encoders
import mlprogram.entrypoint
import mlprogram.functools
import mlprogram.languages.bash
import mlprogram.languages.c
import mlprogram.languages.csg
import mlprogram.languages.csg.transforms
import mlprogram.languages.python
import mlprogram.languages.python.metrics
import mlprogram.metrics
import mlprogram.nn
import mlprogram.nn.action_sequence
import mlprogram.nn.nl2code
import mlprogram.nn.pbe_with_repl
import mlprogram.nn.treegen
import mlprogram.synthesizers
import mlprogram.transforms
import mlprogram.transforms.action_sequence
import mlprogram.transforms.pbe
import mlprogram.transforms.text
import mlprogram.utils.data
from mlprogram.entrypoint.modules.numpy import types as numpy_types
from mlprogram.entrypoint.modules.torch import types as torch_types
from mlprogram.entrypoint.modules.torchnlp import types as torchnlp_types

try:
    from mlprogram.entrypoint.modules.fairseq import types as fairseq_types
except:  # noqa
    fairseq_types = {}
    pass

types = {
    "select": lambda key, options: options[key],

    "Identity": mlprogram.builtins.Identity,
    "Flatten": mlprogram.builtins.Flatten,
    "Threshold": mlprogram.builtins.Threshold,
    "Apply": mlprogram.builtins.Apply,
    "Constant": mlprogram.builtins.Constant,
    "Pack": mlprogram.builtins.Pack,
    "Pick": mlprogram.builtins.Pick,
    "Add": mlprogram.builtins.Add,
    "add": lambda **kwargs: mlprogram.builtins.Add()(**kwargs),
    "Sub": mlprogram.builtins.Sub,
    "sub": lambda lhs, rhs: mlprogram.builtins.Sub()(lhs, rhs),
    "Mul": mlprogram.builtins.Mul,
    "mul": lambda **kwargs: mlprogram.builtins.Mul()(**kwargs),
    "Div": mlprogram.builtins.Div,
    "div": lambda lhs, rhs: mlprogram.builtins.Div()(lhs, rhs),
    "IntDiv": mlprogram.builtins.IntDiv,
    "intdiv": lambda lhs, rhs: mlprogram.builtins.IntDiv()(lhs, rhs),
    "Neg": mlprogram.builtins.Neg,

    "gt": lambda x, y: x > y,
    "ge": lambda x, y: x >= y,
    "lt": lambda x, y: x < y,
    "le": lambda x, y: x <= y,
    "eq": lambda x, y: x == y,
    "ne": lambda x, y: x != y,

    "collections.OrderedDict": lambda items: collections.OrderedDict(items),
    "os.path.join": lambda args: os.path.join(*args),

    "mlprogram.entrypoint.train.Epoch":
        mlprogram.entrypoint.train.Epoch,
    "mlprogram.entrypoint.train.Iteration":
        mlprogram.entrypoint.train.Iteration,
    "mlprogram.entrypoint.train_supervised":
        mlprogram.entrypoint.train_supervised,
    "mlprogram.entrypoint.train_REINFORCE":
        mlprogram.entrypoint.train_REINFORCE,
    "mlprogram.entrypoint.EvaluateSynthesizer":
        mlprogram.entrypoint.EvaluateSynthesizer,
    "mlprogram.entrypoint.evaluate": mlprogram.entrypoint.evaluate,

    "mlprogram.datasets.django.download": mlprogram.datasets.django.download,
    "mlprogram.datasets.django.Parser": mlprogram.datasets.django.Parser,
    "mlprogram.datasets.django.SplitValue":
        mlprogram.datasets.django.SplitValue,
    "mlprogram.datasets.django.TokenizeQuery":
        mlprogram.datasets.django.TokenizeQuery,
    "mlprogram.datasets.hearthstone.download":
        mlprogram.datasets.hearthstone.download,
    "mlprogram.datasets.hearthstone.TokenizeQuery":
        mlprogram.datasets.hearthstone.TokenizeQuery,
    "mlprogram.datasets.hearthstone.SplitValue":
        mlprogram.datasets.hearthstone.SplitValue,
    "mlprogram.datasets.nl2bash.download": mlprogram.datasets.nl2bash.download,
    "mlprogram.datasets.nl2bash.TokenizeQuery":
        mlprogram.datasets.nl2bash.TokenizeQuery,
    "mlprogram.datasets.nl2bash.SplitValue":
        mlprogram.datasets.nl2bash.SplitValue,
    "mlprogram.datasets.deepfix.download":
        mlprogram.datasets.deepfix.download,
    "mlprogram.datasets.deepfix.Lexer":
        mlprogram.datasets.deepfix.Lexer,

    "mlprogram.metrics.use_environment": mlprogram.metrics.use_environment,
    "mlprogram.metrics.Accuracy": mlprogram.metrics.Accuracy,
    "mlprogram.metrics.Bleu": mlprogram.metrics.Bleu,
    "mlprogram.metrics.Iou": mlprogram.metrics.Iou,
    "mlprogram.metrics.TestCaseResult": mlprogram.metrics.TestCaseResult,
    "mlprogram.metrics.ErrorCorrectRate": mlprogram.metrics.ErrorCorrectRate,

    "mlprogram.languages.LexerWithLineNumber": mlprogram.languages.LexerWithLineNumber,

    "mlprogram.languages.python.Parser": mlprogram.languages.python.Parser,
    "mlprogram.languages.python.IsSubtype":
        mlprogram.languages.python.IsSubtype,
    "mlprogram.languages.python.metrics.Bleu":
        mlprogram.languages.python.metrics.Bleu,

    "mlprogram.languages.bash.Parser": mlprogram.languages.bash.Parser,
    "mlprogram.languages.bash.IsSubtype": mlprogram.languages.bash.IsSubtype,

    "mlprogram.functools.Compose": mlprogram.functools.Compose,
    "mlprogram.functools.Map": mlprogram.functools.Map,
    "mlprogram.functools.Sequence": mlprogram.functools.Sequence,
    "mlprogram.functools.Identity": mlprogram.functools.Identity,

    "mlprogram.synthesizers.BeamSearch": mlprogram.synthesizers.BeamSearch,
    "mlprogram.synthesizers.SMC": mlprogram.synthesizers.SMC,
    "mlprogram.synthesizers.FilteredSynthesizer":
        mlprogram.synthesizers.FilteredSynthesizer,
    "mlprogram.synthesizers.SynthesizerWithTimeout":
        mlprogram.synthesizers.SynthesizerWithTimeout,
    "mlprogram.samplers.transform": mlprogram.samplers.transform,
    "mlprogram.samplers.ActionSequenceSampler":
        mlprogram.samplers.ActionSequenceSampler,
    "mlprogram.samplers.SequentialProgramSampler":
        mlprogram.samplers.SequentialProgramSampler,
    "mlprogram.samplers.SamplerWithValueNetwork":
        mlprogram.samplers.SamplerWithValueNetwork,
    "mlprogram.samplers.FilteredSampler":
        mlprogram.samplers.FilteredSampler,

    "mlprogram.utils.data.Collate": mlprogram.utils.data.Collate,
    "mlprogram.utils.data.CollateOptions": mlprogram.utils.data.CollateOptions,
    "mlprogram.utils.data.get_words": mlprogram.utils.data.get_words,
    "mlprogram.utils.data.get_characters": mlprogram.utils.data.get_characters,
    "mlprogram.utils.data.get_samples": mlprogram.utils.data.get_samples,
    "mlprogram.utils.data.to_map_style_dataset":
        mlprogram.utils.data.to_map_style_dataset,
    "mlprogram.utils.data.random_split":
        mlprogram.utils.data.random.random_split,
    "mlprogram.utils.data.transform": mlprogram.utils.data.transform,
    "mlprogram.utils.data.split_by_n_error": mlprogram.utils.data.split_by_n_error,

    "mlprogram.transforms.NormalizeGroundTruth":
        mlprogram.transforms.NormalizeGroundTruth,
    "mlprogram.transforms.action_sequence.AddEmptyReference":
        mlprogram.transforms.action_sequence.AddEmptyReference,
    "mlprogram.transforms.action_sequence.AddPreviousActions":
        mlprogram.transforms.action_sequence.AddPreviousActions,
    "mlprogram.transforms.action_sequence.AddActions":
        mlprogram.transforms.action_sequence.AddActions,
    "mlprogram.transforms.action_sequence.AddPreviousActionRules":
        mlprogram.transforms.action_sequence.AddPreviousActionRules,
    "mlprogram.transforms.action_sequence.AddActionSequenceAsTree":
        mlprogram.transforms.action_sequence.AddActionSequenceAsTree,
    "mlprogram.transforms.action_sequence.AddQueryForTreeGenDecoder":
        mlprogram.transforms.action_sequence.AddQueryForTreeGenDecoder,
    "mlprogram.transforms.action_sequence.AddState":
        mlprogram.transforms.action_sequence.AddState,
    "mlprogram.transforms.action_sequence.GroundTruthToActionSequence":
        mlprogram.transforms.action_sequence.GroundTruthToActionSequence,
    "mlprogram.transforms.action_sequence.EncodeActionSequence":
        mlprogram.transforms.action_sequence.EncodeActionSequence,
    "mlprogram.transforms.text.EncodeWordQuery":
        mlprogram.transforms.text.EncodeWordQuery,
    "mlprogram.transforms.text.EncodeTokenQuery":
        mlprogram.transforms.text.EncodeTokenQuery,
    "mlprogram.transforms.text.EncodeCharacterQuery":
        mlprogram.transforms.text.EncodeCharacterQuery,
    "mlprogram.transforms.pbe.ToEpisode":
        mlprogram.transforms.pbe.ToEpisode,

    "mlprogram.encoders.ActionSequenceEncoder":
        mlprogram.encoders.ActionSequenceEncoder,

    "mlprogram.nn.BidirectionalLSTM": mlprogram.nn.BidirectionalLSTM,
    "mlprogram.nn.AggregatedLoss": mlprogram.nn.AggregatedLoss,
    "mlprogram.nn.Function": mlprogram.nn.Function,
    "mlprogram.nn.CNN2d": mlprogram.nn.CNN2d,
    "mlprogram.nn.MLP": mlprogram.nn.MLP,
    "mlprogram.nn.action_sequence.LSTMDecoder":
        mlprogram.nn.action_sequence.LSTMDecoder,
    "mlprogram.nn.action_sequence.Predictor":
        mlprogram.nn.action_sequence.Predictor,
    "mlprogram.nn.action_sequence.Loss": mlprogram.nn.action_sequence.Loss,
    "mlprogram.nn.action_sequence.Accuracy":
        mlprogram.nn.action_sequence.Accuracy,
    "mlprogram.nn.nl2code.Decoder": mlprogram.nn.nl2code.Decoder,
    "mlprogram.nn.nl2code.Predictor": mlprogram.nn.nl2code.Predictor,
    "mlprogram.nn.treegen.Encoder": mlprogram.nn.treegen.Encoder,
    "mlprogram.nn.treegen.Decoder": mlprogram.nn.treegen.Decoder,
    "mlprogram.nn.pbe_with_repl.Encoder": mlprogram.nn.pbe_with_repl.Encoder,

    "mlprogram.languages.csg.Parser": mlprogram.languages.csg.Parser,
    "mlprogram.languages.csg.Dataset": mlprogram.languages.csg.Dataset,
    "mlprogram.languages.csg.Interpreter": mlprogram.languages.csg.Interpreter,
    "mlprogram.languages.csg.Expander": mlprogram.languages.csg.Expander,
    "mlprogram.languages.csg.IsSubtype": mlprogram.languages.csg.IsSubtype,
    "mlprogram.languages.csg.get_samples": mlprogram.languages.csg.get_samples,
    "mlprogram.languages.csg.transforms.TransformInputs":
        mlprogram.languages.csg.transforms.TransformInputs,
    "mlprogram.languages.csg.transforms.TransformVariables":
        mlprogram.languages.csg.transforms.TransformVariables,
    "mlprogram.languages.csg.transforms.AddTestCases":
        mlprogram.languages.csg.transforms.AddTestCases,

    "mlprogram.languages.c.Analyzer": mlprogram.languages.c.Analyzer,
    "mlprogram.languages.c.Lexer": mlprogram.languages.c.Lexer,
    "mlprogram.languages.c.TypoMutator": mlprogram.languages.c.TypoMutator,

    "mlprogram.languages.linediff.Interpreter":
    mlprogram.languages.linediff.Interpreter,
    "mlprogram.languages.linediff.Expander": mlprogram.languages.linediff.Expander,
    "mlprogram.languages.linediff.Parser": mlprogram.languages.linediff.Parser,
    "mlprogram.languages.linediff.IsSubtype": mlprogram.languages.linediff.IsSubtype,
    "mlprogram.languages.linediff.get_samples":
    mlprogram.languages.linediff.get_samples,
    "mlprogram.languages.linediff.ToEpisode":
    mlprogram.languages.linediff.ToEpisode,
    "mlprogram.languages.linediff.AddTestCases":
    mlprogram.languages.linediff.AddTestCases,
    "mlprogram.languages.linediff.UpdateInput":
    mlprogram.languages.linediff.UpdateInput,

}

types.update(torch_types)
types.update(torchnlp_types)
types.update(fairseq_types)
types.update(numpy_types)
