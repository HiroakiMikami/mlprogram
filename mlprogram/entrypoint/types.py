import os
import collections

import mlprogram.entrypoint
import mlprogram.builtins
import mlprogram.datasets.django
import mlprogram.datasets.hearthstone
import mlprogram.datasets.nl2bash
import mlprogram.datasets.deepfix
import mlprogram.functools
import mlprogram.nn
import mlprogram.nn.action_sequence
import mlprogram.metrics
import mlprogram.languages.python
import mlprogram.languages.python.metrics
import mlprogram.languages.bash
import mlprogram.languages.csg
import mlprogram.languages.csg.transform
import mlprogram.languages.c
import mlprogram.actions
import mlprogram.synthesizers
import mlprogram.utils.data
import mlprogram.utils.transform
import mlprogram.utils.transform.action_sequence
import mlprogram.encoders

import mlprogram.nn.nl2code
import mlprogram.utils.transform.nl2code

import mlprogram.nn.treegen
import mlprogram.utils.transform.treegen

import mlprogram.nn.pbe_with_repl
import mlprogram.utils.transform.pbe_with_repl

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

    "mlprogram.metrics.transform": mlprogram.metrics.transform,
    "mlprogram.metrics.Accuracy": mlprogram.metrics.Accuracy,
    "mlprogram.metrics.Bleu": mlprogram.metrics.Bleu,
    "mlprogram.metrics.Iou": mlprogram.metrics.Iou,
    "mlprogram.metrics.TestCaseResult": mlprogram.metrics.TestCaseResult,

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
    "mlprogram.utils.data.random.random_split":
        mlprogram.utils.data.random.random_split,
    "mlprogram.utils.data.transform": mlprogram.utils.data.transform,

    "mlprogram.utils.transform.NormalizeGroundTruth":
        mlprogram.utils.transform.NormalizeGroudTruth,
    "mlprogram.utils.transform.action_sequence.AddEmptyReference":
        mlprogram.utils.transform.action_sequence.AddEmptyReference,
    "mlprogram.utils.transform.action_sequence.TransformActionSequenceForRnnDecoder":  # noqa
        mlprogram.utils.transform.action_sequence.TransformActionSequenceForRnnDecoder,  # noqa
    "mlprogram.utils.transform.action_sequence.TransformCode":
        mlprogram.utils.transform.action_sequence.TransformCode,
    "mlprogram.utils.transform.action_sequence.TransformGroundTruth":
        mlprogram.utils.transform.action_sequence.TransformGroundTruth,
    "mlprogram.utils.transform.nl2code.TransformQuery":
        mlprogram.utils.transform.nl2code.TransformQuery,
    "mlprogram.utils.transform.nl2code.TransformActionSequence":
        mlprogram.utils.transform.nl2code.TransformActionSequence,
    "mlprogram.utils.transform.treegen.TransformQuery":
        mlprogram.utils.transform.treegen.TransformQuery,
    "mlprogram.utils.transform.treegen.TransformActionSequence":
        mlprogram.utils.transform.treegen.TransformActionSequence,
    "mlprogram.utils.transform.pbe_with_repl.ToEpisode":
        mlprogram.utils.transform.pbe_with_repl.ToEpisode,

    "mlprogram.encoders.ActionSequenceEncoder":
        mlprogram.encoders.ActionSequenceEncoder,

    "mlprogram.nn.Apply": mlprogram.nn.Apply,
    "mlprogram.nn.AggregatedLoss": mlprogram.nn.AggregatedLoss,
    "mlprogram.nn.Function": mlprogram.nn.Function,
    "mlprogram.nn.CNN2d": mlprogram.nn.CNN2d,
    "mlprogram.nn.MLP": mlprogram.nn.MLP,
    "mlprogram.nn.action_sequence.ActionSequenceReader":
        mlprogram.nn.action_sequence.ActionSequenceReader,
    "mlprogram.nn.action_sequence.RnnDecoder":
        mlprogram.nn.action_sequence.RnnDecoder,
    "mlprogram.nn.action_sequence.Predictor":
        mlprogram.nn.action_sequence.Predictor,
    "mlprogram.nn.action_sequence.Loss": mlprogram.nn.action_sequence.Loss,
    "mlprogram.nn.action_sequence.Accuracy":
        mlprogram.nn.action_sequence.Accuracy,
    "mlprogram.nn.nl2code.NLReader": mlprogram.nn.nl2code.NLReader,
    "mlprogram.nn.nl2code.ActionSequenceReader":
        mlprogram.nn.nl2code.ActionSequenceReader,
    "mlprogram.nn.nl2code.Decoder": mlprogram.nn.nl2code.Decoder,
    "mlprogram.nn.nl2code.Predictor": mlprogram.nn.nl2code.Predictor,
    "mlprogram.nn.treegen.NLReader": mlprogram.nn.treegen.NLReader,
    "mlprogram.nn.treegen.ActionSequenceReader":
        mlprogram.nn.treegen.ActionSequenceReader,
    "mlprogram.nn.treegen.Decoder": mlprogram.nn.treegen.Decoder,
    "mlprogram.nn.pbe_with_repl.Encoder": mlprogram.nn.pbe_with_repl.Encoder,

    "mlprogram.languages.csg.Parser": mlprogram.languages.csg.Parser,
    "mlprogram.languages.csg.Dataset": mlprogram.languages.csg.Dataset,
    "mlprogram.languages.csg.Interpreter": mlprogram.languages.csg.Interpreter,
    "mlprogram.languages.csg.Expander": mlprogram.languages.csg.Expander,
    "mlprogram.languages.csg.IsSubtype": mlprogram.languages.csg.IsSubtype,
    "mlprogram.languages.csg.get_samples": mlprogram.languages.csg.get_samples,
    "mlprogram.languages.csg.transform.TransformCanvas":
        mlprogram.languages.csg.transform.TransformCanvas,
    "mlprogram.languages.csg.transform.AddTestCases":
        mlprogram.languages.csg.transform.AddTestCases,

    "mlprogram.languages.c.Analyzer": mlprogram.languages.c.Analyzer,
    "mlprogram.languages.c.Lexer": mlprogram.languages.c.Lexer
}

types.update(torch_types)
types.update(torchnlp_types)
types.update(fairseq_types)
types.update(numpy_types)
