import os
import collections

import mlprogram.entrypoint
import mlprogram.datasets.django
import mlprogram.datasets.hearthstone
import mlprogram.datasets.nl2bash
import mlprogram.datasets.deepfix
import mlprogram.nn
import mlprogram.nn.action_sequence
import mlprogram.metrics
import mlprogram.languages.python
import mlprogram.languages.python.metrics
import mlprogram.languages.bash
import mlprogram.languages.csg
import mlprogram.languages.c
import mlprogram.actions
import mlprogram.synthesizers
import mlprogram.utils
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

import mlprogram.utils.transform.csg

from mlprogram.entrypoint.numpy import types as numpy_types
from mlprogram.entrypoint.torch import types as torch_types
from mlprogram.entrypoint.torchnlp import types as torchnlp_types
try:
    from mlprogram.entrypoint.fairseq import types as fairseq_types
except:  # noqa
    fairseq_types = {}
    pass

types = {
    "select": lambda key, options: options[key],

    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": lambda x, y: x / y,
    "intdiv": lambda x, y: x // y,
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
    "mlprogram.datasets.django.TokenizeToken":
        mlprogram.datasets.django.TokenizeToken,
    "mlprogram.datasets.django.TokenizeQuery":
        mlprogram.datasets.django.TokenizeQuery,
    "mlprogram.datasets.hearthstone.download":
        mlprogram.datasets.hearthstone.download,
    "mlprogram.datasets.hearthstone.TokenizeQuery":
        mlprogram.datasets.hearthstone.TokenizeQuery,
    "mlprogram.datasets.hearthstone.TokenizeToken":
        mlprogram.datasets.hearthstone.TokenizeToken,
    "mlprogram.datasets.nl2bash.load": mlprogram.datasets.nl2bash.load,
    "mlprogram.datasets.nl2bash.TokenizeQuery":
        mlprogram.datasets.nl2bash.TokenizeQuery,
    "mlprogram.datasets.nl2bash.TokenizeToken":
        mlprogram.datasets.nl2bash.TokenizeToken,
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

    "mlprogram.utils.Compose": mlprogram.utils.Compose,
    "mlprogram.utils.Map": mlprogram.utils.Map,
    "mlprogram.utils.Flatten": mlprogram.utils.Flatten,
    "mlprogram.utils.Sequence": mlprogram.utils.Sequence,
    "mlprogram.utils.Threshold": mlprogram.utils.Threshold,
    "mlprogram.utils.Pick": mlprogram.utils.Pick,
    "mlprogram.utils.save": mlprogram.utils.save,
    "mlprogram.utils.load": mlprogram.utils.load,

    "mlprogram.synthesizers.BeamSearch": mlprogram.synthesizers.BeamSearch,
    "mlprogram.synthesizers.SMC": mlprogram.synthesizers.SMC,
    "mlprogram.synthesizers.FilteredSynthesizer":
        mlprogram.synthesizers.FilteredSynthesizer,
    "mlprogram.synthesizers.SynthesizerWithTimeout":
        mlprogram.synthesizers.SynthesizerWithTimeout,
    "mlprogram.samplers.transform": mlprogram.samplers.transform,
    "mlprogram.samplers.ActionSequenceSampler":
        mlprogram.samplers.ActionSequenceSampler,
    "mlprogram.samplers.AstReferenceSampler":
        mlprogram.samplers.AstReferenceSampler,
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

    "mlprogram.actions.AstToActionSequence":
        mlprogram.actions.AstToActionSequence,
    "mlprogram.utils.transform.RandomChoice":
        mlprogram.utils.transform.RandomChoice,
    "mlprogram.utils.transform.EvaluateGroundTruth":
        mlprogram.utils.transform.EvaluateGroundTruth,
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
    "mlprogram.utils.transform.pbe_with_repl.EvaluateCode":
        mlprogram.utils.transform.pbe_with_repl.EvaluateCode,


    "mlprogram.encoders.ActionSequenceEncoder":
        mlprogram.encoders.ActionSequenceEncoder,

    "mlprogram.nn.Apply": mlprogram.nn.Apply,
    "mlprogram.nn.AggregatedLoss": mlprogram.nn.AggregatedLoss,
    "mlprogram.nn.Pick": mlprogram.nn.Pick,
    "mlprogram.nn.Add": mlprogram.nn.Add,
    "mlprogram.nn.Sub": mlprogram.nn.Sub,
    "mlprogram.nn.Mul": mlprogram.nn.Mul,
    "mlprogram.nn.Div": mlprogram.nn.Div,
    "mlprogram.nn.IntDiv": mlprogram.nn.IntDiv,
    "mlprogram.nn.Neg": mlprogram.nn.Neg,
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

    "mlprogram.languages.csg.ToAst": mlprogram.languages.csg.ToAst,
    "mlprogram.languages.csg.ToCsgAst": mlprogram.languages.csg.ToCsgAst,
    "mlprogram.languages.csg.Dataset": mlprogram.languages.csg.Dataset,
    "mlprogram.languages.csg.Interpreter": mlprogram.languages.csg.Interpreter,
    "mlprogram.languages.csg.GetTokenType":
        mlprogram.languages.csg.GetTokenType,
    "mlprogram.languages.csg.IsSubtype": mlprogram.languages.csg.IsSubtype,
    "mlprogram.languages.csg.get_samples": mlprogram.languages.csg.get_samples,
    "mlprogram.utils.transform.csg.TransformCanvas":
        mlprogram.utils.transform.csg.TransformCanvas,

    "mlprogram.languages.c.Analyzer": mlprogram.languages.c.Analyzer,
    "mlprogram.languages.c.Tokenizer": mlprogram.languages.c.Tokenizer
}

types.update(torch_types)
types.update(torchnlp_types)
types.update(fairseq_types)
types.update(numpy_types)
