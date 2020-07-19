import os
import collections

import mlprogram.entrypoint.nl2prog
import mlprogram.datasets.django
import mlprogram.datasets.hearthstone
import mlprogram.datasets.nl2bash
import mlprogram.nn
import mlprogram.nn.action_sequence
import mlprogram.metrics
import mlprogram.metrics.python
import mlprogram.metrics.accuracy
import mlprogram.metrics.python.bleu
import mlprogram.languages.python
import mlprogram.languages.bash
import mlprogram.actions
import mlprogram.synthesizers
import mlprogram.utils
import mlprogram.utils.data
import mlprogram.utils.transform
import mlprogram.encoders

import mlprogram.nn.nl2code
import mlprogram.utils.transform.nl2code

import mlprogram.nn.treegen
import mlprogram.utils.transform.treegen

from mlprogram.entrypoint.torch import types as torch_types
from mlprogram.entrypoint.torchnlp import types as torchnlp_types
try:
    from mlprogram.entrypoint.fairseq import types as fairseq_types
except:  # noqa
    fairseq_types = {}
    pass

types = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": lambda x, y: x / y,

    "collections.OrderedDict": lambda items: collections.OrderedDict(items),
    "os.path.join": lambda args: os.path.join(*args),

    "mlprogram.entrypoint.nl2prog.train": mlprogram.entrypoint.nl2prog.train,
    "mlprogram.entrypoint.nl2prog.evaluate":
        mlprogram.entrypoint.nl2prog.evaluate,

    "mlprogram.datasets.django.download": mlprogram.datasets.django.download,
    "mlprogram.datasets.django.Parse": mlprogram.datasets.django.Parse,
    "mlprogram.datasets.django.TokenizeQuery":
        mlprogram.datasets.django.TokenizeQuery,
    "mlprogram.datasets.hearthstone.download":
        mlprogram.datasets.hearthstone.download,
    "mlprogram.datasets.hearthstone.TokenizeQuery":
        mlprogram.datasets.hearthstone.TokenizeQuery,
    "mlprogram.datasets.nl2bash.download": mlprogram.datasets.nl2bash.download,
    "mlprogram.datasets.nl2bash.TokenizeQuery":
        mlprogram.datasets.nl2bash.TokenizeQuery,
    "mlprogram.datasets.nl2bash.TokenizeToken":
        mlprogram.datasets.nl2bash.TokenizeToken,

    "mlprogram.metrics.Accuracy": mlprogram.metrics.Accuracy,
    "mlprogram.metrics.Bleu": mlprogram.metrics.Bleu,
    "mlprogram.metrics.python.Bleu": mlprogram.metrics.python.Bleu,

    "mlprogram.languages.python.Parse": mlprogram.languages.python.Parse,
    "mlprogram.languages.python.Unparse": mlprogram.languages.python.Unparse,
    "mlprogram.languages.python.IsSubtype":
        mlprogram.languages.python.IsSubtype,
    "mlprogram.languages.python.TokenizeToken":
        mlprogram.languages.python.TokenizeToken,
    "mlprogram.languages.python.ParseMode.Eval":
        lambda: mlprogram.languages.python.ParseMode.Eval,
    "mlprogram.languages.python.ParseMode.Exec":
        lambda: mlprogram.languages.python.ParseMode.Exec,
    "mlprogram.languages.python.ParseMode.Single":
        lambda: mlprogram.languages.python.ParseMode.Single,
    "mlprogram.languages.bash.Parse": mlprogram.languages.bash.Parse,
    "mlprogram.languages.bash.Unparse": mlprogram.languages.bash.Unparse,
    "mlprogram.languages.bash.IsSubtype": mlprogram.languages.bash.IsSubtype,

    "mlprogram.utils.Compose": mlprogram.utils.Compose,
    "mlprogram.utils.Map": mlprogram.utils.Map,
    "mlprogram.utils.Sequence": mlprogram.utils.Sequence,
    "mlprogram.utils.save": mlprogram.utils.save,
    "mlprogram.utils.load": mlprogram.utils.load,

    "mlprogram.synthesizers.BeamSearch": mlprogram.synthesizers.BeamSearch,
    "mlprogram.synthesizers.SMC": mlprogram.synthesizers.SMC,
    "mlprogram.samplers.ActionSequenceSampler":
        mlprogram.samplers.ActionSequenceSampler,

    "mlprogram.utils.data.Collate": mlprogram.utils.data.Collate,
    "mlprogram.utils.data.CollateOptions": mlprogram.utils.data.CollateOptions,
    "mlprogram.utils.data.get_words": mlprogram.utils.data.get_words,
    "mlprogram.utils.data.get_characters": mlprogram.utils.data.get_characters,
    "mlprogram.utils.data.get_samples": mlprogram.utils.data.get_samples,

    "mlprogram.utils.transform.AstToSingleActionSequence":
        mlprogram.utils.transform.AstToSingleActionSequence,
    "mlprogram.utils.transform.RandomChoice":
        mlprogram.utils.transform.RandomChoice,
    "mlprogram.utils.transform.TransformCode":
        mlprogram.utils.transform.TransformCode,
    "mlprogram.utils.transform.TransformGroundTruth":
        mlprogram.utils.transform.TransformGroundTruth,
    "mlprogram.utils.transform.nl2code.TransformQuery":
        mlprogram.utils.transform.nl2code.TransformQuery,
    "mlprogram.utils.transform.nl2code.TransformActionSequence":
        mlprogram.utils.transform.nl2code.TransformActionSequence,
    "mlprogram.utils.transform.treegen.TransformQuery":
        mlprogram.utils.transform.treegen.TransformQuery,
    "mlprogram.utils.transform.treegen.TransformActionSequence":
        mlprogram.utils.transform.treegen.TransformActionSequence,

    "mlprogram.encoders.ActionSequenceEncoder":
        mlprogram.encoders.ActionSequenceEncoder,

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
}

types.update(torch_types)
types.update(torchnlp_types)
types.update(fairseq_types)
