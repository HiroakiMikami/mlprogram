import gin

import mlprogram.gin
import mlprogram.gin.workspace
import mlprogram.gin.optimizer
import mlprogram.gin.nl2prog
import mlprogram.gin.nl2code
import mlprogram.gin.treegen
import mlprogram.datasets.django
import mlprogram.datasets.hearthstone
import mlprogram.datasets.nl2bash
import mlprogram.nn
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
import mlprogram.utils.python

import mlprogram.nn.nl2code
import mlprogram.utils.transform.nl2code

import mlprogram.nn.treegen
import mlprogram.utils.transform.treegen

import mlprogram.gin.torch.external_configurables  # noqa
try:
    import mlprogram.gin.fairseq.external_configurables  # noqa
except:  # noqa
    pass

gin.external_configurable(mlprogram.gin.get_key,
                          module="mlprogram.gin")

gin.external_configurable(mlprogram.gin.workspace.get,
                          module="mlprogram.gin.workspace")
gin.external_configurable(mlprogram.gin.workspace.put,
                          module="mlprogram.gin.workspace")
gin.external_configurable(mlprogram.gin.optimizer.create_optimizer,
                          module="mlprogram.gin.optimizer")
gin.external_configurable(mlprogram.gin.nl2prog.train,
                          module="mlprogram.gin.nl2prog")
gin.external_configurable(mlprogram.gin.nl2prog.evaluate,
                          module="mlprogram.gin.nl2prog")
gin.external_configurable(mlprogram.gin.nl2code.prepare_encoder,
                          module="mlprogram.gin.nl2code")
gin.external_configurable(mlprogram.gin.treegen.prepare_encoder,
                          module="mlprogram.gin.treegen")

gin.external_configurable(mlprogram.datasets.django.download,
                          module="mlprogram.datasets.django")
gin.external_configurable(mlprogram.datasets.django.parse,
                          module="mlprogram.datasets.django")
gin.external_configurable(mlprogram.datasets.django.tokenize_query,
                          module="mlprogram.datasets.django")
gin.external_configurable(mlprogram.datasets.hearthstone.download,
                          module="mlprogram.datasets.hearthstone")
gin.external_configurable(mlprogram.datasets.hearthstone.tokenize_query,
                          module="mlprogram.datasets.hearthstone")
gin.external_configurable(mlprogram.datasets.nl2bash.download,
                          module="mlprogram.datasets.nl2bash")
gin.external_configurable(mlprogram.datasets.nl2bash.tokenize_query,
                          module="mlprogram.datasets.nl2bash")
gin.external_configurable(mlprogram.datasets.nl2bash.tokenize_token,
                          module="mlprogram.datasets.nl2bash")

gin.external_configurable(mlprogram.nn.NL2ProgLoss, module="mlprogram.nn")
gin.external_configurable(mlprogram.nn.NL2ProgAccuracy, module="mlprogram.nn")
gin.external_configurable(mlprogram.nn.TrainModel, module="mlprogram.nn")
gin.external_configurable(mlprogram.metrics.Accuracy,
                          module="mlprogram.metrics")
gin.external_configurable(mlprogram.metrics.Bleu, module="mlprogram.metrics")
gin.external_configurable(mlprogram.metrics.python.bleu.Bleu,
                          module="mlprogram.metrics.python")
gin.external_configurable(mlprogram.languages.python.parse,
                          module="mlprogram.languages.python")
gin.external_configurable(mlprogram.languages.python.unparse,
                          module="mlprogram.languages.python")
gin.external_configurable(mlprogram.languages.python.is_subtype,
                          module="mlprogram.languages.python")
gin.external_configurable(mlprogram.languages.bash.parse,
                          module="mlprogram.languages.bash")
gin.external_configurable(mlprogram.languages.bash.unparse,
                          module="mlprogram.languages.bash")
gin.external_configurable(mlprogram.languages.bash.is_subtype,
                          module="mlprogram.languages.bash")
gin.external_configurable(mlprogram.actions.ActionOptions,
                          module="mlprogram.actions")
gin.external_configurable(mlprogram.utils.Compose, module="mlprogram.utils")
gin.external_configurable(mlprogram.utils.Sequence, module="mlprogram.utils")

gin.external_configurable(mlprogram.synthesizers.BeamSearch,
                          module="mlprogram.synthesizers")
gin.external_configurable(mlprogram.synthesizers.SMC,
                          module="mlprogram.synthesizers")
gin.external_configurable(mlprogram.samplers.ActionSequenceSampler,
                          module="mlprogram.samplers")

gin.external_configurable(mlprogram.utils.data.Collate,
                          module="mlprogram.utils.data")
gin.external_configurable(mlprogram.utils.data.CollateOptions,
                          module="mlprogram.utils.data")
gin.external_configurable(mlprogram.utils.transform.AstToSingleActionSequence,
                          module="mlprogram.utils.transform")
gin.external_configurable(mlprogram.utils.transform.RandomChoice,
                          module="mlprogram.utils.transform")
gin.external_configurable(mlprogram.utils.transform.TransformCode,
                          module="mlprogram.utils.transform")
gin.external_configurable(mlprogram.utils.transform.TransformGroundTruth,
                          module="mlprogram.utils.transform")
gin.external_configurable(mlprogram.utils.python.tokenize_token,
                          module="mlprogram.utils.python")

gin.external_configurable(mlprogram.nn.nl2code.TrainModel,
                          module="mlprogram.nn.nl2code")
gin.external_configurable(mlprogram.nn.nl2code.NLReader,
                          module="mlprogram.nn.nl2code")
gin.external_configurable(mlprogram.nn.nl2code.ActionSequenceReader,
                          module="mlprogram.nn.nl2code")
gin.external_configurable(mlprogram.nn.nl2code.Decoder,
                          module="mlprogram.nn.nl2code")
gin.external_configurable(mlprogram.nn.nl2code.Predictor,
                          module="mlprogram.nn.nl2code")
gin.external_configurable(mlprogram.utils.transform.nl2code.TransformQuery,
                          module="mlprogram.utils.transform.nl2code")
gin.external_configurable(
    mlprogram.utils.transform.nl2code.TransformActionSequence,
    module="mlprogram.utils.transform.nl2code")

gin.external_configurable(mlprogram.nn.treegen.TrainModel,
                          module="mlprogram.nn.treegen")
gin.external_configurable(mlprogram.nn.treegen.NLReader,
                          module="mlprogram.nn.treegen")
gin.external_configurable(mlprogram.nn.treegen.ActionSequenceReader,
                          module="mlprogram.nn.treegen")
gin.external_configurable(mlprogram.nn.treegen.Decoder,
                          module="mlprogram.nn.treegen")
gin.external_configurable(mlprogram.nn.treegen.Predictor,
                          module="mlprogram.nn.treegen")
gin.external_configurable(mlprogram.utils.transform.treegen.TransformQuery,
                          module="mlprogram.utils.transform.treegen")
gin.external_configurable(
    mlprogram.utils.transform.treegen.TransformActionSequence,
    module="mlprogram.utils.transform.treegen")

# Constants
gin.constant("mlprogram.languages.python.ParseMode.Eval",
             mlprogram.languages.python.ParseMode.Eval)
gin.constant("mlprogram.languages.python.ParseMode.Exec",
             mlprogram.languages.python.ParseMode.Exec)
gin.constant("mlprogram.languages.python.ParseMode.Single",
             mlprogram.languages.python.ParseMode.Single)
