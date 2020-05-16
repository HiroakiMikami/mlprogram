import gin

import mlprogram.gin
import mlprogram.gin.workspace
import mlprogram.gin.optimizer
import mlprogram.gin.nl2prog
import mlprogram.gin.nl2code
import mlprogram.gin.treegen
import mlprogram.dataset.django
import mlprogram.dataset.hearthstone
import mlprogram.dataset.nl2bash
import mlprogram.nn
import mlprogram.metrics
import mlprogram.metrics.python
import mlprogram.metrics.accuracy
import mlprogram.metrics.python.bleu
import mlprogram.language.python
import mlprogram.language.bash
import mlprogram.action.action
import mlprogram.utils
import mlprogram.utils.data
import mlprogram.utils.transform
import mlprogram.utils.python

import mlprogram.nn.nl2code
import mlprogram.utils.data.nl2code
import mlprogram.utils.transform.nl2code

import mlprogram.nn.treegen
import mlprogram.utils.data.treegen
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

gin.external_configurable(mlprogram.dataset.django.download,
                          module="mlprogram.dataset.django")
gin.external_configurable(mlprogram.dataset.django.parse,
                          module="mlprogram.dataset.django")
gin.external_configurable(mlprogram.dataset.django.tokenize_query,
                          module="mlprogram.dataset.django")
gin.external_configurable(mlprogram.dataset.hearthstone.download,
                          module="mlprogram.dataset.hearthstone")
gin.external_configurable(mlprogram.dataset.nl2bash.download,
                          module="mlprogram.dataset.nl2bash")
gin.external_configurable(mlprogram.dataset.nl2bash.tokenize_query,
                          module="mlprogram.dataset.nl2bash")
gin.external_configurable(mlprogram.dataset.nl2bash.tokenize_token,
                          module="mlprogram.dataset.nl2bash")

gin.external_configurable(mlprogram.nn.Loss, module="mlprogram.nn")
gin.external_configurable(mlprogram.nn.Accuracy, module="mlprogram.nn")
gin.external_configurable(mlprogram.nn.TrainModel, module="mlprogram.nn")
gin.external_configurable(mlprogram.metrics.Accuracy,
                          module="mlprogram.metrics")
gin.external_configurable(mlprogram.metrics.Bleu, module="mlprogram.metrics")
gin.external_configurable(mlprogram.metrics.python.bleu.Bleu,
                          module="mlprogram.metrics.python")
gin.external_configurable(mlprogram.language.python.parse,
                          module="mlprogram.language.python")
gin.external_configurable(mlprogram.language.python.unparse,
                          module="mlprogram.language.python")
gin.external_configurable(mlprogram.language.python.is_subtype,
                          module="mlprogram.language.python")
gin.external_configurable(mlprogram.language.bash.parse,
                          module="mlprogram.language.bash")
gin.external_configurable(mlprogram.language.bash.unparse,
                          module="mlprogram.language.bash")
gin.external_configurable(mlprogram.language.bash.is_subtype,
                          module="mlprogram.language.bash")
gin.external_configurable(mlprogram.action.action.ActionOptions,
                          module="mlprogram.action.action")
gin.external_configurable(mlprogram.action.action.code_to_action_sequence,
                          module="mlprogram.action.action")
gin.external_configurable(mlprogram.synthesizer.CommonBeamSearchSynthesizer,
                          module="mlprogram.utils")
gin.external_configurable(
    mlprogram.synthesizer.CommonBeamSearchSynthesizer.create,
    module="mlprogram.synthesizer.CommonBeamSearchSynthesizer")
gin.external_configurable(mlprogram.utils.data.Collate,
                          module="mlprogram.utils.data")
gin.external_configurable(mlprogram.utils.data.CollateGroundTruth,
                          module="mlprogram.utils.data")
gin.external_configurable(mlprogram.utils.data.CollateNlFeature,
                          module="mlprogram.utils.data")
gin.external_configurable(mlprogram.utils.data.collate_none,
                          module="mlprogram.utils.data")
gin.external_configurable(mlprogram.utils.data.split_none,
                          module="mlprogram.utils.data")
gin.external_configurable(mlprogram.utils.transform.TransformDataset,
                          module="mlprogram.utils.transform")
gin.external_configurable(mlprogram.utils.transform.TransformCode,
                          module="mlprogram.utils.transform")
gin.external_configurable(mlprogram.utils.transform.TransformGroundTruth,
                          module="mlprogram.utils.transform")
gin.external_configurable(mlprogram.utils.python.tokenize_token,
                          module="mlprogram.utils.python")
gin.external_configurable(mlprogram.utils.python.tokenize_query,
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
gin.external_configurable(mlprogram.utils.data.nl2code.CollateInput,
                          module="mlprogram.utils.data.nl2code")
gin.external_configurable(mlprogram.utils.data.nl2code.CollateActionSequence,
                          module="mlprogram.utils.data.nl2code")
gin.external_configurable(mlprogram.utils.data.nl2code.CollateState,
                          module="mlprogram.utils.data.nl2code")
gin.external_configurable(mlprogram.utils.data.nl2code.split_states,
                          module="mlprogram.utils.data.nl2code")
gin.external_configurable(mlprogram.utils.transform.nl2code.TransformQuery,
                          module="mlprogram.utils.transform.nl2code")
gin.external_configurable(mlprogram.utils.transform.nl2code.TransformEvaluator,
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
gin.external_configurable(mlprogram.utils.data.treegen.CollateInput,
                          module="mlprogram.utils.data.treegen")
gin.external_configurable(mlprogram.utils.data.treegen.CollateActionSequence,
                          module="mlprogram.utils.data.treegen")
gin.external_configurable(mlprogram.utils.data.treegen.CollateQuery,
                          module="mlprogram.utils.data.treegen")
gin.external_configurable(mlprogram.utils.transform.treegen.TransformQuery,
                          module="mlprogram.utils.transform.treegen")
gin.external_configurable(mlprogram.utils.transform.treegen.TransformEvaluator,
                          module="mlprogram.utils.transform.treegen")

# Constants
gin.constant("mlprogram.language.python.ParseMode.Eval",
             mlprogram.language.python.ParseMode.Eval)
gin.constant("mlprogram.language.python.ParseMode.Exec",
             mlprogram.language.python.ParseMode.Exec)
gin.constant("mlprogram.language.python.ParseMode.Single",
             mlprogram.language.python.ParseMode.Single)
