import gin

import nl2prog.gin
import nl2prog.gin.workspace
import nl2prog.gin.optimizer
import nl2prog.gin.nl2prog
import nl2prog.gin.nl2code
import nl2prog.gin.treegen
import nl2prog.dataset.django
import nl2prog.dataset.hearthstone
import nl2prog.dataset.nl2bash
import nl2prog.nn
import nl2prog.metrics
import nl2prog.metrics.python
import nl2prog.metrics.accuracy
import nl2prog.metrics.python.bleu
import nl2prog.language.python
import nl2prog.language.bash
import nl2prog.ast.action
import nl2prog.utils
import nl2prog.utils.data
import nl2prog.utils.transform
import nl2prog.utils.python

import nl2prog.nn.nl2code
import nl2prog.utils.data.nl2code
import nl2prog.utils.transform.nl2code

import nl2prog.nn.treegen
import nl2prog.utils.data.treegen
import nl2prog.utils.transform.treegen

import nl2prog.gin.torch.external_configurables  # noqa
try:
    import nl2prog.gin.fairseq.external_configurables  # noqa
except:  # noqa
    pass

gin.external_configurable(nl2prog.gin.get_key,
                          module="nl2prog.gin")

gin.external_configurable(nl2prog.gin.workspace.get,
                          module="nl2prog.gin.workspace")
gin.external_configurable(nl2prog.gin.workspace.put,
                          module="nl2prog.gin.workspace")
gin.external_configurable(nl2prog.gin.optimizer.create_optimizer,
                          module="nl2prog.gin.optimizer")
gin.external_configurable(nl2prog.gin.nl2prog.train,
                          module="nl2prog.gin.nl2prog")
gin.external_configurable(nl2prog.gin.nl2prog.evaluate,
                          module="nl2prog.gin.nl2prog")
gin.external_configurable(nl2prog.gin.nl2code.prepare_encoder,
                          module="nl2prog.gin.nl2code")
gin.external_configurable(nl2prog.gin.treegen.prepare_encoder,
                          module="nl2prog.gin.treegen")

gin.external_configurable(nl2prog.dataset.django.download,
                          module="nl2prog.dataset.django")
gin.external_configurable(nl2prog.dataset.django.parse,
                          module="nl2prog.dataset.django")
gin.external_configurable(nl2prog.dataset.django.tokenize_query,
                          module="nl2prog.dataset.django")
gin.external_configurable(nl2prog.dataset.hearthstone.download,
                          module="nl2prog.dataset.hearthstone")
gin.external_configurable(nl2prog.dataset.nl2bash.download,
                          module="nl2prog.dataset.nl2bash")
gin.external_configurable(nl2prog.dataset.nl2bash.tokenize_query,
                          module="nl2prog.dataset.nl2bash")
gin.external_configurable(nl2prog.dataset.nl2bash.tokenize_token,
                          module="nl2prog.dataset.nl2bash")

gin.external_configurable(nl2prog.nn.Loss, module="nl2prog.nn")
gin.external_configurable(nl2prog.nn.Accuracy, module="nl2prog.nn")
gin.external_configurable(nl2prog.nn.TrainModel, module="nl2prog.nn")
gin.external_configurable(nl2prog.metrics.Accuracy, module="nl2prog.metrics")
gin.external_configurable(nl2prog.metrics.Bleu, module="nl2prog.metrics")
gin.external_configurable(nl2prog.metrics.python.bleu.Bleu,
                          module="nl2prog.metrics.python")
gin.external_configurable(nl2prog.language.python.parse,
                          module="nl2prog.language.python")
gin.external_configurable(nl2prog.language.python.unparse,
                          module="nl2prog.language.python")
gin.external_configurable(nl2prog.language.python.is_subtype,
                          module="nl2prog.language.python")
gin.external_configurable(nl2prog.language.bash.parse,
                          module="nl2prog.language.bash")
gin.external_configurable(nl2prog.language.bash.unparse,
                          module="nl2prog.language.bash")
gin.external_configurable(nl2prog.language.bash.is_subtype,
                          module="nl2prog.language.bash")
gin.external_configurable(nl2prog.ast.action.ActionOptions,
                          module="nl2prog.ast.action")
gin.external_configurable(nl2prog.ast.action.code_to_action_sequence,
                          module="nl2prog.ast.action")
gin.external_configurable(nl2prog.utils.CommonBeamSearchSynthesizer,
                          module="nl2prog.utils")
gin.external_configurable(nl2prog.utils.CommonBeamSearchSynthesizer.create,
                          module="nl2prog.utils.CommonBeamSearchSynthesizer")
gin.external_configurable(nl2prog.utils.data.Collate,
                          module="nl2prog.utils.data")
gin.external_configurable(nl2prog.utils.data.CollateGroundTruth,
                          module="nl2prog.utils.data")
gin.external_configurable(nl2prog.utils.data.CollateNlFeature,
                          module="nl2prog.utils.data")
gin.external_configurable(nl2prog.utils.data.collate_none,
                          module="nl2prog.utils.data")
gin.external_configurable(nl2prog.utils.data.split_none,
                          module="nl2prog.utils.data")
gin.external_configurable(nl2prog.utils.transform.TransformDataset,
                          module="nl2prog.utils.transform")
gin.external_configurable(nl2prog.utils.transform.TransformCode,
                          module="nl2prog.utils.transform")
gin.external_configurable(nl2prog.utils.transform.TransformGroundTruth,
                          module="nl2prog.utils.transform")
gin.external_configurable(nl2prog.utils.python.tokenize_token,
                          module="nl2prog.utils.python")
gin.external_configurable(nl2prog.utils.python.tokenize_query,
                          module="nl2prog.utils.python")

gin.external_configurable(nl2prog.nn.nl2code.TrainModel,
                          module="nl2prog.nn.nl2code")
gin.external_configurable(nl2prog.nn.nl2code.NLReader,
                          module="nl2prog.nn.nl2code")
gin.external_configurable(nl2prog.nn.nl2code.ActionSequenceReader,
                          module="nl2prog.nn.nl2code")
gin.external_configurable(nl2prog.nn.nl2code.Decoder,
                          module="nl2prog.nn.nl2code")
gin.external_configurable(nl2prog.nn.nl2code.Predictor,
                          module="nl2prog.nn.nl2code")
gin.external_configurable(nl2prog.utils.data.nl2code.CollateInput,
                          module="nl2prog.utils.data.nl2code")
gin.external_configurable(nl2prog.utils.data.nl2code.CollateActionSequence,
                          module="nl2prog.utils.data.nl2code")
gin.external_configurable(nl2prog.utils.data.nl2code.CollateState,
                          module="nl2prog.utils.data.nl2code")
gin.external_configurable(nl2prog.utils.data.nl2code.split_states,
                          module="nl2prog.utils.data.nl2code")
gin.external_configurable(nl2prog.utils.transform.nl2code.TransformQuery,
                          module="nl2prog.utils.transform.nl2code")
gin.external_configurable(nl2prog.utils.transform.nl2code.TransformEvaluator,
                          module="nl2prog.utils.transform.nl2code")

gin.external_configurable(nl2prog.nn.treegen.TrainModel,
                          module="nl2prog.nn.treegen")
gin.external_configurable(nl2prog.nn.treegen.NLReader,
                          module="nl2prog.nn.treegen")
gin.external_configurable(nl2prog.nn.treegen.ActionSequenceReader,
                          module="nl2prog.nn.treegen")
gin.external_configurable(nl2prog.nn.treegen.Decoder,
                          module="nl2prog.nn.treegen")
gin.external_configurable(nl2prog.nn.treegen.Predictor,
                          module="nl2prog.nn.treegen")
gin.external_configurable(nl2prog.utils.data.treegen.CollateInput,
                          module="nl2prog.utils.data.treegen")
gin.external_configurable(nl2prog.utils.data.treegen.CollateActionSequence,
                          module="nl2prog.utils.data.treegen")
gin.external_configurable(nl2prog.utils.data.treegen.CollateQuery,
                          module="nl2prog.utils.data.treegen")
gin.external_configurable(nl2prog.utils.transform.treegen.TransformQuery,
                          module="nl2prog.utils.transform.treegen")
gin.external_configurable(nl2prog.utils.transform.treegen.TransformEvaluator,
                          module="nl2prog.utils.transform.treegen")

# Constants
gin.constant("nl2prog.language.python.ParseMode.Eval",
             nl2prog.language.python.ParseMode.Eval)
gin.constant("nl2prog.language.python.ParseMode.Exec",
             nl2prog.language.python.ParseMode.Exec)
gin.constant("nl2prog.language.python.ParseMode.Single",
             nl2prog.language.python.ParseMode.Single)
