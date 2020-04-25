import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Callable, Optional, Tuple, TypeVar, Dict, Generic
from dataclasses import dataclass
from nl2prog.language.action import ActionOptions
from nl2prog.language.evaluator import Evaluator
from nl2prog.encoders import ActionSequenceEncoder
from nl2prog.utils \
    import BeamSearchSynthesizer as BaseBeamSearchSynthesizer, \
    IsSubtype, LazyLogProbability
from nl2prog.utils.beam_search_synthesizer import Hypothesis
from nl2prog.nn import TrainModel


logger = logging.getLogger(__name__)


Input = TypeVar("Input")
EncodedInput = TypeVar("EncodedInput")
InputReaderInput = TypeVar("InputReaderInput")
NlFeature = TypeVar("NlFeature")
NlFeatureInput = TypeVar("NlFeatureInput")
OtherFeature = TypeVar("OtherFeature")
OtherFeatureInput = TypeVar("OtherFeatureInput")
AdditionalState = TypeVar("AdditionalState")
AdditionalStateInput = TypeVar("AdditionalStateInput")
EncodedActionSequence = TypeVar("EncodedActionSequence")
EncodedQuery = TypeVar("EncodedQuery")
QueryInput = TypeVar("QueryInput")
ActionSequenceReaderInput = TypeVar("ActionSequenceReaderInput")


@dataclass
class State(Generic[NlFeature, OtherFeature, AdditionalState]):
    query: List[str]
    nl_feature: NlFeature
    other_feature: OtherFeature
    state: Optional[AdditionalState]


class CommonBeamSearchSynthesizer(BaseBeamSearchSynthesizer):
    def __init__(self, beam_size: int,
                 transform_input: Callable[[Input],
                                           Tuple[List[str], EncodedInput]],
                 transform_evaluator: Callable[
                     [Evaluator, List[str]],
                     Optional[Tuple[EncodedActionSequence, EncodedQuery]]],
                 collate_input: Callable[[List[EncodedInput]],
                                         InputReaderInput],
                 collate_action_sequence: Callable[
                     [List[EncodedActionSequence]], ActionSequenceReaderInput],
                 collate_query: Callable[[List[EncodedQuery]], QueryInput],
                 collate_state: Callable[[List[AdditionalState]],
                                         AdditionalStateInput],
                 collate_nl_feature: Callable[[List[NlFeature]],
                                              NlFeatureInput],
                 collate_other_feature: Callable[[List[OtherFeature]],
                                                 OtherFeatureInput],
                 split_states: Callable[[AdditionalStateInput],
                                        List[AdditionalState]],
                 input_reader: nn.Module,
                 action_sequence_reader: nn.Module,
                 decoder: nn.Module,
                 predictor: nn.Module,
                 action_sequence_encoder: ActionSequenceEncoder,
                 is_subtype: IsSubtype,
                 options: ActionOptions = ActionOptions(True, True),
                 eps: float = 1e-8,
                 max_steps: Optional[int] = None,
                 device: torch.device = torch.device("cpu")):
        """
        Parameters
        ----------
        beam_size: int
            The number of candidates
        input_reader:
            The encoder module
        action_sequence_reader:
        decoder:
        predictor: Predictor
            The module to predict the probabilities of actions
        action_seqeunce_encoder: ActionSequenceEncoder
        is_subtype: IsSubType
            The function to check the type relations between 2 node types.
            This returns true if the argument 0 is subtype of the argument 1.
        options: ActionOptions
        eps: float
        max_steps: Optional[int]
        """
        super(CommonBeamSearchSynthesizer, self).__init__(
            beam_size, is_subtype, options, max_steps)
        self.transform_input = transform_input
        self.transform_evaluator = transform_evaluator
        self.collate_input = collate_input
        self.collate_action_sequence = collate_action_sequence
        self.collate_query = collate_query
        self.collate_state = collate_state
        self.collate_nl_feature = collate_nl_feature
        self.collate_other_feature = collate_other_feature
        self.split_states = split_states
        self.input_reader = input_reader.to(device)
        self.action_sequence_reader = action_sequence_reader.to(device)
        self.decoder = decoder.to(device)
        self.predictor = predictor.to(device)
        self.action_sequence_encoder = action_sequence_encoder
        self.eps = eps

    @staticmethod
    def create(beam_size: int,
               transform_input: Callable[[Input],
                                         Tuple[List[str], EncodedInput]],
               transform_evaluator: Callable[
                   [Evaluator, List[str]],
                   Optional[Tuple[EncodedActionSequence, EncodedQuery]]],
               collate_input: Callable[[List[EncodedInput]],
                                       InputReaderInput],
               collate_action_sequence: Callable[
                   [List[EncodedActionSequence]], ActionSequenceReaderInput],
               collate_query: Callable[[List[EncodedQuery]], QueryInput],
               collate_state: Callable[[List[AdditionalState]],
                                       AdditionalStateInput],
               collate_nl_feature: Callable[[List[NlFeature]],
                                            NlFeatureInput],
               collate_other_feature: Callable[[List[OtherFeature]],
                                               OtherFeatureInput],
               split_states: Callable[[AdditionalStateInput],
                                      List[AdditionalState]],
               model: TrainModel,
               action_sequence_encoder: ActionSequenceEncoder,
               is_subtype: IsSubtype,
               options: ActionOptions = ActionOptions(True, True),
               eps: float = 1e-8,
               max_steps: Optional[int] = None,
               device: torch.device = torch.device("cpu")):
        return CommonBeamSearchSynthesizer(
            beam_size, transform_input, transform_evaluator, collate_input,
            collate_action_sequence, collate_query, collate_state,
            collate_nl_feature, collate_other_feature, split_states,
            model.input_reader, model.action_sequence_reader, model.decoder,
            model.predictor, action_sequence_encoder, is_subtype, options, eps,
            max_steps, device)

    def state_dict(self) -> Dict[str, object]:
        return TrainModel(self.input_reader, self.action_sequence_reader,
                          self.decoder, self.predictor).state_dict()

    def load_state_dict(self, state_dict) -> None:
        TrainModel(self.input_reader, self.action_sequence_reader,
                   self.decoder, self.predictor).load_state_dict(state_dict)

    def initialize(self, input: Input) -> State:
        query_for_synth, enc_input = self.transform_input(input)
        cinput = self.collate_input([enc_input])
        nl_feature, other_feature = self.input_reader(cinput)
        nl_feature = nl_feature.data
        L = nl_feature.shape[0]
        nl_feature = nl_feature.view(L, -1)

        return State(query_for_synth, nl_feature, other_feature, None)

    def batch_update(self, hs: List[Hypothesis]) \
            -> List[Tuple[State, LazyLogProbability]]:
        # Create batch of hypothesis
        nl_features = []
        other_features = []
        action_sequences = []
        queries = []
        states = []
        for h in hs:
            opt = self.transform_evaluator(h.evaluator, h.state.query)
            if opt is not None:
                action_sequence, query = opt
                action_sequences.append(action_sequence)
                queries.append(query)
                nl_features.append(h.state.nl_feature)
                other_features.append(h.state.other_feature)
                states.append(h.state.state)
            else:
                logger.info("Invalid evaluater is in hypothesis")
        cnl_features = self.collate_nl_feature(nl_features)
        cother_features = self.collate_other_feature(other_features)
        caction_sequences = self.collate_action_sequence(action_sequences)
        cqueries = self.collate_query(queries)
        cstates = self.collate_state(states)

        with torch.no_grad():
            ast_feature = self.action_sequence_reader(caction_sequences)
            feature, state = self.decoder(cqueries, cnl_features,
                                          cother_features,
                                          ast_feature, cstates)
            results = self.predictor(cnl_features, feature)
        # (len(hs), n_rules)
        rule_pred = results[0].data[-1, :, :].cpu().reshape(len(hs), -1)
        # (len(hs), n_tokens)
        token_pred = results[1].data[-1, :, :].cpu().reshape(len(hs), -1)
        # (len(hs), query_length)
        copy_pred = results[2].data[-1, :, :].cpu().reshape(len(hs), -1)
        states = self.split_states(state)

        retval = []
        for i, (h, s) in enumerate(zip(hs, states)):
            state = State(h.state.query, h.state.nl_feature,
                          h.state.other_feature, s)

            class Functions:
                def __init__(self, i, action_sequence_encoder, eps):
                    self.i = i
                    self.action_sequence_encoder = action_sequence_encoder
                    self.eps = eps

                def get_rule_prob(self):
                    # TODO
                    idx_to_rule = \
                        self.action_sequence_encoder._rule_encoder.vocab
                    retval = {}
                    # 0 is unknown rule
                    for j in range(1, rule_pred.shape[1]):
                        p = rule_pred[self.i, j].item()
                        if p < self.eps:
                            retval[idx_to_rule[j]] = np.log(self.eps)
                        else:
                            retval[idx_to_rule[j]] = np.log(p)
                    return retval

                def get_token_prob(self):
                    probs = {}
                    n_words = len(h.state.query)
                    # 0 is UnknownToken
                    for j in range(1, token_pred.shape[1]):
                        p = token_pred[self.i, j].item()
                        t = \
                            self.action_sequence_encoder._token_encoder.decode(
                                torch.LongTensor([j]))
                        if t in probs:
                            probs[t] = probs.get(t) + p
                        else:
                            probs[t] = p
                    for j in range(n_words):
                        p = copy_pred[self.i, j].item()
                        t = h.state.query[j]
                        if t in probs:
                            probs[t] = probs.get(t) + p
                        else:
                            probs[t] = p

                    log_prob = {}
                    for t, p in probs.items():
                        if p < self.eps:
                            log_prob[t] = np.log(self.eps)
                        else:
                            log_prob[t] = np.log(p)
                    return log_prob
            funcs = Functions(i, self.action_sequence_encoder, self.eps)
            prob = LazyLogProbability(funcs.get_rule_prob,
                                      funcs.get_token_prob)
            retval.append((state, prob))
        return retval
