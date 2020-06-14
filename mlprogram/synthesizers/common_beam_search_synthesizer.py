import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Callable, Optional, Tuple, Any, Dict, Union
from dataclasses import dataclass
from mlprogram.actions import ActionOptions, Rule, CloseNode
from mlprogram.actions import ActionSequence
from mlprogram.encoders import ActionSequenceEncoder
from mlprogram.synthesizers.beam_search_synthesizer import Hypothesis
from mlprogram.synthesizers \
    import BeamSearchSynthesizer as BaseBeamSearchSynthesizer, \
    IsSubtype, LazyLogProbability
from mlprogram.nn import TrainModel
from mlprogram.utils.data import Collate


logger = logging.getLogger(__name__)


@dataclass
class State:
    query: List[str]
    data: Dict[str, Any]


class CommonBeamSearchSynthesizer(BaseBeamSearchSynthesizer):
    def __init__(self, beam_size: int,
                 transform_input: Callable[[Any],
                                           Tuple[List[str], Dict[str, Any]]],
                 transform_action_sequence: Callable[
                     [ActionSequence, List[str]], Optional[Dict[str, Any]]],
                 collate: Collate,
                 input_reader: nn.Module,  # Callable[[Any], Tuple[Any, Any]],
                 action_sequence_reader: nn.Module,  # Callable[[Any], Any],
                 decoder: nn.Module,  # Callable[[Any, Any, Any,
                                      #           Optional[Any]],
                                      #          Tuple[Any, Optional[Any]]],
                 predictor: nn.Module,  # Callable[[Any],
                                        #          Tuple[PaddedSequenceWithMask,
                                        #          PaddedSequenceWithMask,
                                        #          PaddedSequenceWithMask]],
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
        self.transform_action_sequence = transform_action_sequence
        self.collate = collate
        self.input_reader = input_reader.to(device)
        self.action_sequence_reader = action_sequence_reader.to(device)
        self.decoder = decoder.to(device)
        self.predictor = predictor.to(device)
        self.action_sequence_encoder = action_sequence_encoder
        self.eps = eps

    @staticmethod
    def create(beam_size: int,
               transform_input: Callable[[Any],
                                         Tuple[List[str], Dict[str, Any]]],
               transform_action_sequence: Callable[[ActionSequence, List[str]],
                                                   Optional[Dict[str, Any]]],
               collate: Collate,
               model: TrainModel,
               action_sequence_encoder: ActionSequenceEncoder,
               is_subtype: IsSubtype,
               options: ActionOptions = ActionOptions(True, True),
               eps: float = 1e-8,
               max_steps: Optional[int] = None,
               device: torch.device = torch.device("cpu")):
        return CommonBeamSearchSynthesizer(
            beam_size, transform_input, transform_action_sequence,
            collate,
            model.input_reader, model.action_sequence_reader, model.decoder,
            model.predictor, action_sequence_encoder, is_subtype, options, eps,
            max_steps, device)

    def state_dict(self) -> Dict[str, Any]:
        return TrainModel(self.input_reader, self.action_sequence_reader,
                          self.decoder, self.predictor).state_dict()

    def load_state_dict(self, state_dict) -> None:
        TrainModel(self.input_reader, self.action_sequence_reader,
                   self.decoder, self.predictor).load_state_dict(state_dict)

    def initialize(self, input: Any) -> State:
        query_for_synth, input = self.transform_input(input)
        inputs = self.collate.collate([input])
        self.input_reader.eval()
        inputs = self.input_reader(**inputs)
        input = self.collate.split(inputs)[0]

        return State(query_for_synth, input)

    def batch_update(self, hs: List[Hypothesis]) \
            -> List[Tuple[State, LazyLogProbability]]:
        # Create batch of hypothesis
        data = []
        for h in hs:
            tmp = self.transform_action_sequence(
                h.action_sequence, h.state.query)
            if tmp is not None:
                new_data = {key: value.clone()
                            for key, value in h.state.data.items()}
                for key, value in tmp.items():
                    if value is None and key in new_data.keys():
                        continue
                    new_data[key] = value
                data.append(new_data)
            else:
                logger.warn(
                    "Invalid action_sequence is in the set of hypothesis")
        inputs = self.collate.collate(data)

        with torch.no_grad():
            self.action_sequence_reader.eval()
            self.decoder.eval()
            self.predictor.eval()
            inputs = self.action_sequence_reader(**inputs)
            inputs = self.decoder(**inputs)
            results = self.predictor(**inputs)
        # (len(hs), n_rules)
        rule_pred = \
            results["rule_probs"].data.cpu().reshape(len(hs), -1)
        # (len(hs), n_tokens)
        token_pred = \
            results["token_probs"].data.cpu().reshape(len(hs), -1)
        # (len(hs), query_length)
        copy_pred = \
            results["copy_probs"].data.cpu().reshape(len(hs), -1)
        inputs = {key: value for key, value in inputs.items()
                  if key in self.collate.options.keys()}
        data = self.collate.split(inputs)

        retval = []
        for i, (h, d) in enumerate(zip(hs, data)):
            state = State(h.state.query, d)

            class Functions:
                def __init__(self, i: int,
                             action_sequence_encoder: ActionSequenceEncoder,
                             eps: float):
                    self.i = i
                    self.action_sequence_encoder = action_sequence_encoder
                    self.eps = eps

                def get_rule_prob(self) -> Dict[Rule, float]:
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

                def get_token_prob(self) -> Dict[Union[CloseNode, str], float]:
                    probs: Dict[Union[CloseNode, str], float] = {}
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
