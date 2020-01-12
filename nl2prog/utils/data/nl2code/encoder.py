from torchnlp.encoders import LabelEncoder
from dataclasses import dataclass
from typing import List, Union
from nl2prog.language.encoder import Encoder as ActionEncoder
from nl2prog.language.action \
    import Rule, NodeType, CloseNode, ActionOptions


@dataclass
class Samples:
    words: List[str]
    rules: List[Rule]
    node_types: List[NodeType]
    tokens: List[Union[str, CloseNode]]


@dataclass
class Encoder:
    annotation_encoder: LabelEncoder
    action_sequence_encoder: ActionEncoder

    def __init__(self,
                 samples: Samples,
                 word_threshold: int,
                 token_threshold: int,
                 options: ActionOptions = ActionOptions(True, True)):
        """
        Parameters
        ----------
        samples: Samples
            The list of words, tokens, rules, and node_types
        word_threshold: int
        token_threshold: int
        """
        self.annotation_encoder = LabelEncoder(samples.words,
                                               min_occurrences=word_threshold)
        self.action_sequence_encoder = ActionEncoder(
            samples.rules, samples.node_types,
            samples.tokens, token_threshold, options=options)
