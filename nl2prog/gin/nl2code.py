from torch.utils.data import Dataset
from typing import Callable, Any, List, Optional
from torchnlp.encoders import LabelEncoder
from nl2prog.encoders import ActionSequenceEncoder
from nl2prog.gin import workspace
from nl2prog.utils import Query
from nl2prog.utils.data import get_samples, get_words
from nl2prog.language.ast import AST
from nl2prog.language.action import ActionSequence


def prepare_encoder(dataset: Dataset, word_threshold: int,
                    token_threshold: int, parse: Callable[[Any], AST],
                    to_action_sequence: Callable[[Any],
                                                 Optional[ActionSequence]],
                    extract_query: Callable[[Any], Query],
                    tokenize_token: Callable[[Any], List[str]],
                    encoder_path_prefix: str = ""):
    words = get_words(dataset, extract_query)
    samples = get_samples(dataset, tokenize_token,
                          to_action_sequence)

    qencoder = LabelEncoder(words, word_threshold)
    aencoder = ActionSequenceEncoder(samples, token_threshold)
    workspace.put(f"{encoder_path_prefix}query_encoder", qencoder)
    workspace.put(f"{encoder_path_prefix}action_sequence_encoder", aencoder)
