from torchnlp.encoders import LabelEncoder

from nl2prog.utils import Query
from typing import List, Union, Callable, Tuple, Any


class TransformQuery:
    def __init__(self, tokenize_query: Callable[[str], Query],
                 word_encoder: LabelEncoder):
        self.tokenize_query = tokenize_query
        self.word_encoder = word_encoder

    def __call__(self, query: Union[str, List[str]]) -> Tuple[List[str], Any]:
        if isinstance(query, str):
            query = self.tokenize_query(query)
        else:
            q = Query([], [])
            for word in query:
                q2 = self.tokenize_query(word)
                q.query_for_dnn.extend(q2.query_for_dnn)
                q.query_for_synth.extend(q2.query_for_synth)
            query = q

        return query.query_for_synth, \
            self.word_encoder.batch_encode(query.query_for_dnn)
