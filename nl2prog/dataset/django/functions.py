import re
from nltk import tokenize
from nl2prog.utils import Query

tokenizer = tokenize.WhitespaceTokenizer()


def tokenize_query(query: str) -> Query:
    """
    Tokenize query

    Parameters
    ----------
    query: str

    Returns
    -------
    Query
    """
    # Preprocess annotation
    def placeholder(id):
        return "####{}####".format(id)

    # Replace quoted string literals with placeholders
    mappings = {}
    word_to_placeholder = {}
    literal = r'\'\\\'\'|\"[^\"]+\"|\'[^\']+\'|`[^`]+`|"""[^"]+"""'
    while True:
        m = re.search(literal, query)
        if m is None:
            break

        w = m.group(0)[1:len(m.group(0)) - 1]
        if str(w) in word_to_placeholder:
            p = word_to_placeholder[str(w)]
        else:
            p = placeholder(len(mappings))
        query = query[:m.start()] + p + query[m.end():]

        assert (not ("####" in w))
        mappings[p] = str(w)
        word_to_placeholder[str(w)] = p

    query_for_synth = []
    query_for_dnn = []
    for word in tokenizer.tokenize(query):
        if word in mappings:
            query_for_synth.append(mappings[word])
        else:
            query_for_synth.append(word)
        query_for_dnn.append(word)

        vars = list(filter(lambda x: len(x) > 0,
                           word.split('.')))  # split by '.'
        if len(vars) > 1:
            for v in vars:
                query_for_synth.append(v)
                query_for_dnn.append(v)
    return Query(query_for_synth, query_for_dnn)
