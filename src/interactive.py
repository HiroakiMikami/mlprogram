import os
import json
import argparse
import numpy as np
import nnabla as nn
import nnabla.solvers as S
import nnabla.functions as F
import nnabla.logger as logger
from nnabla.ext_utils import get_extension_context
from nltk import tokenize

import src.model as model
from .dataset import Dataset
from .grammar import Grammar, Rule, NodeType
from .python.grammar import to_ast
from .annotation import to_encoder_input, Annotation
from .decoder import Decoder

tokenizer = tokenize.WhitespaceTokenizer()

parser = argparse.ArgumentParser()
parser.add_argument('--context', "-c", type=str, default="cpu")
parser.add_argument('--max-query-length', type=int, default=70)
parser.add_argument('--max-action-length', type=int, default=100)
parser.add_argument('--embedding-size', type=int, default=128)
parser.add_argument('--node-type-embedding-size', type=int, default=64)
parser.add_argument('--lstm-state-size', type=int, default=256)
parser.add_argument('--hidden-state-size', type=int, default=50)
parser.add_argument(
    '--result', type=str, default=os.path.join("result", "django"))
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--beam-size', type=int, default=15)
args = parser.parse_args()

# Context
extension_module = args.context
if args.context is None:
    extension_module = 'cpu'
logger.info("Running in %s" % extension_module)
ctx = get_extension_context(extension_module, device_id=0)
nn.set_default_context(ctx)

logger.info("Load grammar/words info")
with open(os.path.join(args.result, "words.json")) as f:
    word_to_id = json.load(f)
with open(os.path.join(args.result, "tokens.json")) as f:
    tokens = json.load(f)
with open(os.path.join(args.result, "rules.json")) as f:
    rules = json.load(f)
    rules = list(map(Rule.from_json, rules))
with open(os.path.join(args.result, "node_types.json")) as f:
    node_types = json.load(f)
    node_types = list(map(NodeType.from_json, node_types))
grammar = Grammar(node_types, rules, tokens)

logger.info("Load parameter")
nn.load_parameters(os.path.join(args.result, "model.h5"))

logger.info("Prepare decoder")
decoder = Decoder(args.beam_size, args.max_query_length,
                  args.max_action_length, word_to_id, grammar,
                  args.embedding_size, args.node_type_embedding_size,
                  args.lstm_state_size, args.hidden_state_size, args.dropout)

raw_query = input(">")
raw_query = tokenizer.tokenize(raw_query)
annotation = Annotation(raw_query, {})
encoder_input = to_encoder_input(annotation, word_to_id)

import transpyle
unparser = transpyle.python.unparser.NativePythonUnparser()

h = decoder.decode(annotation, encoder_input)
result = None
try:
    if not (h is None):
        result = unparser.unparse(to_ast(h.sequence))
except:
    pass

if result is None:
    logger.info("Fail to synthesis")
else:
    logger.info(result)
