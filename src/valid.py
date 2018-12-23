import os
import json
import argparse
import numpy as np
import nnabla as nn
import nnabla.solvers as S
import nnabla.functions as F
import nnabla.logger as logger
from nnabla.ext_utils import get_extension_context

import src.model as model
from .dataset import Dataset
from .grammar import NodeType, Rule, Grammar
from .python.grammar import *
from .decoder import Decoder
from .utils import bleu4

parser = argparse.ArgumentParser()
parser.add_argument('--context', "-c", type=str, default="cpu")
parser.add_argument(
    '--valid', type=str, default=os.path.join("dataset", "django", "valid"))
parser.add_argument('--max-query-length', type=int, default=70)
parser.add_argument('--max-action-length', type=int, default=100)
parser.add_argument('--embedding-size', type=int, default=128)
parser.add_argument('--node-type-embedding-size', type=int, default=64)
parser.add_argument('--lstm-state-size', type=int, default=256)
parser.add_argument('--hidden-state-size', type=int, default=50)
parser.add_argument(
    '--result', type=str, default=os.path.join("result", "django"))
parser.add_argument(
    '--output', type=str, default=os.path.join("result", "django", "valid"))
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

# Create directory for output
if not os.path.exists(args.output):
    os.makedirs(args.output)

# Load grammar info
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

# Load dataset
logger.info("Load dataset")
data = Dataset(args.valid, shuffle=False)
data.prepare(word_to_id, grammar)

logger.info("Create monitors")
import nnabla.monitor as M
monitor = M.Monitor(args.output)
monitor_vacc = M.MonitorSeries("validation-accuracy", monitor, interval=1)
monitor_vbleu4 = M.MonitorSeries("validation-bleu4", monitor, interval=1)

logger.info("Load parameter")
nn.load_parameters(os.path.join(args.result, "model.h5"))

logger.info("Prepare decoder")
decoder = Decoder(args.beam_size, args.max_query_length,
                  args.max_action_length, word_to_id, grammar,
                  args.embedding_size, args.node_type_embedding_size,
                  args.lstm_state_size, args.hidden_state_size, args.dropout)

import transpyle
unparser = transpyle.python.unparser.NativePythonUnparser()
# validation
sum_bleu4 = 0.0
acc = 0.0
N = 0
for i in range(data.size):
    if i % 10 == 0:
        logger.info("valid : {} / {} samples".format(i, data.size))
    sample = data.next()

    length = min(args.max_query_length, len(sample.encoder_input.query))
    if len(sample.sequence) > args.max_action_length:
        continue

    valid = False
    try:
        reference = unparser.unparse(to_ast(sample.sequence))
        valid = True
        N += 1
    except RuntimeError as e:
        pass

    if valid:
        h = decoder.decode(sample.annotation, sample.encoder_input)
        try:
            if not (h is None):
                result = unparser.unparse(to_ast(h.sequence))
                if result == reference:
                    acc += 1.0
                sum_bleu4 += bleu4(reference, result)
        except:
            pass
acc /= N
sum_bleu4 /= N
monitor_vacc.add(1, acc)
monitor_vbleu4.add(1, sum_bleu4)
