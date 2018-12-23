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
from .annotation import UNKNOWN
from .grammar import CLOSE_NODE, Grammar
from .dataset import Dataset
from .python.grammar import to_ast
from .decoder import Decoder
from .utils import bleu4

parser = argparse.ArgumentParser()
parser.add_argument('--context', "-c", type=str, default="cpu")
parser.add_argument('--seed', type=int, default=1000)
parser.add_argument(
    '--train', type=str, default=os.path.join("dataset", "django", "train"))
parser.add_argument(
    '--test', type=str, default=os.path.join("dataset", "django", "test"))
parser.add_argument('--max-query-length', type=int, default=70)
parser.add_argument('--max-action-length', type=int, default=100)
parser.add_argument('--word-threshold', type=int, default=5)
parser.add_argument('--token-threshold', type=int, default=5)
parser.add_argument('--embedding-size', type=int, default=128)
parser.add_argument('--node-type-embedding-size', type=int, default=64)
parser.add_argument('--lstm-state-size', type=int, default=256)
parser.add_argument('--hidden-state-size', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=10)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--train-epochs', type=int, default=50)
parser.add_argument(
    '--output', type=str, default=os.path.join("result", "django"))
parser.add_argument('--beam-size', type=int, default=15)
parser.add_argument('--validation-interval', type=int, default=5)
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

# Load dataset
logger.info("Load dataset")
data = Dataset(args.train, shuffle=True, rng=np.random.RandomState(args.seed))

# Load test dataset
logger.info("Load test dataset")
tdata = Dataset(args.test, shuffle=False)

# Create the sets of words and tokens
logger.info("Get words/token vocabulary")
words = data.words(args.word_threshold)
words.insert(0, UNKNOWN)
words.insert(0, '<empty>')
tokens = data.tokens(args.token_threshold)
tokens.insert(0, CLOSE_NODE)

word_to_id = {}
for i in range(len(words)):
    word_to_id[words[i]] = i

# Save grammar/words info
logger.info("Save grammar/words info")
with open(os.path.join(args.output, "words.json"), mode='w') as f:
    json.dump(word_to_id, f)
with open(os.path.join(args.output, "tokens.json"), mode='w') as f:
    json.dump(tokens, f)
with open(os.path.join(args.output, "rules.json"), mode='w') as f:
    json.dump(data.rules, f)
with open(os.path.join(args.output, "node_types.json"), mode='w') as f:
    json.dump(data.node_types, f)

# Prepare dataset
logger.info("Prepare dataset")
grammar = Grammar(data.node_types, data.rules, tokens)
data.prepare(word_to_id, grammar)
tdata.prepare(word_to_id, grammar)

# Dump dataset stats
logger.info("#rules: {}".format(len(data.rules)))
logger.info("#words: {}".format(len(word_to_id)))
logger.info("#tokens: {}".format(len(tokens)))
logger.info("#node_type: {}".format(len(data.node_types)))
logger.info("#samples: {}".format(len(data._data)))
logger.info("avg_query_length: {}".format(
    np.average(list(map(lambda x: len(x.annotation.query), data._data)))))
logger.info("max_query_length: {}".format(
    np.max(list(map(lambda x: len(x.annotation.query), data._data)))))
logger.info("avg_action_length: {}".format(
    np.average(list(map(lambda x: len(x.sequence), data._data)))))
logger.info("max_action_length: {}".format(
    np.max(list(map(lambda x: len(x.sequence), data._data)))))

# Create DNN model
logger.info("Create DNN model")
query = nn.Variable((args.batch_size, args.max_query_length), need_grad=False)
action = nn.Variable((args.batch_size, args.max_action_length, 3),
                     need_grad=False)
action_type = nn.Variable((args.batch_size, args.max_action_length, 3),
                          need_grad=False)
node_type = nn.Variable((args.batch_size, args.max_action_length),
                        need_grad=False)
parent_rule = nn.Variable((args.batch_size, args.max_action_length),
                          need_grad=False)
parent_index = nn.Variable((args.batch_size, args.max_action_length),
                           need_grad=False)
query_embed, query_embed_mask = model.encoder(
    query,
    len(word_to_id),
    args.embedding_size,
    args.lstm_state_size,
    dropout=args.dropout)
_, decoder_states, _, ctx_vector, decoder_mask, _ = model.decoder(
    action,
    action_type,
    node_type,
    parent_rule,
    parent_index,
    query_embed,
    query_embed_mask,
    len(data.rules),
    len(tokens),
    len(data.node_types),
    args.embedding_size,
    args.node_type_embedding_size,
    args.lstm_state_size,
    args.hidden_state_size,
    dropout=args.dropout)
rule_prob, gen_prob, token_prob, copy_prob = model.pred(
    decoder_states, ctx_vector, query_embed, query_embed_mask, len(data.rules),
    len(tokens), args.embedding_size, args.hidden_state_size)
loss = model.loss(action, action_type, decoder_mask, rule_prob, gen_prob,
                  token_prob, copy_prob)
error = model.top_k_error(
    action,
    action_type,
    decoder_mask,
    rule_prob.get_unlinked_variable(),
    gen_prob.get_unlinked_variable(),
    token_prob.get_unlinked_variable(),
    copy_prob.get_unlinked_variable(),
    k=1)

logger.info("Create test model")
tquery = nn.Variable((args.batch_size, args.max_query_length), need_grad=False)
taction = nn.Variable((args.batch_size, args.max_action_length, 3),
                      need_grad=False)
taction_type = nn.Variable((args.batch_size, args.max_action_length, 3),
                           need_grad=False)
tnode_type = nn.Variable((args.batch_size, args.max_action_length),
                         need_grad=False)
tparent_rule = nn.Variable((args.batch_size, args.max_action_length),
                           need_grad=False)
tparent_index = nn.Variable((args.batch_size, args.max_action_length),
                            need_grad=False)
tquery_embed, tquery_embed_mask = model.encoder(
    tquery,
    len(word_to_id),
    args.embedding_size,
    args.lstm_state_size,
    dropout=args.dropout,
    train=False)
taction_embed, tdecoder_states, _, tctx_vector, tdecoder_mask, thist = model.decoder(
    taction,
    taction_type,
    tnode_type,
    tparent_rule,
    tparent_index,
    tquery_embed,
    tquery_embed_mask,
    len(data.rules),
    len(tokens),
    len(data.node_types),
    args.embedding_size,
    args.node_type_embedding_size,
    args.lstm_state_size,
    args.hidden_state_size,
    dropout=args.dropout,
    train=False)
trule_prob, tgen_prob, ttoken_prob, tcopy_prob = model.pred(
    tdecoder_states, tctx_vector, tquery_embed, tquery_embed_mask,
    len(data.rules), len(tokens), args.embedding_size, args.hidden_state_size)
terror = model.top_k_error(
    taction,
    taction_type,
    tdecoder_mask,
    trule_prob,
    tgen_prob,
    ttoken_prob,
    tcopy_prob,
    k=1)

logger.info("Create a decoder")
decoder = Decoder(args.beam_size, args.max_query_length,
                  args.max_action_length, word_to_id, grammar,
                  args.embedding_size, args.node_type_embedding_size,
                  args.lstm_state_size, args.hidden_state_size, args.dropout)

logger.info("Create a solver")
solver = S.Adam()
solver.set_parameters(nn.get_parameters())


def create_batch(data, query, action, action_type, node_type, parent_rule,
                 parent_index):
    # Clear data
    query.data.zero()
    action.data.zero()
    action_type.data.zero()
    node_type.data.zero()
    parent_rule.data.zero()
    parent_index.data.zero()
    for i in range(args.batch_size):
        sample = data.next()
        query_length = min(args.max_query_length,
                           len(sample.encoder_input.query))
        sequence_length = min(args.max_action_length, len(sample.sequence))
        query.d[i, :query_length] = sample.encoder_input.query[:query_length]
        action.d[
            i, :
            sequence_length, :] = sample.decoder_input.action[:sequence_length]
        action_type.d[
            i, :
            sequence_length] = sample.decoder_input.action_type[:
                                                                sequence_length]
        node_type.d[
            i, :
            sequence_length] = sample.decoder_input.node_type[:sequence_length]
        parent_rule.d[
            i, :
            sequence_length] = sample.decoder_input.parent_action[:
                                                                  sequence_length]
        parent_index.d[
            i, :
            sequence_length] = sample.decoder_input.parent_index[:
                                                                 sequence_length]


logger.info("Create monitors")
import nnabla.monitor as M
monitor = M.Monitor(args.output)
monitor_loss = M.MonitorSeries("training loss", monitor, interval=data.size)
monitor_err = M.MonitorSeries("training error", monitor, interval=data.size)
monitor_tacc = M.MonitorSeries("test accuracy", monitor, interval=data.size)
monitor_tbleu4 = M.MonitorSeries("test bleu4", monitor, interval=data.size)
monitor_terr = M.MonitorSeries(
    "test error with oracle sequence", monitor, interval=data.size)
monitor_time = M.MonitorTimeElapsed(
    "training time", monitor, interval=data.size)

import transpyle
unparser = transpyle.python.unparser.NativePythonUnparser()
best_bleu4 = 0.0

# Training loop
iter = 0
logger.info("Start training")
while iter < data.size * args.train_epochs:
    prev_epoch = int(iter / data.size)
    create_batch(data, query, action, action_type, node_type, parent_rule,
                 parent_index)

    solver.zero_grad()
    loss.forward()
    error.forward()
    loss.backward(clear_buffer=True)
    solver.update()

    monitor_loss.add(iter, loss.d.copy())
    monitor_err.add(iter, error.d.copy())
    monitor_time.add(iter)
    iter += args.batch_size

    epoch = int(iter / data.size)
    if epoch != prev_epoch:
        if epoch % args.validation_interval == 0:
            # test with oracle sequence
            test_iter = 0
            e = 0.0
            cnt = 0
            while test_iter < tdata.size:
                create_batch(tdata, tquery, taction, taction_type, tnode_type,
                             tparent_rule, tparent_index)
                terror.forward()
                test_iter += args.batch_size
                e += terror.d.copy()

                cnt += 1
            monitor_terr.add(iter, e / cnt)

            # test
            sum_bleu4 = 0.0
            acc = 0.0
            N = 0
            for i in range(tdata.size):
                if i % 10 == 0:
                    logger.info("test : {} / {} samples".format(i, tdata.size))
                sample = tdata.next()

                length = min(args.max_query_length,
                             len(sample.encoder_input.query))
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
                    hypothesis = decoder.decode(sample.annotation,
                                                sample.encoder_input)
                    try:
                        if not (hypothesis is None):
                            result = unparser.unparse(
                                to_ast(hypothesis.sequence))
                            if result == reference:
                                acc += 1.0
                            sum_bleu4 += bleu4(reference, result)
                    except:
                        pass
            acc /= N
            sum_bleu4 /= N
            monitor_tacc.add(iter, acc)
            monitor_tbleu4.add(iter, sum_bleu4)

            if best_bleu4 < sum_bleu4:
                best_bleu4 = sum_bleu4
                logger.info("Save parameter (BLUE4={})".format(sum_bleu4))
                nn.save_parameters(os.path.join(args.output, "model.h5"))
                if best_bleu4 == 1.0:
                    break

logger.info("Save parameter")
nn.save_parameters(os.path.join(args.output, "final.h5"))
