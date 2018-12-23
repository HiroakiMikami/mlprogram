import nnabla as nn
import nnabla.functions as F
import nnabla.initializer as I

from .nn.layers.embeddings import embedding
from .nn.layers.core import dense
from .nn.layers.recurrent import bilstm
from .nn.layers.utils import split, concatenate, embed_inverse
from .nn.cond_att_lstm import cond_att_lstm
from .nn.pointer_net import pointer_net


def encoder(query_tokens,
            word_num,
            embedding_size,
            state_size,
            dropout=0.0,
            train=True):
    """
    query_tokens: (batch_size, max_query_length)
    """
    with nn.parameter_scope("word_embedding"):
        # query_token_embed: (batch_size, max_query_length, embedding_size)
        # query_token_embed_mask: (batch_size, max_query_length)
        query_token_embed, query_token_embed_mask = embedding(
            query_tokens, word_num, embedding_size, mask_zero=True)
    with nn.parameter_scope("encoder"):
        # (batch_size, max_query_length, state_size)
        query_embed = bilstm(
            query_token_embed,
            query_token_embed_mask,
            int(state_size / 2),
            dropout=dropout,
            train=train)

    return query_embed, query_token_embed_mask


def decoder(target_action,
            target_action_type,
            target_node_type,
            target_parent_rule,
            target_parent_index,
            query_embed,
            query_embed_mask,
            rule_num,
            token_num,
            node_type_num,
            embedding_size,
            node_type_embedding_size,
            state_size,
            hidden_size,
            previous_action_embed=None,
            initial_state=None,
            initial_cell=None,
            hist=None,
            dropout=0.0,
            train=True):
    """
    target_action: (batch_size, max_action_length, 3)
    target_action_type: (batch_size, max_action_length, 3)
    target_node_type: (batch_size, max_action_length)
    target_parent_rule: (batch_size, max_action_length)
    target_parent_index: (batch_size, max_action_length)
    """
    batch_size, max_action_length, _ = target_action.shape

    # Node type ebedding
    with nn.parameter_scope("node_type_embedding"):
        target_node_type_embed = embedding(
            target_node_type,
            node_type_num,
            node_type_embedding_size,
            mask_zero=False,
            init=I.NormalInitializer(0.01))

    # Previous action embedding
    ## (batch_size, max_action_length)
    target_apply_rule, target_gen_token, target_copy_token = split(
        target_action, axis=2)
    with nn.parameter_scope("rule_embedding"):
        # (batch_size, max_action_length, embedding_size)
        target_apply_rule_embed = embedding(
            target_apply_rule,
            rule_num,
            embedding_size,
            mask_zero=False,
            init=I.NormalInitializer(0.01))
        target_apply_rule_embed = F.reshape(
            target_apply_rule_embed,
            (batch_size, max_action_length, 1, embedding_size))
    with nn.parameter_scope("token_embedding"):
        # (batch_size, max_action_length, embedding_size)
        target_gen_token_embed = embedding(
            target_gen_token,
            token_num,
            embedding_size,
            mask_zero=False,
            init=I.NormalInitializer(0.01))
        target_gen_token_embed = F.reshape(
            target_gen_token_embed,
            (batch_size, max_action_length, 1, embedding_size))
    target_copy_token = F.reshape(target_copy_token,
                                  (batch_size, max_action_length, 1, 1))
    target_copy_token = F.broadcast(
        target_copy_token, (batch_size, max_action_length, 1, embedding_size))
    target_copy_token *= 0
    # (batch_size, max_action_length, 3, embedding_size)
    target_action_embed = concatenate(
        target_apply_rule_embed,
        target_gen_token_embed,
        target_copy_token,
        axis=2)
    target_action_type2 = F.reshape(target_action_type,
                                    (batch_size, max_action_length, 3, 1))
    target_action_type2 = F.broadcast(
        target_action_type2,
        (batch_size, max_action_length, 3, embedding_size))
    # (batch_size, max_action_length, 3, embedding_size)
    target_action_embed = target_action_embed * target_action_type2
    # (batch_size, max_action_length, embedding_size)
    target_action_embed = F.sum(target_action_embed, axis=2)

    # Shift action
    if previous_action_embed is None:
        previous_action_embed = nn.Variable((batch_size, 1, embedding_size),
                                            need_grad=False)
        previous_action_embed.data.zero()
    # (batch_size, max_action_length + 1, embedding_size)
    target_action_embed = concatenate(
        previous_action_embed, target_action_embed, axis=1)
    # (batch_size, max_action_length, embedding_size)
    target_action_embed = F.slice(
        target_action_embed,
        start=[0, 0, 0],
        stop=[batch_size, max_action_length, embedding_size])

    # Parent action embedding
    parent_rule_mask = 1 - F.equal_scalar(target_parent_rule,
                                          0)  # (batch_size, max_action_length)
    parent_rule_mask = F.reshape(parent_rule_mask,
                                 (batch_size, max_action_length, 1))
    parent_rule_mask = F.broadcast(
        parent_rule_mask, (batch_size, max_action_length, embedding_size))
    with nn.parameter_scope("rule_embedding"):
        target_parent_rule_embed = embedding(
            target_parent_rule, rule_num, embedding_size, mask_zero=False)
    target_parent_rule_embed = parent_rule_mask * target_parent_rule_embed

    # (batch_size, max_action_length, embedding_size * 2 + node_type_embedding_size)
    decoder_input = concatenate(
        target_action_embed,
        target_node_type_embed,
        target_parent_rule_embed,
        axis=2)
    target_action_mask = 1 - F.equal_scalar(
        F.sum(target_action_type, axis=2),
        0)  # (batch_size, max_action_length)
    with nn.parameter_scope("decoder"):
        decoder_hidden_states, decoder_cells, ctx_vectors, new_hist = cond_att_lstm(
            decoder_input,
            target_parent_index,
            target_action_mask,
            query_embed,
            query_embed_mask,
            state_size,
            hidden_size,
            initial_state=initial_state,
            initial_cell=initial_cell,
            hist=hist,
            dropout=dropout,
            train=train)
    return target_action_embed, decoder_hidden_states, decoder_cells, ctx_vectors, target_action_mask, new_hist


def pred(decoder_hidden_states, ctx_vectors, query_embed, query_embed_mask,
         rule_num, token_num, embedding_size, hidden_size):
    """
    decoder_hidden_states: (batch_size, max_action_length, decoder_state_size)
    ctx_vectors: (batch_size, max_action_length, encoder_state_size)
    """
    batch_size, max_action_length, _ = decoder_hidden_states.shape
    dc = concatenate(decoder_hidden_states, ctx_vectors, axis=2)
    with nn.parameter_scope("decoder_state_rule"):
        # (batch_size, max_action_length, embedding_size)
        decoder_hidden_state_trans_rule = dense(
            decoder_hidden_states, embedding_size, base_axis=2)
    with nn.parameter_scope("decoder_state_token"):
        # (batch_size, max_action_length, decoder_state_size + encoder_state_size)
        # (batch_size, max_action_length, embedding_size)
        decoder_hidden_state_trans_token = dense(
            dc, embedding_size, base_axis=2)

    with nn.parameter_scope("rule_embedding"):
        # (batch_size, max_action_length, rule_num)
        rule_predict = embed_inverse(
            decoder_hidden_state_trans_rule,
            rule_num,
            embedding_size,
            base_axis=2)
        embed_b = nn.parameter.get_parameter_or_create(
            "embed/b", (rule_num, ), need_grad=True)
        embed_b.data.zero()
        embed_b = F.reshape(embed_b, (1, 1, rule_num), inplace=False)
        embed_b = F.broadcast(embed_b,
                              (batch_size, max_action_length, rule_num))
        rule_predict = F.softmax(rule_predict + embed_b)

    with nn.parameter_scope("gen_action"):
        terminal_gen_action_prob = dense(
            decoder_hidden_states, 2, base_axis=2, activation=F.softmax)

    with nn.parameter_scope("token_embedding"):
        # (batch_size, max_action_length, token_num)
        token_predict = embed_inverse(
            decoder_hidden_state_trans_token,
            token_num,
            embedding_size,
            base_axis=2)
        embed_b = nn.parameter.get_parameter_or_create(
            "embed/b", (token_num, ), need_grad=True)
        embed_b.data.zero()
        embed_b = F.reshape(embed_b, (1, 1, token_num), inplace=False)
        embed_b = F.broadcast(embed_b,
                              (batch_size, max_action_length, token_num))
        token_predict = F.softmax(token_predict + embed_b)

    with nn.parameter_scope("copy_token"):
        # (batch_size, max_action_length, max_query_length)
        copy_prob = pointer_net(query_embed, query_embed_mask, dc, hidden_size)
    return rule_predict, terminal_gen_action_prob, token_predict, copy_prob


def loss(target_action, target_action_type, target_action_mask, rule_prob,
         terminal_gen_action_prob, token_prob, copy_prob):
    batch_size, max_action_length, _ = target_action.shape
    _, _, rule_num = rule_prob.shape
    _, _, token_num = token_prob.shape
    _, _, max_query_length = copy_prob.shape

    # (batch_size, max_action_length)
    target_rule, target_token, target_copy = F.split(target_action, axis=2)

    target_rule = F.reshape(target_rule, (batch_size, max_action_length, 1))
    target_rule = F.one_hot(
        target_rule, (rule_num, ))  # (batch_size, max_action_length, rule_num)
    rule_tgt_prob = rule_prob * target_rule  # (batch_size, max_action_length, rule_num)
    rule_tgt_prob = F.sum(
        rule_tgt_prob, axis=2)  # (batch_size, max_action_length)

    target_token = F.reshape(target_token, (batch_size, max_action_length, 1))
    target_token = F.one_hot(
        target_token,
        (token_num, ))  # (batch_size, max_action_length, token_num)
    token_tgt_prob = token_prob * target_token  # (batch_size, max_action_length, token_num)
    token_tgt_prob = F.sum(
        token_tgt_prob, axis=2)  # (batch_size, max_action_length)

    target_copy = F.reshape(target_copy, (batch_size, max_action_length, 1))
    target_copy = F.one_hot(
        target_copy, (max_query_length,
                      ))  # (batch_size, max_action_length, max_query_lenght)
    copy_tgt_prob = copy_prob * target_copy  # (batch_size, max_action_length, max_query_length)
    copy_tgt_prob = F.sum(
        copy_tgt_prob, axis=2)  # (batch_size, max_action_length)

    # (batch_size, max_action_length)
    gen_token_prob, copy_token_prob = F.split(terminal_gen_action_prob, axis=2)
    # (batch_size, max_action_length)
    rule_mask, token_mask, copy_mask = F.split(target_action_type, axis=2)

    # (batch_size, max_action_length)
    target_prob = rule_mask * rule_tgt_prob + \
                  token_mask * gen_token_prob * token_tgt_prob + \
                  copy_mask * copy_token_prob * copy_tgt_prob
    # (batch_size, max_action_length)
    likelihood = F.log(target_prob + 1e-7)
    loss = -likelihood * target_action_mask
    # (batch_size)
    loss = F.sum(loss, axis=1)
    return F.mean(loss)


def top_k_error(target_action,
                target_action_type,
                target_action_mask,
                rule_prob,
                terminal_gen_action_prob,
                token_prob,
                copy_prob,
                k=5):
    batch_size, max_action_length, _ = target_action.shape
    _, _, rule_num = rule_prob.shape
    _, _, token_num = token_prob.shape
    _, _, max_query_length = copy_prob.shape

    # (batch_size, max_action_length)
    rule_mask, token_mask, copy_mask = F.split(target_action_type, axis=2)

    # (batch_size, max_action_length)
    target_rule, target_token, target_copy = F.split(target_action, axis=2)
    target_rule = F.reshape(target_rule, (batch_size, max_action_length, 1))

    # (batch_size, max_action_length)
    gen_token_prob, copy_token_prob = F.split(terminal_gen_action_prob, axis=2)
    gen_token_prob = F.reshape(gen_token_prob,
                               (batch_size, max_action_length, 1))
    gen_token_prob = F.broadcast(gen_token_prob,
                                 (batch_size, max_action_length, token_num))
    copy_token_prob = F.reshape(copy_token_prob,
                                (batch_size, max_action_length, 1))
    copy_token_prob = F.broadcast(
        copy_token_prob, (batch_size, max_action_length, max_query_length))
    # (batch_size, max_action_length, token_num)
    token_prob = gen_token_prob * token_prob
    # (batch_size, max_action_length, max_query_length)
    copy_prob = copy_token_prob * copy_prob
    # (batch_size, max_action_length, token_num + max_query_length)
    gen_or_copy = F.concatenate(token_prob, copy_prob, axis=2)

    # (batch_size, max_action_length)
    token_label = token_mask * target_token + (copy_mask *
                                               (target_copy + token_num))
    token_label = F.reshape(token_label, (batch_size, max_action_length, 1))

    # (batch_size, max_action_length, 1)
    rule_err = F.top_n_error(rule_prob, target_rule, axis=2, n=k)
    rule_err = F.reshape(rule_err, (batch_size, max_action_length))
    # (batch_size, max_action_length, 1)
    token_err = F.top_n_error(gen_or_copy, token_label, axis=2, n=k)
    token_err = F.reshape(token_err, (batch_size, max_action_length))

    # (batch_size, max_action_length)
    err = rule_mask * rule_err + (token_mask + copy_mask) * token_err
    # (batch_size,)
    num = F.sum(
        rule_mask, axis=1) + F.sum(
            token_mask, axis=1) + F.sum(
                copy_mask, axis=1)
    # (batch_size,)
    err = F.sum(err, axis=1)
    # (batch_size,)
    err = err / (num + 1e-7)
    return F.mean(err)
