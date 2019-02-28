# python: 3.5
# encoding: utf-8
# time: 18/10/15
# author: zhenke

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


POS_INF = 100000
NEG_INF = -POS_INF


def seq_mask(seq_len, max_len):
    """Create sequence mask.

    :param seq_len: torch.long, shape [batch_size],
        Lengths of sequences in a batch.
    :param max_len: int
        The maximum sequence length in a batch.
    :return mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.
    """

    idx = torch.arange(max_len).to(seq_len).repeat(seq_len.size(0), 1)
    mask = torch.gt(seq_len.unsqueeze(1), idx).to(seq_len)

    # mask = [torch.ge(torch.LongTensor(seq_len), i + 1)
    # for i in range(max_len)]
    # mask = torch.stack(mask, 1)

    return mask


def seq2seq_mask(seq_lens_1, seq_lens_2, max_len_1, max_len_2):
    """Create sequence to sequence mask.

    Parameters
    ----------
    seq_lens_1 : torch.long, shape [batch_size],
        Lengths of first sequence.
    seq_lens_2 : torch.long, shape [batch_size],
        Lengths of second sequence.
    max_len_1 : int
        The maximum sequence length in a batch 1.
    max_len_2 : int
        The maximum sequence length in a batch 2.

    Returns
    -------
    mask: torch.long, shape [batch_size, max_len_1, max_len_2]
        Mask tensor for sequence-to-sequence.
    """

    mask_1 = seq_mask(seq_lens_1, max_len_1)  # [b, max_len_1]
    mask_2 = seq_mask(seq_lens_2, max_len_2)  # [b, max_len_2]
    return mask_1.unsqueeze(2) * mask_2.unsqueeze(1)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention score for two sequences.

    Reference
    ---------
    Ashish Vaswani, et. al. Attention Is All You Need. NIPS. 2017.
    (https://arxiv.org/pdf/1706.03762.pdf)

    Parameters
    ----------
    scale : bool, default True.
        Scale dot-product by 'key_dim' or not.

    Input
    -----
    query : torch.float, shape [batch_size, max_query_len, key_dim]
        Query vector.
    key : torch.float, shape [batch_size, max_key_len, key_dim]
        Key Vector.
    value : torch.float, shape [batch_size, max_key_len, value_dim]
        Value Vector.
    mask : torch.long, shape [batch_size, max_query_len, max_key_len],
        default None.
        Mask vector, 0 for position to be masked.

    Output
    ------
    output : torch.float, shape [batch_size, max_query_len, value_dim]
        Attended output vector.
    attention_score : torch.float, [batch_size, max_query_len, max_key_len]
        Attention score.
    """

    def __init__(self, scale=True):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale

    def forward(self, query, key, value, mask=None):
        key_dim = key.size(-1)  # key dimension

        attention = torch.bmm(query, key.transpose(1, 2))  # [b,mql,mkl]

        # scale by inverse square root of key dimension
        if self.scale:
            scale_factor = 1 / math.sqrt(key_dim)
            attention = attention * scale_factor

        # mask attention score by adding -inf
        if mask is not None:
            attention = attention + (1 - mask.float()) * NEG_INF

        # attention score
        attention_score = F.softmax(attention, dim=-1)  # [b,mql,mkl]

        # output
        output = torch.bmm(attention_score, value)  # [b,mql,vd]

        return output, attention_score


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention score for two sequences.

    Reference
    ---------
    Ashish Vaswani, et. al. Attention Is All You Need. NIPS. 2017.
    (https://arxiv.org/pdf/1706.03762.pdf)

    Parameters
    ----------
    num_heads : int, default 8
        Number of heads.
    model_dim : int, default 512
        Dimension of input vector.
    key_dim : int, default 64
        Dimension of query vector and key vector per head.
    value_dim : int, default 64
        Dimension of value vector per head.

    Input
    -----
    query : torch.float, shape [batch_size, max_query_len, model_dim]
        Query vector.
    key : torch.float, shape [batch_size, max_key_len, model_dim]
        Key Vector.
    value : torch.float, shape [batch_size, max_key_len, model_dim]
        Value Vector.
    mask : torch.long, shape [batch_size, max_query_len, max_key_len],
        default None.
        Mask vector, 0 for position to be masked.

    Output
    ------
    output : torch.float, shape [batch_size, max_query_len, model_dim]
        Attended output vector.
    """

    def __init__(self, num_heads=8, model_dim=512, key_dim=64, value_dim=64):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.model_dim = model_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.w_q = nn.Linear(model_dim, key_dim * num_heads, bias=False)
        self.w_k = nn.Linear(model_dim, key_dim * num_heads, bias=False)
        self.w_v = nn.Linear(model_dim, value_dim * num_heads, bias=False)

        self.att = ScaledDotProductAttention(scale=True)

        self.w_o = nn.Linear(value_dim * num_heads, model_dim, bias=False)

    def forward(self, query, key, value, mask=None):
        nh, kd, vd = self.num_heads, self.key_dim, self.value_dim
        bs, mql, _ = query.size()
        bs, mkl, _ = key.size()

        # linear mapping
        q = self.w_q(query)  # [b,mql,md]->[b,mql,kd*nh]
        k = self.w_k(key)  # [b,mkl,md]->[b,mkl,kd*nh]
        v = self.w_v(value)  # [b,mkl,md]->[b,mkl,vd*nh]

        # split heads, [nh*b,mql/mkl,kd/vd]
        q = q.reshape(bs, mql, nh, kd).permute(2, 0, 1, 3).reshape(-1, mql, kd)
        k = k.reshape(bs, mkl, nh, kd).permute(2, 0, 1, 3).reshape(-1, mkl, kd)
        v = v.reshape(bs, mkl, nh, vd).permute(2, 0, 1, 3).reshape(-1, mkl, vd)

        # attention, output [b*nh,mql,vd]
        mask = mask.repeat(nh, 1, 1)  # [nh,mql,mkl]
        o, att_score = self.att(q, k, v, mask=mask)

        # concat output from every head, [b,mql,nh*vd]
        o = o.reshape(nh, bs, mql, vd).permute(1, 2, 0, 3).reshape(bs, mql, -1)

        # linear mapping
        o = self.w_o(o)

        return o


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feed-forward module, composed of two linear modules.

    Reference
    ---------
    Ashish Vaswani, et. al. Attention Is All You Need. NIPS. 2017.
    (https://arxiv.org/pdf/1706.03762.pdf)

    Args
    ----
    input_dim : int, input dimension
    inner_dim : int, dimension of inner layer
    out_dim : int, dimension of output
    activation : str, default 'relu'. Activation function

    Input
    -----
    x : torch.float, shape [batch_size, ..., input_dim]

    Output
    ------
    o : torch.float, shape [batch_size, ..., out_dim]
    """

    def __init__(self, input_dim, inner_dim, out_dim, activation='relu'):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(input_dim, inner_dim)
        if activation == 'relu':
            self.act = nn.ReLU()
        else:
            raise ValueError("Illegal activation {activation}.")
        self.linear2 = nn.Linear(inner_dim, out_dim)

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


class TransformerEncoder(nn.Module):
    """Encoder for transformer model.

    Parameters
    ----------
    model_dim : int,
        Dimension of input vector.
    num_heads : int,
        Number of heads.
    dropout : float,
        Dropout probability.

    Inputs
    ------
    x : torch.float, size [batch, max_seq_len, model_dim]
        Input vector.
    seq_len : torch.long, size [batch]
        Sequence length.

    Outputs
    -------
    o : torch.float, size [batch, max_seq_len, model_dim]
    """

    def __init__(self, model_dim, num_heads, dropout=0.5):
        super(TransformerEncoder, self).__init__()

        self.att = MultiHeadAttention(
            num_heads=num_heads, model_dim=model_dim,
            key_dim=model_dim // num_heads,
            value_dim=model_dim // num_heads)
        self.norm_att = nn.LayerNorm(model_dim)

        self.ff = PositionWiseFeedForward(
            model_dim, model_dim * 4, model_dim)
        self.norm_ff = nn.LayerNorm(model_dim)

        self.drop = nn.Dropout(p=dropout)

    def forward(self, x, seq_len):
        msl = x.size(1)
        mask = seq2seq_mask(seq_len, seq_len, msl, msl)

        # multi-head attention
        a = self.drop(self.att(x, x, x, mask))  # [b,msl,d]
        a = self.norm_att(a + x)

        # position-wise feed-forward
        f = self.drop(self.ff(a))
        f = self.norm_ff(f + a)  # [b,msl,d]

        return f
