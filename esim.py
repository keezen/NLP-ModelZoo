# python: 3.5
# encoding: utf-8
# time: 18/10/15
# author: zhenke
# pytorch: 0.4.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


NEG_INF = -10000
TINY_FLOAT = 1e-6


def mask_softmax(matrix, mask=None):
    """Perform softmax on length dimension with masking.

    :param matrix: torch.float, shape [batch_size, .., max_len]
    :param mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.
    """

    if mask is None:
        result = F.softmax(matrix, dim=-1)
    else:
        mask_norm = ((1 - mask) * NEG_INF).to(matrix)
        for i in range(matrix.dim() - mask_norm.dim()):
            mask_norm = mask_norm.unsqueeze(1)
        result = F.softmax(matrix + mask_norm, dim=-1)

    return result


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

    return mask


class DynamicLSTM(nn.Module):
    """
    Dynamic LSTM module, which can handle variable length input sequence.

    Args:
        input_size : input size
        hidden_size : hidden size
        num_layers : number of hidden layers. Default: 1
        dropout : dropout rate. Default: 0.5
        bidirectional : If True, becomes a bidirectional RNN. Default: False.

    Inputs: input, seq_lens
        input: tensor, shaped [batch, max_step, input_size]
        seq_lens: tensor, shaped [batch], sequence lengths of batch

    Outputs: output
        output: tensor, shaped [batch, max_step, num_directions * hidden_size],
             tensor containing the output features (h_t) from the last layer
             of the LSTM, for each t.
    """

    def __init__(self, input_size, hidden_size=100,
                 num_layers=1, dropout=0., bidirectional=False):
        super(DynamicLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, bias=True,
            batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, seq_lens):
        # sort input by descending length
        _, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        x_sort = torch.index_select(x, dim=0, index=idx_sort)
        seq_lens_sort = torch.index_select(seq_lens, dim=0, index=idx_sort)

        # pack input
        x_packed = pack_padded_sequence(
            x_sort, seq_lens_sort, batch_first=True)

        # pass through rnn
        y_packed, _ = self.lstm(x_packed)

        # unpack output
        y_sort, length = pad_packed_sequence(y_packed, batch_first=True)

        # unsort output to original order
        y = torch.index_select(y_sort, dim=0, index=idx_unsort)

        return y


class AttentionDot(torch.nn.Module):
    """Calculate dot-product attention score for every time step given
    query and mask, with query of any shape.

    Parameters
    ----------
    n_channels : int, Number of channels in vector.
    normalize : bool, Normalize or not.

    Input
    -----
    query : torch.float, shape [batch_size, .., n_channels]
        Query vector.
    context : torch.float, shape [batch_size, max_cont_len, n_channels]
        Context Vector.
    mask : torch.long, shape [batch_size, max_cont_len], default None
        Mask vector for context.

    Output
    ------
    attention_score : torch.float, shape [batch_size, .., max_cont_len]
        Attention score for every token in context.
    """

    def __init__(self, n_channels, normalize=True):
        super(AttentionDot, self).__init__()
        self.normalize = normalize

    def forward(self, query, context, mask=None):
        similarities = self._atten_forward(query, context)
        if self.normalize:
            return mask_softmax(similarities, mask)
        return similarities

    def _atten_forward(self, query, context):
        """Additive attention given query.

        Parameters
        ----------
        query : torch.float, shape [batch_size, .., n_channels]
            Query vector.
            Query vector.
        context : torch.float, shape [batch_size, max_cont_len, n_channels]
            Context vector.

        Returns
        -------
        similarities : torch.float, shape [batch_size, .., max_cont_len]
            Similarities between query and every context token.
        """

        dim_q = query.dim()  # number of dimensions in query
        dim_c = context.dim()  # number of dimensions in context

        # query
        map_q = query.unsqueeze(-2)  # [bc,..,nc]->[bc,..,1,nc]

        # context
        map_c = context  # [bs,mcl,nc]->[bs,mcl,nc]
        for i in range(dim_q + 1 - dim_c):
            map_c = map_c.unsqueeze(1)  # [bc,..1..,mcl,nc]

        similar = torch.sum(map_q * map_c, dim=-1)  # [bs,..,mcl]

        return similar


class ESIM(torch.nn.Module):
    """Standard enhanced sequential inference model (ESIM) for semantic
    matching, with shared parameter for premise and hypothesis
    and dot-product attention.

    Reference
    ---------
    * Qian Chen, et. al. Enhanced LSTM for Natural Language Inference.
        ACL. 2017.

    Parameters
    ----------
    embed_dim : int, default 300
        Dimension of embedding layer before ESIM.
    hidden_dim : int, default 300
        Dimension of hidden state in Bi-RNN layer.
    dropout : float, default 0.5
        Dropout.

    Input
    -----
    premise : torch.float, size [batch, max_prem_len, embed_dim]
        Vector for premise.
    hypothesis : torch.float, size [batch, max_hypo_len, embed_dim]
        Vector for hypothesis.
    prem_lens : torch.long, size [batch]
        Premise sequence length.
    hypo_lens : torch.long, size [batch]
        Hypothesis sequence length.

    Output
    ------
    rep_pair : torch.float, size [batch, hidden_dim]
        Vector for premise-hypothesis pair.
    """

    def __init__(self, embed_dim=300, hidden_dim=300, dropout=0.5):
        super(ESIM, self).__init__()

        self.rnn_enc = DynamicLSTM(
            embed_dim, hidden_dim, dropout=dropout, bidirectional=True)

        self.att_inf = AttentionDot(2 * hidden_dim)

        self.fc_map = nn.Linear(8 * hidden_dim, hidden_dim)
        self.act_map = nn.ReLU()

        self.rnn_comp = DynamicLSTM(
            hidden_dim, hidden_dim, dropout=dropout, bidirectional=True)

        self.fc_mlp = nn.Linear(hidden_dim * 8, hidden_dim)
        self.act_mlp = nn.Tanh()

        self.drop = nn.Dropout(p=dropout)

    def forward(self, premise, hypothesis, prem_lens, hypo_lens):
        max_prem_len = premise.size(1)
        max_hypo_len = hypothesis.size(1)
        mask_p = seq_mask(prem_lens, max_prem_len)  # [b,pl]
        mask_h = seq_mask(hypo_lens, max_hypo_len)  # [b,hl]
        prem_lens_one = torch.where(  # at least length 1
            prem_lens > 0, prem_lens, torch.ones_like(prem_lens))
        hypo_lens_one = torch.where(
            hypo_lens > 0, hypo_lens, torch.ones_like(hypo_lens))

        # input encoding
        p = self.rnn_enc(premise, prem_lens_one)  # [b,pl,e]->[b,pl,2*h]
        h = self.rnn_enc(hypothesis, hypo_lens_one)  # # [b,hl,e]->[b,hl,2*h]

        # local inference
        att_score_p = self.att_inf(p, h, mask_h)  # [b,pl,hl]
        att_score_h = self.att_inf(h, p, mask_p)  # [b,hl,pl]
        p_local = torch.sum(
            att_score_p.unsqueeze(-1) * h.unsqueeze(1),
            dim=-2).squeeze(-2)  # [b,pl,2*h]
        h_local = torch.sum(
            att_score_h.unsqueeze(-1) * p.unsqueeze(1),
            dim=-2).squeeze(-2)  # [b,hl,2*h]
        m_p = torch.cat(  # [b,pl,8*h]
            [p, p_local, p - p_local, p * p_local],
            dim=-1)
        m_h = torch.cat(  # [b,hl,8*h]
            [h, h_local, h - h_local, h * h_local],
            dim=-1)

        # inference composition
        m_p = self.drop(self.act_map(self.fc_map(m_p)))  # [b,pl,8*h]->[b,pl,h]
        m_h = self.drop(self.act_map(self.fc_map(m_h)))  # [b,hl,8*h]->[b,hl,h]
        v_p = self.rnn_comp(m_p, prem_lens_one)  # [b,pl,h]->[b,pl,2*h]
        v_h = self.rnn_comp(m_h, hypo_lens_one)  # [b,hl,h]->[b,hl,2*h]
        v_p_avg = torch.sum(v_p * mask_p.unsqueeze(-1).to(v_p), dim=1) / \
            (prem_lens.unsqueeze(-1).to(v_p) + TINY_FLOAT)  # [b,2*h]
        v_h_avg = torch.sum(v_h * mask_h.unsqueeze(-1).to(v_h), dim=1) / \
            (hypo_lens.unsqueeze(-1).to(v_h) + TINY_FLOAT)  # [b,2*h]
        v_p_max, _ = torch.max(  # [b,2*h]
            (1 - mask_p.unsqueeze(-1).to(v_p)) * NEG_INF + v_p,
            dim=1)
        v_h_max, _ = torch.max(  # [b,2*h]
            (1 - mask_h.unsqueeze(-1).to(v_h)) * NEG_INF + v_h,
            dim=1)
        v = torch.cat(  # [b,8*h]
            [v_p_avg, v_p_max, v_h_avg, v_h_max],
            dim=-1)

        # feed-forward
        o = self.drop(self.act_mlp(self.fc_mlp(v)))  # [b,8*h]->[b,h]

        return o
