import torch
import torch.nn.functional as F

NEG_INF = -10000
TINY_FLOAT = 1e-6


def mask_softmax(matrix, mask=None):
    """Perform softmax on length dimension with masking.

    Parameters
    ----------
    matrix: torch.float, shape [batch_size, .., max_len]
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.

    Returns
    -------
    output: torch.float, shape [batch_size, .., max_len]
        Normalized output in length dimension.
    """

    if mask is None:
        result = F.softmax(matrix, dim=-1)
    else:
        mask_norm = ((1 - mask) * NEG_INF).to(matrix)
        for i in range(matrix.dim() - mask_norm.dim()):
            mask_norm = mask_norm.unsqueeze(1)
        result = F.softmax(matrix + mask_norm, dim=-1)

    return result


def mask_mean(seq, mask=None):
    """Compute mask average on length dimension.

    Parameters
    ----------
    seq : torch.float, size [batch, max_seq_len, n_channels],
        Sequence vector.
    mask : torch.long, size [batch, max_seq_len],
        Mask vector, with 0 for mask.

    Returns
    -------
    mask_mean : torch.float, size [batch, n_channels]
        Mask mean of sequence.
    """

    if mask is None:
        return torch.mean(seq, dim=1)

    mask_sum = torch.sum(  # [b,msl,nc]->[b,nc]
        seq * mask.unsqueeze(-1).float(), dim=1)
    seq_len = torch.sum(mask, dim=-1)  # [b]
    mask_mean = mask_sum / (seq_len.unsqueeze(-1).float() + TINY_FLOAT)

    return mask_mean


def mask_max(seq, mask=None):
    """Compute mask max on length dimension.

    Parameters
    ----------
    seq : torch.float, size [batch, max_seq_len, n_channels],
        Sequence vector.
    mask : torch.long, size [batch, max_seq_len],
        Mask vector, with 0 for mask.

    Returns
    -------
    mask_max : torch.float, size [batch, n_channels]
        Mask max of sequence.
    """

    if mask is None:
        return torch.mean(seq, dim=1)

    torch
    mask_max, _ = torch.max(  # [b,msl,nc]->[b,nc]
        seq + (1 - mask.unsqueeze(-1).float()) * NEG_INF,
        dim=1)

    return mask_max


def seq_mask(seq_len, max_len):
    """Create sequence mask.

    Parameters
    ----------
    seq_len: torch.long, shape [batch_size],
        Lengths of sequences in a batch.
    max_len: int
        The maximum sequence length in a batch.

    Returns
    -------
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.
    """

    idx = torch.arange(max_len).to(seq_len).repeat(seq_len.size(0), 1)
    mask = torch.gt(seq_len.unsqueeze(1), idx).to(seq_len)

    return mask



def list_level(index_list):
    """Nested level of list."""

    if not isinstance(index_list, (tuple, list)):
        return 0

    if len(index_list) == 0:
        return 1

    return list_level(index_list[0]) + 1


def get_pad_shape(index_list):
    """Find pad shape of nested list.

    Returns
    -------
    pad_shape : list, shape of padded list
    """

    if not isinstance(index_list, (list, tuple)):
        return []

    if len(index_list) == 0:  # empty list
        return [0]

    dim0 = len(index_list)

    shape_other = None
    for item in index_list:
        shape = get_pad_shape(item)
        if shape_other is None:
            shape_other = shape
        else:  # keep max sub-shape
            assert len(shape) == len(shape_other)
            shape_other = [
                max(shape_other[i], shape[i])
                for i in range(len(shape_other))]

    pad_shape = [dim0] + shape_other

    return pad_shape


def pad_list_shape(index_list, pad_shape, pad_index=0):
    """Pad list given shape.

    Parameters
    ----------
    pad_shape : list, shape to pad

    Returns
    -------
    list_pad : np.ndarray, whose shape is pad_shape
    """

    # 1-dimensional
    if len(pad_shape) <= 1:
        list_pad = index_list.copy() + \
            (pad_shape[0] - len(index_list)) * [pad_index]
        return np.asarray(list_pad)

    # pad sub-list
    arrays = []
    for item in index_list:
        array = pad_list_shape(item, pad_shape[1:], pad_index)
        arrays.append(array)

    # pad first dim
    n_subs = len(arrays)
    if n_subs < pad_shape[0]:
        for i in range(pad_shape[0] - n_subs):
            array_zero = np.zeros(pad_shape[1:], dtype=np.int)
            arrays.append(array_zero)

    # concat in first dim
    list_pad = np.stack(arrays, axis=0)

    return list_pad


def pad_list(index_list, pad_index=0):
    """Pad index list.

    Returns
    -------
    index_pad: np.ndarray, shape [batch_size, max_len_1, max_len_2, ...]
    """

    # not list, return directly
    if not isinstance(index_list, (list, tuple)):
        return index_list

    # pad shape
    pad_shape = get_pad_shape(index_list)

    # pad list
    index_pad = pad_list_shape(index_list, pad_shape, pad_index)

    return index_pad


def flatten_list(nested_list):
    """Flatten nested list, if not list, return [x].

    Examples
    --------
    >>> x = [1, [2, 3]]
    >>> x_flat = list(flatten_list(x))
    >>> x_flat
    [1, 2, 3]
    """

    if not isinstance(nested_list, (list, tuple)):
        yield nested_list
    else:
        for item in nested_list:
            yield from flatten_list(item)
