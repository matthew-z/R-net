import torch
from torch.nn import functional as F



def reverse_padded_sequence_fast(inputs, lengths, batch_first=False):
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (Variable): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A Variable with the same size as inputs, but with each sequence
        reversed according to its length.
    """
    if not batch_first:
        inputs = inputs.transpose(0, 1)
    if inputs.size(0) != len(lengths):
        raise ValueError('inputs incompatible with lengths.')
    reversed_indices = [list(range(inputs.size(1))) for _ in range(inputs.size(0))]
    for i, length in enumerate(lengths):
        if length > 0:
            reversed_indices[i][:length] = reversed_indices[i][length-1::-1]
    reversed_indices = torch.LongTensor(reversed_indices).unsqueeze(2).expand_as(inputs)
    if inputs.is_cuda:
        reversed_indices = reversed_indices.cuda()
    reversed_inputs = torch.gather(inputs, 1, reversed_indices)
    if not batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs


def softmax_mask(logits, mask, INF=1e12, dim=None):
    masked_logits = torch.where(mask, logits, torch.full_like(logits, -INF))
    score = F.softmax(masked_logits, dim=dim)
    return masked_logits, score

def get_rnn(rnn_type):
    return getattr(torch.nn, rnn_type)