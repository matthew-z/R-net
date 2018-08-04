import torch
from torch import nn
from torch.nn import functional as F

def softmax_mask(logits, mask, INF=1e12, dim=None):
    masked_logits = torch.where(mask, logits, torch.full_like(logits, -INF))
    score = F.softmax(masked_logits, dim=dim)
    return masked_logits, score


class Max(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        max_result, _ = input.max(dim=self.dim)
        return max_result


class Gate(nn.Module):
    def __init__(self, input_size, dropout=0.2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, input_size, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return input * self.gate(input)


class _PairEncodeCell(nn.Module):
    def __init__(self, input_size, cell, attention_size, memory_size=None, use_state_in_attention=True,
                 dropout=0.2):
        super().__init__()
        if memory_size is None:
            memory_size = input_size

        self.cell = cell
        self.use_state = use_state_in_attention

        attention_input_size = input_size + memory_size
        if use_state_in_attention:
            attention_input_size += cell.hidden_size

        self.attention_w = nn.Sequential(
            nn.Dropout(),
            nn.Linear(attention_input_size, attention_size, bias=False),
            nn.Tanh(),
            nn.Linear(attention_size, 1, bias=False),
        )

        self.gate = Gate(input_size + memory_size, dropout=dropout)


    def forward(self, input, memory=None, memory_mask=None, state=None):
        """

        :param input:  B x H
        :param memory: T x B x H
        :param memory_mask: T x B
        :param state: B x H
        :return:
        """
        assert input.size(0) == memory.size(1) == memory_mask.size(
            1), "input batch size does not match memory batch size"

        memory_time_length = memory.size(0)

        if state is None:
            state = input.new_zeros(input.size(0), self.cell.hidden_size, requires_grad=False)

        if self.use_state:
            hx = state
            if isinstance(state, tuple):
                hx = state[0]
            attention_input = torch.cat([input, hx], dim=-1)
            attention_input = attention_input.unsqueeze(0).expand(memory_time_length, -1, -1)  # T B H
        else:
            attention_input = input.unsqueeze(0).expand(memory_time_length, -1, -1)

        attention_logits = self.attention_w(torch.cat([attention_input, memory], dim=-1)).squeeze(-1)
        # import ipdb; ipdb.set_trace()
        attention_scores = F.softmax(
            torch.where(memory_mask, attention_logits, torch.full_like(attention_logits, -1e12)), dim=0)
        attention_vector = torch.sum(attention_scores.unsqueeze(-1) * memory, dim=0)

        new_input = torch.cat([input, attention_vector], dim=-1)
        new_input = self.gate(new_input)

        return self.cell(new_input, state)


class PairEncodeCell(_PairEncodeCell):
    def __init__(self, input_size, cell, attention_size,
                 memory_size=None, dropout=0.2):
        super().__init__(
            input_size, cell, attention_size,
            memory_size=memory_size, use_state_in_attention=True,
            dropout=dropout)


class SelfMatchCell(_PairEncodeCell):
    def __init__(self, input_size, cell, attention_size,
                 memory_size=None, dropout=0.2):
        super().__init__(
            input_size, cell, attention_size,
            memory_size=memory_size, use_state_in_attention=False,
            dropout=dropout)


class RNNDropout(nn.Module):
    def __init__(self, p, batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.batch_first = batch_first

    def forward(self, input):

        if not self.training:
            return input
        if self.batch_first:
            mask = input.new_ones(input.size(0), 1, input.size(2), requires_grad=False)
        else:
            mask = input.new_ones(1, input.size(1), input.size(2), requires_grad=False)
        return self.dropout(mask) * input


def unroll_attention_cell(cell, input, memory, memory_mask, batch_first=False, initial_state=None, backward=False):
    if batch_first:
        input = input.transpose(0, 1)
    output = []
    state = initial_state
    steps = range(input.size(0))
    if backward:
        steps = range(input.size(0)-1, -1, -1)
    for t in steps:
        state = cell(input[t], memory=memory, memory_mask=memory_mask, state=state)
        output.append(state)
    if backward:
        output = output[::-1]
    output = torch.stack(output, dim=1 if batch_first else 0)
    return output, state


def bidirectional_unroll_attention_cell(cell_fw, cell_bw, input, memory, memory_mask, batch_first=False,
                                        initial_state=None):
    if initial_state is None:
        initial_state = [None, None]

    output_fw, state_fw = unroll_attention_cell(
        cell_fw, input, memory, memory_mask,
        batch_first=batch_first,
        initial_state=initial_state[0], backward=False)

    output_bw, state_bw = unroll_attention_cell(
        cell_bw, input, memory, memory_mask,
        batch_first=batch_first,
        initial_state=initial_state[1], backward=True)

    return torch.cat([output_fw, output_bw], dim=-1), (state_fw, state_bw)


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
