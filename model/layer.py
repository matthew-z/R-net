import torch
from torch import nn
from torch.nn import functional as F


def softmax_mask(logits, mask, INF=1e12, dim=None):
    masked_logits = torch.where(mask, logits, torch.full_like(logits, -INF))
    score = F.softmax(masked_logits, dim=dim)
    return masked_logits, score


class Max(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        max_result, _ = input.max(dim=self.dim)
        return max_result


class Gate(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_size, input_size, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return input * self.gate(input)


class _PairEncodeCell(nn.Module):
    def __init__(self, input_size, cell, attention_size, memory_size=None, use_state_in_attention=True,
                 attention_dropout=0.2):
        super().__init__()
        if memory_size is None:
            memory_size = input_size

        self.cell = cell
        self.use_state = use_state_in_attention

        attention_input_size = input_size + memory_size
        if use_state_in_attention:
            attention_input_size += cell.hidden_size

        self.attention_w = nn.Sequential(
            nn.Linear(attention_input_size, attention_size, bias=False),
            nn.Dropout(attention_dropout),
            nn.Tanh(),
            nn.Linear(attention_size, 1, bias=False),
            nn.Dropout(attention_dropout),
        )

        self.gate = Gate(input_size + memory_size)

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
                 memory_size=None, attention_dropout=0.2):
        super().__init__(
            input_size, cell, attention_size,
            memory_size=memory_size, use_state_in_attention=True,
            attention_dropout=attention_dropout)


class SelfMatchCell(_PairEncodeCell):
    def __init__(self, input_size, cell, attention_size,
                 memory_size=None, attention_dropout=0.2):
        super().__init__(
            input_size, cell, attention_size,
            memory_size=memory_size, use_state_in_attention=False,
            attention_dropout=attention_dropout)


def unroll_attention_cell(cell, input, memory, memory_mask, batch_first=False, initial_state=None):
    if batch_first:
        input = input.transpose(0, 1)
    output = []
    state = initial_state
    for t, input_t in enumerate(input):
        state = cell(input_t, memory=memory, memory_mask=memory_mask, state=state)
        output.append(state)
    output = torch.stack(output, dim=1 if batch_first else 0)
    return output, state


def bidirectional_unroll_attention_cell(cell_fw, cell_bw, input, memory, memory_mask, batch_first=False,
                                        initial_state=None):
    if initial_state is None:
        initial_state = [None, None]

    output_fw, state_fw = unroll_attention_cell(
        cell_fw, input, memory, memory_mask,
        batch_first=batch_first,
        initial_state=initial_state[0])
    output_bw, state_bw = unroll_attention_cell(
        cell_bw, input, memory, memory_mask,
        batch_first=batch_first,
        initial_state=initial_state[1])

    return torch.cat([output_fw, output_bw], dim=-1), (state_fw, state_bw)
