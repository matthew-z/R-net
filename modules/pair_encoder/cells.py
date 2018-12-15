import torch
from torch import nn
from torch import Tensor
from allennlp.nn.util import masked_softmax

class _PairEncodeCell(nn.Module):
    def __init__(self, input_size, cell, attention_size, memory_size=None, use_state_in_attention=True, batch_first=False):
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
            nn.Linear(attention_size, 1, bias=False))

        self.batch_first = batch_first

    def forward(self, inputs: Tensor, memory: Tensor = None, memory_mask: Tensor = None, state: Tensor = None):
        """
        :param inputs:  B x H
        :param memory: T x B x H if not batch_first
        :param memory_mask: T x B if not batch_first
        :param state: B x H
        :return:
        """
        if self.batch_first:
            memory = memory.transpose(0, 1)
            memory_mask = memory_mask.transpose(0, 1)

        assert inputs.size(0) == memory.size(1) == memory_mask.size(
            1), "inputs batch size does not match memory batch size"

        memory_time_length = memory.size(0)

        if state is None:
            state = inputs.new_zeros(inputs.size(0), self.cell.hidden_size, requires_grad=False)

        if self.use_state:
            hx = state
            if isinstance(state, tuple):
                hx = state[0]
            attention_input = torch.cat([inputs, hx], dim=-1)
            attention_input = attention_input.unsqueeze(0).expand(memory_time_length, -1, -1)  # T B H
        else:
            attention_input = inputs.unsqueeze(0).expand(memory_time_length, -1, -1)

        attention_logits = self.attention_w(torch.cat([attention_input, memory], dim=-1)).squeeze(-1)

        attention_scores = masked_softmax(attention_logits, memory_mask, dim=0)

        attention_vector = torch.sum(attention_scores.unsqueeze(-1) * memory, dim=0)

        new_input = torch.cat([inputs, attention_vector], dim=-1)

        return self.cell(new_input, state)


class PairEncodeCell(_PairEncodeCell):
    def __init__(self, input_size, cell, attention_size,
                 memory_size=None, batch_first=False):
        super().__init__(
            input_size, cell, attention_size,
            memory_size=memory_size, use_state_in_attention=True, batch_first=batch_first)


class SelfMatchCell(_PairEncodeCell):
    def __init__(self, input_size, cell, attention_size,
                 memory_size=None, batch_first=False):
        super().__init__(
            input_size, cell, attention_size,
            memory_size=memory_size, use_state_in_attention=False, batch_first=batch_first)
