import torch
from torch import nn

from modules.dropout import RNNDropout
from modules.utils import softmax_mask
from torch import Tensor

def unroll_attention_cell(cell, inputs, memory, memory_mask, batch_first=False, initial_state=None, backward=False):
    if batch_first:
        inputs = inputs.transpose(0, 1)
    output = []
    state = initial_state
    steps = range(inputs.size(0))
    if backward:
        steps = range(inputs.size(0)-1, -1, -1)
    for t in steps:
        state = cell(inputs[t], memory=memory, memory_mask=memory_mask, state=state)
        output.append(state)
    if backward:
        output = output[::-1]
    output = torch.stack(output, dim=1 if batch_first else 0)
    return output, state


def bidirectional_unroll_attention_cell(cell_fw, cell_bw, inputs, memory, memory_mask, batch_first=False,
                                        initial_state=None):
    if initial_state is None:
        initial_state = [None, None]

    output_fw, state_fw = unroll_attention_cell(
        cell_fw, inputs, memory, memory_mask,
        batch_first=batch_first,
        initial_state=initial_state[0], backward=False)

    output_bw, state_bw = unroll_attention_cell(
        cell_bw, inputs, memory, memory_mask,
        batch_first=batch_first,
        initial_state=initial_state[1], backward=True)

    return torch.cat([output_fw, output_bw], dim=-1), (state_fw, state_bw)


# class StaticAddAttention(nn.Module):
#     def __init__(self, memory_size, input_size, attention_size, dropout=0.2, batch_first=False):
#         super().__init__()
#         self.batch_first = batch_first

#         self.attention_w = nn.Sequential(
#             nn.Linear(input_size + memory_size, attention_size, bias=False),
#             nn.Dropout(dropout),
#             nn.Tanh(),
#             nn.Linear(attention_size, 1, bias=False),
#             nn.Dropout(dropout),
#         )

#     def forward(self, inputs: Tensor, memory: Tensor, memory_mask: Tensor):
#         if not self.batch_first:
#             raise NotImplementedError

#         T = inputs.size(0)
#         memory_mask = memory_mask.unsqueeze(0)
#         memory_key = memory.unsqueeze(0).expand(T, -1, -1, -1)
#         input_key = inputs.unsqueeze(1).expand(-1, T, -1, -1)
#         attention_logits = self.attention_w(torch.cat([input_key, memory_key], -1)).squeeze(-1)
#         logits, score = softmax_mask(attention_logits, memory_mask, dim=1)
#         context = torch.sum(score.unsqueeze(-1) * inputs.unsqueeze(0), dim=1)
#         new_input = torch.cat([context, inputs], dim=-1)
#         return new_input


class StaticDotAttention(nn.Module):
    def __init__(self, memory_size, input_size, attention_size, batch_first=False, dropout=0.2):

        super().__init__()

        self.input_linear = nn.Sequential(
            RNNDropout(dropout, batch_first=True),
            nn.Linear(input_size, attention_size, bias=False),
            nn.ReLU()
        )

        self.memory_linear = nn.Sequential(
            RNNDropout(dropout, batch_first=True),
            nn.Linear(memory_size, attention_size, bias=False),
            nn.ReLU()
        )
        self.attention_size = attention_size
        self.batch_first = batch_first

    def forward(self, inputs: Tensor, memory: Tensor, memory_mask: Tensor):
        if not self.batch_first:
            inputs = inputs.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_mask = memory_mask.transpose(0, 1)

        input_ = self.input_linear(inputs)
        memory_ = self.memory_linear(memory)

        logits = torch.bmm(input_, memory_.transpose(2, 1)) / (self.attention_size ** 0.5)

        memory_mask = memory_mask.unsqueeze(1).expand(-1, inputs.size(1), -1)
        logits, score = softmax_mask(logits, memory_mask, dim=-1)

        context = torch.bmm(score, memory)
        new_input = torch.cat([context, inputs], dim=-1)

        if not self.batch_first:
            return new_input.transpose(0, 1)
        return new_input

