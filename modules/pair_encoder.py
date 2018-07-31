from torch import nn
import torch
from torch.nn import functional as F

class _PairEncodeCell(nn.Module):
    def __init__(self, input_size, cell, attention_size, memory_size=None, use_state_in_attention=True):
        super().__init__()
        if memory_size is None:
            memory_size = input_size

        self.cell = cell
        self.use_state = use_state_in_attention

        attention_input_size = input_size + memory_size
        if use_state_in_attention:
            attention_size += cell.hidden_size

        self.attention_w = nn.Sequential(
            nn.Linear(attention_input_size, attention_size, bias=False),
            nn.Tanh())
        self.attention_v = nn.Linear(attention_size, 1, bias=False)

        self.gate = nn.Sequential(nn.Linear(input_size + memory_size, input_size + memory_size, bias=False),
                                  nn.Sigmoid())
        self.memory = None

    def feed_memory(self, memory, memory_mask):
        self.memory = memory
        self.memory_mask = memory_mask.float().unsqueeze(-1)

    def forward(self, input, state):
        memory = self.memory
        if memory is None:
            raise AttributeError("memory has not been set")

        if self.use_state:
            hx = state
            if isinstance(state, tuple):
                hx = state[0]
            attention_input = torch.cat([input, hx], dim=-1).unsqueeze(1).expand(-1, self.memory.size(1), -1)
        else:
            attention_input = input.unsqueeze(1).expand(-1, self.memory.size(1), -1)
        attention_logits = self.attention_v(self.attention_w(torch.cat([attention_input, memory], dim=-1)))

        attention_scores = F.softmax(attention_logits * self.memory_mask)
        attention_vector = torch.sum(attention_scores * memory, dim=1)

        new_input = torch.cat([input, attention_vector], dim=-1)
        new_input = self.gate(new_input) * new_input

        return self.cell(new_input, state)


class PairEncodeCell(_PairEncodeCell):
    def __init__(self, input_size, cell, attention_size, memory_size=None):
        super().__init__(input_size, cell, attention_size, memory_size=memory_size, use_state_in_attention=True)


class SelfMatchCell(_PairEncodeCell):
    def __init__(self, input_size, cell, attention_size, memory_size=None):
        super().__init__(input_size, cell, attention_size, memory_size=memory_size, use_state_in_attention=False)


def unroll_cell(cell, input, batch_first=False, initial_state=None):
    if batch_first:
        input = input.transpose(0, 1)
    output = []
    state = initial_state
    for t, input_t in enumerate(input):
        o, state = cell(input_t, state)
        output.append(o)
    output = torch.stack(output, dim=1 if batch_first else 0)
    return output, state


def bidirectional_unroll_cell(cell_fw, cell_bw, input, batch_first=False, initial_state=None):
    if initial_state is None:
        initial_state = [None, None]

    output_fw, state_fw = unroll_cell(cell_fw, input, batch_first=batch_first, initial_state=initial_state[0])
    output_bw, state_bw = unroll_cell(cell_bw, input, batch_first=batch_first, initial_state=initial_state[0])

    return torch.cat([output_fw, output_bw], dim=-1), (state_fw, state_bw)

if __name__ == "__main__":
    x = torch.rand(32, 50)
    memory = torch.rand(32, 15, 50)
    decoder_cell = PairEncodeCell(50, nn.GRUCell(100, 50), 75)
    decoder_cell.feed_memory(memory)
    decoder_cell(x, torch.zeros(32, 50))
