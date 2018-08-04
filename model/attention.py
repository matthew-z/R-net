import torch
from torch import nn
from model import layer
from model.layer import RNNDropout


class StaticAddAttention(nn.Module):
    def __init__(self, memory_size, input_size, attention_size, dropout=0.2):
        super().__init__()

        self.attention_w = nn.Sequential(
            nn.Linear(input_size + memory_size, attention_size, bias=False),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(attention_size, 1, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, input, memory, memory_mask):
        T = input.size(0)
        memory_mask = memory_mask.unsqueeze(0)
        memory_key = memory.unsqueeze(0).expand(T, -1, -1, -1)
        input_key = input.unsqueeze(1).expand(-1, T, -1, -1)
        attention_logits = self.attention_w(torch.cat([input_key, memory_key], -1)).squeeze(-1)
        logits, score = layer.softmax_mask(attention_logits, memory_mask, dim=1)
        context = torch.sum(score.unsqueeze(-1) * input.unsqueeze(0), dim=1)
        new_input = torch.cat([context, input], dim=-1)
        return new_input


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
            nn.Linear(input_size, attention_size, bias=False),
            nn.ReLU()
        )
        self.attention_size = attention_size
        self.batch_first = batch_first

    def forward(self, input, memory, memory_mask):
        if not self.batch_first:
            input = input.transpose(0, 1)
            memory = memory.transpose(0, 1)
            mask = memory_mask.transpose(0, 1)

        input_ = self.input_linear(input)
        memory_ = self.input_linear(memory)
        logits = torch.bmm(input_, memory_.transpose(2, 1)) / (self.attention_size ** 0.5)
        mask = mask.unsqueeze(1).expand(-1, input.size(1), -1)
        logits, score = layer.softmax_mask(logits, mask, dim=-1)

        context = torch.bmm(score, memory)
        new_input = torch.cat([context, input], dim=-1)

        if not self.batch_first:
            return new_input.transpose(0, 1)
        return new_input
