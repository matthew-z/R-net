import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch import nn
from utils import get_rnn
from torch.autograd import Variable
from attention import PairAttentionLayer

class RNN(nn.Module):
    """ RNN Module """

    def __init__(self, input_size, hidden_size, output_projection_size=None, num_layers=1,
                 bidirectional=True, cell_type="lstm", dropout=0, pack=False, batch_first=False):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)

        if output_projection_size is not None:
            self.output_layer = nn.Linear(hidden_size * 2 if bidirectional else hidden_size,
                                                output_projection_size)
        self.pack = pack
        network = get_rnn(cell_type)
        self.network = network(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, bidirectional=bidirectional, dropout=dropout,
                               batch_first=batch_first)

    def forward(self, input_variable):
        outputs, hidden = self.network(input_variable)
        if self.pack:
            padded_outputs, lengths = pad_packed_sequence(outputs)
            if hasattr(self, "output_layer"):
                outputs = pack_padded_sequence(self.output_layer(padded_outputs), lengths)

        elif hasattr(self, "output_layer"):
                outputs = self.output_layer(outputs)

        return outputs, hidden


class StackedCell(nn.Module):
    """
    From https://github.com/eladhoffer/seq2seq.pytorch
    MIT License  Copyright (c) 2017 Elad Hoffer
    """
    def __init__(self, input_size, hidden_size, num_layers=1,
                 dropout=0, bias=True, rnn_cell=nn.GRUCell, residual=False):
        super(StackedCell, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.residual = residual
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(rnn_cell(input_size, hidden_size, bias=bias))
            input_size = hidden_size

    def forward(self, inputs, hidden):
        def select_layer(h_state, i):  # To work on both LSTM / GRU, RNN
            if isinstance(h_state, tuple):
                return tuple([select_layer(s, i) for s in h_state])
            else:
                return h_state[i]

        next_hidden = []
        for i, layer in enumerate(self.layers):
            next_hidden_i = layer(inputs, select_layer(hidden, i))
            output = next_hidden_i[0] if isinstance(next_hidden_i, tuple) \
                else next_hidden_i
            if i + 1 != self.num_layers:
                output = self.dropout(output)
            if self.residual:
                inputs = output + inputs
            else:
                inputs = output
            next_hidden.append(next_hidden_i)
        if isinstance(hidden, tuple):
            next_hidden = tuple([torch.stack(h) for h in zip(*next_hidden)])
        else:
            next_hidden = torch.stack(next_hidden)
        return inputs, next_hidden


class GatedAttentionCellForPairedEncoding(StackedCell):
    def __init__(self, input_size, hidden_size, context_size, attention_layer, num_layers=1,
                 dropout=0, bias=True, rnn_cell=nn.GRUCell, residual=False):

        super().__init__(input_size, hidden_size, num_layers=1,
                 dropout=0, bias=True, rnn_cell=nn.GRUCell, residual=False)

        self.context_size = context_size
        self.attention = attention_layer
        self.context_size = context_size
        self.gate = nn.Sequential(nn.Linear(context_size + input_size, hidden_size, bias=False),
                                  nn.Sigmoid())

    def forward(self, input_with_context_context_mask, hidden):
        inputs, context, context_mask = input_with_context_context_mask
        context = self.attention(context, inputs, hidden, context_mask)
        inputs = self.gate(torch.cat([context, inputs], dim=2))
        return super().forward(inputs, hidden)











#
# class GatedAttentionCellForPairedEncoding(nn.Module):
#     def __init__(self, input_size, hidden_size, context_size, attention_layer, num_layers=1,
#                  dropout=0, bias=True, rnn_cell=nn.GRUCell, residual=False):
#
#         super().__init__()
#         self.cell = StackedCell(input_size, hidden_size, num_layers,
#                                 dropout, bias, rnn_cell, residual)
#         self.attention = attention_layer
#         self.context_size = context_size
#         self.gate = nn.Sequential(nn.Linear(context_size + input_size, hidden_size, bias=False),
#                                   nn.Sigmoid())
#
#     def forward(self, input_with_context_context_mask, hidden, get_attention=False):
#         input, context, context_mask = input_with_context_context_mask
#         context = self.attention(context, input, hidden, context_mask)
#         input = self.gate(torch.cat([context, input], dim=2))
#         return self.cell(input, hidden)

