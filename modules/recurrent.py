import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence

from utils.utils import get_rnn


class RNN(nn.Module):
    """ RNN Module """

    def __init__(self, input_size, hidden_size,
                 output_projection_size=None, num_layers=1,
                 bidirectional=True, cell_type="lstm", dropout=0,
                 pack=False, batch_first=False, init_method="default"):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)

        if output_projection_size is not None:
            self.output_layer = nn.Linear(hidden_size * 2 if bidirectional else hidden_size,
                                          output_projection_size)
        self.pack = pack
        network = get_rnn(cell_type)
        self.network = network(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, bidirectional=bidirectional,
                               dropout=dropout, batch_first=batch_first)


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
            hidden_i = select_layer(hidden, i)
            next_hidden_i = layer(inputs, hidden_i)
            output = next_hidden_i[0] if isinstance(next_hidden_i, tuple) \
                else next_hidden_i
            if i + 1 != self.num_layers:
                output = self.dropout(output)
            if i > 0 and self.residual:
                inputs = output + inputs
            else:
                inputs = output
            next_hidden.append(next_hidden_i)
        if isinstance(hidden, tuple):
            next_hidden = tuple([torch.stack(h) for h in zip(*next_hidden)])
        else:
            next_hidden = torch.stack(next_hidden)
        return inputs, next_hidden


class AttentionEncoderCell(StackedCell):
    def __init__(self, question_embed_size, passage_embed_size, hidden_size,
                 attention_layer_factory, attn_args, attn_kwags, attn_mode="pair_encoding", num_layers=1,
                 dropout=0, bias=True, rnn_cell=nn.GRUCell, residual=False,
                 gated=True):
        input_size = question_embed_size + passage_embed_size
        super().__init__(input_size, hidden_size, num_layers,
                         dropout, bias, rnn_cell, residual)
        self.attention = attention_layer_factory(*attn_args, **attn_kwags)
        self.gated = gated
        self.attn_mode = attn_mode
        if gated:
            self.gate = nn.Sequential(
                nn.Linear(input_size, input_size, bias=False),
                nn.Sigmoid()
            )

    def forward(self, input_with_context_context_mask, hidden):
        inputs, context, context_mask = input_with_context_context_mask

        if isinstance(hidden, tuple):
            hidden_for_attention = hidden[0]
        else:
            hidden_for_attention = hidden
        hidden_for_attention = hidden_for_attention[0:1]
        #
        key = context
        if self.attn_mode == "pair_encoding":
            queries = [inputs, hidden_for_attention]
        elif self.attn_mode == "self_matching":
            queries = [inputs]
        else:
            raise ValueError("invalid attention_mode %s" % self.attn_mode)

        context = self.attention(key, queries, key_mask=context_mask)
        inputs = torch.cat([context, inputs], dim=context.dim()-1)
        if self.gated:
            inputs = inputs * self.gate(inputs)
        return super().forward(inputs.squeeze(0), hidden)


class AttentionEncoder(nn.Module):
    def __init__(self, cell_factory, *args, bidirectional=False, mode="GRU", **kwargs):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size = args[2]
        self.forward_cell = cell_factory(*args, **kwargs)
        self.num_layers = self.forward_cell.num_layers
        self.mode=mode
        if bidirectional:
            self.reversed_cell = cell_factory(*args, **kwargs)

    def _forward(self, inputs, hidden):
        pack, context, context_mask = inputs
        output = []
        input_offset = 0
        hiddens = []
        last_batch_size = pack.batch_sizes[0]
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden, )

        for batch_size in pack.batch_sizes:
            step_input = pack.data[input_offset:input_offset+batch_size].unsqueeze(0)
            input_offset += batch_size
            dec = last_batch_size - batch_size
            if dec > 0:
                hiddens.append(tuple(h[:, -dec:, ] for h in hidden))
                hidden = tuple(h[:, :-dec,] for h in hidden)
            last_batch_size = batch_size

            if flat_hidden:
                hidden = (self.forward_cell((step_input, context, context_mask), hidden[0])[1],)
            else:
                hidden = self.forward_cell((step_input, context, context_mask), hidden)[1]

            output.append(hidden[0][0])

        hiddens.append(hidden)
        hiddens.reverse()
        hidden = tuple(torch.cat(h, 1) for h in zip(*hiddens))
        assert hidden[0].size(1) == pack.batch_sizes[0]
        if flat_hidden:
            hidden = hidden[0]
        output = torch.cat(output, 0)
        output= PackedSequence(output, pack.batch_sizes)
        return output, hidden

    def _reversed_forward(self, inputs, hidden):
        pack, context, context_mask = inputs

        batch_sizes = pack.batch_sizes
        output = []
        input_offset = pack.data.size(0)
        last_batch_size = batch_sizes[-1]
        initial_hidden = hidden
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
            initial_hidden = (initial_hidden,)
        hidden = tuple(h[:, :batch_sizes[-1]] for h in hidden)
        for batch_size in reversed(batch_sizes):
            inc = batch_size - last_batch_size
            if inc > 0:
                hidden = tuple(torch.cat((h, ih[:,last_batch_size:batch_size]), 1)
                               for h, ih in zip(hidden, initial_hidden))
            last_batch_size = batch_size
            step_input = pack.data[input_offset - batch_size:input_offset].unsqueeze(0)
            input_offset -= batch_size

            if flat_hidden:
                hidden = (self.reversed_cell((step_input, context, context_mask), hidden[0])[1],)
            else:
                hidden = self.reversed_cell((step_input, context, context_mask), hidden)[1]
            output.append(hidden[0][0])

        output.reverse()
        output = torch.cat(output, 0)
        output = PackedSequence(output, batch_sizes)

        if flat_hidden:
            hidden = hidden[0]
        return output, hidden

    def forward(self, inputs, hidden=None):
        pack, context, context_mask = inputs
        if hidden is None:
            max_batch_size = pack.batch_sizes[0]
            hidden = torch.autograd.Variable(torch.zeros(self.num_layers,
                                                       max_batch_size,
                                                       self.hidden_size))
            if torch.cuda.is_available():
                hidden = hidden.cuda()

            if self.mode == 'LSTM':
                hidden = (hidden, hidden)

        output_forward, hidden_forward = self._forward(inputs, hidden)
        if not self.bidirectional:
            return output_forward, hidden_forward
        output_reversed, hidden_reversed = self._reversed_forward(inputs, hidden)

        # concat forward and reversed forward
        hidden = torch.cat([hidden_forward, hidden_reversed], dim=hidden_forward.dim() - 1)
        output_data = torch.cat([output_forward.data, output_reversed.data],
                                dim=output_reversed.data.dim() - 1)

        output = PackedSequence(output_data, output_forward.batch_sizes)
        assert output.data.size(0) == pack.data.size(0)
        return output, hidden
