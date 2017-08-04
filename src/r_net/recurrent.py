import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from utils import get_rnn


class RNN(torch.nn.Module):
    """ RNN Module """

    def __init__(self, input_size, hidden_size, output_projection_size=None, num_layers=1,
                 bidirectional=True, cell_type="lstm", dropout=0, pack=False, batch_first=False):
        super().__init__()
        self.input_layer = torch.nn.Linear(input_size, hidden_size)

        if output_projection_size is not None:
            self.output_layer = torch.nn.Linear(hidden_size * 2 if bidirectional else hidden_size,
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
