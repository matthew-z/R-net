import torch
from allennlp.modules import Seq2SeqEncoder

from modules.dropout import RNNDropout


@Seq2SeqEncoder.register("concat_rnn")
class ConcatRNN(Seq2SeqEncoder):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, rnn_type="GRU", dropout=0., stateful=False, batch_first=True):
        super().__init__(stateful=stateful)
        rnn_cls = Seq2SeqEncoder.by_name(rnn_type.lower())
        self.rnn_list = torch.nn.ModuleList(
            [rnn_cls(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional, dropout=dropout)])

        for _ in range(num_layers - 1):
            self.rnn_list.append(
                rnn_cls(input_size=hidden_size * 2, hidden_size=hidden_size, bidirectional=bidirectional,
                        dropout=dropout))

        self.dropout = RNNDropout(dropout, batch_first=batch_first)

    def forward(self, inputs, mask, hidden=None):
        outputs_list = []

        for layer in self.rnn_list:
            outputs = layer(self.dropout(inputs), mask, hidden)
            outputs_list.append(outputs)
            inputs = outputs

        return torch.cat(outputs_list, -1)