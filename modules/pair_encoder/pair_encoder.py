from overrides import overrides
from torch import nn

from modules.dropout import RNNDropout
from modules.gate import Gate
from modules.pair_encoder.attentions import bidirectional_unroll_attention_cell, unroll_attention_cell, \
    StaticDotAttention
from modules.pair_encoder.cells import PairEncodeCell, SelfMatchCell
from allennlp.common import Registrable

from modules.utils import get_rnn
from allennlp.modules import Seq2SeqEncoder


class AttentionEncoder(nn.Module, Registrable):
    def forward(self, inputs, inputs_mask, memory, memory_mask):
        raise NotImplementedError


class DynamicAttentionEncoder(AttentionEncoder):
    def __init__(self, memory_size, input_size, hidden_size, attention_size,
                 bidirectional, dropout, cell_factory=PairEncodeCell, batch_first=False):
        super().__init__()
        self.batch_first = batch_first
        cell_fn = lambda: cell_factory(input_size, cell=nn.GRUCell(input_size + memory_size, hidden_size),
                                       attention_size=attention_size, memory_size=memory_size,
                                       batch_first=batch_first)

        num_directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional

        self.cells = nn.ModuleList([cell_fn() for _ in range(num_directions)])
        self.dropout = RNNDropout(dropout)
        self.gate = Gate(input_size=hidden_size*2 if bidirectional else hidden_size, dropout=dropout)

    @overrides
    def forward(self, inputs, inputs_mask, memory, memory_mask):
        if self.bidirectional:
            cell_fw, cell_bw = self.cells
            output, _ = bidirectional_unroll_attention_cell(
                cell_fw, cell_bw, inputs, memory, memory_mask,
                batch_first=self.batch_first)
        else:
            cell, = self.cells
            output, _ = unroll_attention_cell(
                cell, inputs, memory, memory_mask, batch_first=self.batch_first)

        return self.gate(output)


@AttentionEncoder.register("dynamic_pair_encoder")
class DynamicPairEncoder(DynamicAttentionEncoder):
    def __init__(self, memory_size, input_size, hidden_size, attention_size,
                 bidirectional, dropout, batch_first=False):
        super().__init__(memory_size, input_size, hidden_size, attention_size,
                         bidirectional, dropout, PairEncodeCell, batch_first=batch_first)


@AttentionEncoder.register("dynamic_self_encoder")
class DynamicSelfEncoder(DynamicAttentionEncoder):
    def __init__(self, memory_size, input_size, hidden_size, attention_size,
                 bidirectional, dropout, batch_first=False):
        super().__init__(memory_size, input_size, hidden_size, attention_size,
                         bidirectional, dropout, SelfMatchCell, batch_first=batch_first)




@AttentionEncoder.register("pass_through")
class PassThrough(AttentionEncoder):
    def __init__(self, memory_size, input_size, hidden_size):
        super().__init__()


    @overrides
    def forward(self, inputs, inputs_mask, memory, memory_mask):
        return inputs



@AttentionEncoder.register("static_pair_encoder")
class StaticPairEncoder(AttentionEncoder):
    def __init__(self, memory_size, input_size, hidden_size, attention_size, bidirectional, dropout,
                 attention_factory=StaticDotAttention, rnn_type="GRU", batch_first=True):
        super().__init__()
        rnn_fn = Seq2SeqEncoder.by_name(rnn_type.lower())

        self.attention = attention_factory(memory_size, input_size, attention_size,
                                           dropout=dropout, batch_first=batch_first)

        self.gate = nn.Sequential(
            Gate(input_size + memory_size, dropout=dropout),
            RNNDropout(dropout, batch_first=batch_first)
        )

        self.encoder = rnn_fn(input_size=memory_size + input_size, hidden_size=hidden_size,
                              bidirectional=bidirectional, batch_first=batch_first)

    @overrides
    def forward(self, inputs, inputs_mask, memory, memory_mask):
        """
        Memory: T B H
        input: T B H
        """
        new_inputs = self.gate(self.attention(inputs, memory, memory_mask))
        outputs = self.encoder(new_inputs, inputs_mask)
        return outputs


@AttentionEncoder.register("static_self_encoder")
class StaticSelfMatchEncoder(StaticPairEncoder):
    pass
