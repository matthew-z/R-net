import torch
from torch import nn

from modules.attention import AttentionPooling
from modules.pair_encoder import PairEncodeCell, bidirectional_unroll_cell, unroll_cell, SelfMatchCell
from modules.recurrent import StackedCell


class _Encoder(nn.Module):
    def __init__(self, memory_size, input_size, hidden_size, attention_size,
                 bidirectional, dropout, cell_factory):
        super().__init__()

        cell_fn = lambda: cell_factory(input_size, cell=nn.GRUCell(input_size+memory_size, hidden_size),
                                       attention_size=attention_size, memory_size=memory_size)

        num_directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional

        self.cells = nn.ModuleList([cell_fn() for _ in range(num_directions)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, memory, memory_mask, input):

        for cell in self.cells:
            cell.feed_memory(memory, memory_mask)

        if self.bidirectional:
            cell_fw, cell_bw = self.cells
            output, _ = bidirectional_unroll_cell(cell_fw, cell_bw, input, batch_first=True)

        else:
            cell, = self.cells
            output, _ = unroll_cell(cell, input, batch_first=True)

        return self.dropout(output)



class PairEncoder(_Encoder):
    def __init__(self, memory_size, input_size, hidden_size, attention_size,
                 bidirectional, dropout):
        super().__init__(memory_size, input_size, hidden_size, attention_size,
                         bidirectional, dropout, PairEncodeCell)


class SelfMatchEncoder(_Encoder):
    def __init__(self, memory_size, input_size, hidden_size, attention_size,
                 bidirectional, dropout):
        super().__init__(memory_size, input_size, hidden_size, attention_size,
                         bidirectional, dropout, SelfMatchCell)


class Max(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        max_result, _ = input.max(dim=self.dim)
        return max_result


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CharLevelWordEmbeddingCnn(nn.Module):
    def __init__(self,
                 char_embedding_size,
                 char_num,
                 num_filters: int,
                 ngram_filter_sizes=(5,),
                 output_dim=None,
                 activation=nn.ReLU,
                 embedding_weights=None,
                 padding_idx=1,
                 requires_grad=True
                 ):
        super().__init__()

        if embedding_weights is not None:
            char_num, char_embedding_size = embedding_weights.size()

        self.embed_char = nn.Embedding(char_num, char_embedding_size, padding_idx=padding_idx)
        if embedding_weights is not None:
            self.embed_char.weight = nn.Parameter(embedding_weights)
            self.embed_char.weight.requires_grad = requires_grad

        self._embedding_size = char_embedding_size

        self.cnn_layers = nn.ModuleList()
        for i, filter_size in enumerate(ngram_filter_sizes):
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=char_embedding_size,
                              out_channels=num_filters,
                              kernel_size=filter_size),
                    activation(),
                    Max(2)
                )
            )

        conv_out_size = len(ngram_filter_sizes) * num_filters
        self.output_layer = nn.Linear(conv_out_size, output_dim) if output_dim else None
        self.output_dim = output_dim if output_dim else conv_out_size

    def forward(self, tensor, mask=None):
        """

        :param tensor:    batch x word_num x char_num
        :return:
        """
        if mask is not None:
            tensor = tensor * mask.float()

        batch_num, word_num, char_num = tensor.size()
        tensor = self.embed_char(tensor.view(-1, char_num)).transpose(1, 2)
        # import ipdb; ipdb.set_trace()
        conv_out = [layer(tensor) for layer in self.cnn_layers]
        output = torch.cat(conv_out, dim=1) if len(conv_out) > 1 else conv_out[0]

        if self.output_layer:
            output = self.output_layer(output)

        return output.view(batch_num, word_num, -1)


class WordEmbedding(nn.Module):
    """
    Embed word with word-level embedding and char-level embedding
    """

    def __init__(self, word_embedding, padding_idx=1, requires_grad=False):
        super().__init__()

        word_vocab_size, word_embedding_dim = word_embedding.size()
        self.word_embedding_word_level = nn.Embedding(word_vocab_size, word_embedding_dim,
                                                      padding_idx=padding_idx)
        self.word_embedding_word_level.weight = nn.Parameter(word_embedding,
                                                             requires_grad=requires_grad)
        # self.embedding_size = word_embedding_dim + char_embedding_config["output_dim"]
        self.output_dim = word_embedding_dim

    def forward(self, *tensors):
        """
        :param words: all distinct words (char level) in seqs. Tuple of word tensors (batch first, with lengths) and word idx
        :param tensors: lists of documents like: (contexts_tensor, contexts_tensor_new), context_lengths)
        :return:   embedded batch (batch first)
        """

        result = []
        for tensor in tensors:
            word_level_embed = self.word_embedding_word_level(tensor)
            result.append(word_level_embed)
        return result


class SentenceEncoding(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, dropout):
        super().__init__()

        self.question_encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout)

        self.passage_encoder = nn.GRU(
            input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout)

    def forward(self, question, passage):
        question_outputs, _ = self.question_encoder(question)
        passage_outputs, _ = self.passage_encoder(passage)
        return question_outputs, passage_outputs


class PointerNetwork(nn.Module):
    def __init__(self, question_size, passage_size, attn_size=None,
                 cell_type=nn.GRUCell, num_layers=1, dropout=0, residual=False, batch_first=True):
        super().__init__()
        self.num_layers = num_layers
        if attn_size is None:
            attn_size = question_size

        self.batch_first = batch_first

        # TODO: what is V_q? (section 3.4)
        v_q_size = question_size
        self.question_pooling = AttentionPooling(question_size,
                                                 v_q_size, attn_size=attn_size)
        self.passage_pooling = AttentionPooling(passage_size,
                                                question_size, attn_size=attn_size)

        self.V_q = nn.Parameter(torch.randn(1, 1, v_q_size), requires_grad=True)
        self.cell = StackedCell(question_size, question_size, num_layers=num_layers,
                                dropout=dropout, rnn_cell=cell_type, residual=residual)

    def forward(self, question, question_mask, passage, passage_mask):
        if self.batch_first:
            question = question.transpose(0, 1)
            question_mask = question_mask.transpose(0, 1)
            passage = passage.transpose(0, 1)
            passage_mask = passage_mask.transpose(0, 1)

        hidden = self.question_pooling(question, self.V_q,
                                       key_mask=question_mask, broadcast_key=True)  # 1 x batch x n
        inputs, ans_begin = self.passage_pooling(passage, hidden,
                                                 key_mask=passage_mask, return_key_scores=True)
        hidden = hidden.expand([self.num_layers, hidden.size(1), hidden.size(2)])

        output, hidden = self.cell(inputs.squeeze(0), hidden)
        _, ans_end = self.passage_pooling(passage, output.unsqueeze(0),
                                          key_mask=passage_mask, return_key_scores=True)

        ans_begin = ans_begin.transpose(0, 1)
        ans_end = ans_end.transpose(0, 1)
        return ans_begin, ans_end
