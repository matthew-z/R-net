import torch
from torch import nn

from modules import attention
from modules.pair_encoder import PairEncodeCell, bidirectional_unroll_attention_cell, unroll_attention_cell
from torch.nn import functional as F


class _Encoder(nn.Module):
    def __init__(self, memory_size, input_size, hidden_size, attention_size,
                 bidirectional, dropout, cell_factory):
        super().__init__()

        cell_fn = lambda: cell_factory(input_size, cell=nn.GRUCell(input_size + memory_size, hidden_size),
                                       attention_size=attention_size, memory_size=memory_size)

        num_directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional

        self.cells = nn.ModuleList([cell_fn() for _ in range(num_directions)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, memory, memory_mask, input, batch_first=False):

        if self.bidirectional:
            cell_fw, cell_bw = self.cells
            output, _ = bidirectional_unroll_attention_cell(
                cell_fw, cell_bw, input, memory, memory_mask,
                batch_first=batch_first)

        else:
            cell, = self.cells
            output, _ = unroll_attention_cell(
                cell, input, memory, memory_mask, batch_first=batch_first)

        return self.dropout(output)


class PairEncoder(_Encoder):
    def __init__(self, memory_size, input_size, hidden_size, attention_size,
                 bidirectional, dropout):
        super().__init__(memory_size, input_size, hidden_size, attention_size,
                         bidirectional, dropout, PairEncodeCell)


class SelfMatchEncoder(nn.Module):
    def __init__(self, memory_size, input_size, hidden_size, attention_size,
                 bidirectional, dropout, attention_factory=attention.StaticDotAttention):
        super().__init__()
        self.attention = attention_factory(memory_size, input_size, attention_size, dropout=dropout)
        self.gate = nn.Sequential(
            nn.Linear(input_size + memory_size, input_size + memory_size, bias=False),
            nn.Sigmoid())
        self.rnn = nn.GRU(input_size=(input_size), hidden_size=hidden_size,
                          bidirectional=bidirectional, dropout=dropout)

    def forward(self, input, memory, memory_mask):
        """
        Memory: T B H
        input: T B H
        """

        new_input = self.attention(input, memory, memory_mask)
        new_input = self.gate(new_input)
        output, _ = self.rnn(new_input)

        return output


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
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, dropout, batch_first=False):
        super().__init__()

        self.question_encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=batch_first)

        self.passage_encoder = nn.GRU(
            input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=batch_first)

    def forward(self, question, passage):
        question_outputs, _ = self.question_encoder(question)
        passage_outputs, _ = self.passage_encoder(passage)
        return question_outputs, passage_outputs


def softmax_mask(logits, mask, INF=1e12, dim=None):
    masked_logits = torch.where(mask, logits, torch.full_like(logits, -INF))
    score = F.softmax(masked_logits, dim=dim)
    return masked_logits, score


class OutputLayer(nn.Module):
    def __init__(self, question_size, passage_size, attention_size=75,
                 cell_type=nn.GRUCell, dropout=0, batch_first=False):
        super().__init__()

        self.batch_first = batch_first

        # TODO: what is V_q? (section 3.4)
        v_q_size = question_size

        self.cell = cell_type(question_size, question_size)
        self.dropout = dropout

        self.passage_linear = nn.Sequential(
            nn.Linear(question_size + passage_size, attention_size, bias=False),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(attention_size, 1, bias=False),
            nn.Dropout(dropout))

        self.question_linear = nn.Sequential(
            nn.Linear(question_size + v_q_size, attention_size, bias=False),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(attention_size, 1, bias=False),
            nn.Dropout(dropout),
        )

        self.V_q = nn.Parameter(torch.randn(1, 1, v_q_size), requires_grad=True)

    def forward(self, question, question_mask, passage, passage_mask):
        """
        :param question: T B H
        :param question_mask: T B
        :param passage:
        :param passage_mask:
        :return:
        """

        if self.batch_first:
            question = question.transpose(0, 1)
            question_mask = question_mask.transpose(0, 1)
            passage = passage.transpose(0, 1)
            passage_mask = passage_mask.transpose(0, 1)

        state = self._question_pooling(question, question_mask)
        cell_input, ans_start_logits = self._passage_attention(passage, passage_mask, state)
        state = self.cell(cell_input, hx=state)
        _, ans_end_logits = self._passage_attention(passage, passage_mask, state)

        return ans_start_logits.transpose(0, 1), ans_end_logits.transpose(0, 1)

    def _question_pooling(self, question, question_mask):
        V_q = self.V_q.expand(question.size(0), question.size(1), -1)
        logits = self.question_linear(torch.cat([question, V_q], dim=-1)).squeeze(-1)
        logits, score = softmax_mask(logits, question_mask, dim=0)
        state = torch.sum(score.unsqueeze(-1) * question, dim=0)
        return state

    def _passage_attention(self, passage, passage_mask, state):
        state_expand = state.unsqueeze(0).expand(passage.size(0), passage.size(1), -1)
        logits = self.passage_linear(torch.cat([passage, state_expand], dim=-1)).squeeze(-1)
        logits, score = softmax_mask(logits, passage_mask, dim=0)
        cell_input = torch.sum(score.unsqueeze(-1) * passage, dim=0)
        return cell_input, logits
