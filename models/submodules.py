import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from modules.attention import AttentionPooling
from modules.recurrent import AttentionEncoderCell, AttentionEncoder, RNN, StackedCell


class PairEncoder(nn.Module):
    def __init__(self, question_embed_size, passage_embed_size, hidden_size,
                 bidirectional, mode, num_layers, dropout,
                 residual, gated, rnn_cell,
                 cell_factory=AttentionEncoderCell):
        super().__init__()

        attn_args = [question_embed_size, passage_embed_size, hidden_size]
        attn_kwargs = {"attn_size": 75, "batch_first": False}

        self.pair_encoder = AttentionEncoder(
            cell_factory,
            question_embed_size,
            passage_embed_size,
            hidden_size,
            AttentionPooling,
            attn_args,
            attn_kwargs,
            bidirectional=bidirectional,
            mode=mode,
            num_layers=num_layers,
            dropout=dropout,
            residual=residual,
            gated=gated,
            rnn_cell=rnn_cell,
            attn_mode="pair_encoding"
        )

    def forward(self, questions, question_mark, passage):
        inputs = (passage, questions, question_mark)
        result = self.pair_encoder(inputs)
        return result


class SelfMatchingEncoder(nn.Module):
    def __init__(self, passage_embed_size, hidden_size,
                 bidirectional, mode, num_layers,
                 dropout, residual, gated,
                 rnn_cell,
                 cell_factory=AttentionEncoderCell):
        super().__init__()
        attn_args = [passage_embed_size, passage_embed_size]
        attn_kwargs = {"attn_size": 75, "batch_first": False}

        self.pair_encoder = AttentionEncoder(
            cell_factory, passage_embed_size, passage_embed_size,
            hidden_size,
            AttentionPooling,
            attn_args,
            attn_kwargs,
            bidirectional=bidirectional,
            mode=mode,
            num_layers=num_layers,
            dropout=dropout,
            residual=residual,
            gated=gated,
            rnn_cell=rnn_cell,
            attn_mode="self_matching"
        )

    def forward(self, questions, question_mark, passage):
        inputs = (passage, questions, question_mark)
        return self.pair_encoder(inputs)


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
        #import ipdb; ipdb.set_trace()
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

        self.question_encoder = RNN(input_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    bidirectional=bidirectional,
                                    dropout=dropout)

        self.passage_encoder = RNN(input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bidirectional=bidirectional,
                                   dropout=dropout)

    def forward(self, question_pack, context_pack):
        question_outputs, _ = self.question_encoder(question_pack)
        passage_outputs, _ = self.passage_encoder(context_pack)
        return question_outputs, passage_outputs


class PointerNetwork(nn.Module):
    def __init__(self, question_size, passage_size, attn_size=None,
                 cell_type=nn.GRUCell, num_layers=1, dropout=0, residual=False, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        if attn_size is None:
            attn_size = question_size

        # TODO: what is V_q? (section 3.4)
        v_q_size = question_size
        self.question_pooling = AttentionPooling(question_size,
                                                 v_q_size, attn_size=attn_size)
        self.passage_pooling = AttentionPooling(passage_size,
                                                question_size, attn_size=attn_size)

        self.V_q = nn.Parameter(torch.randn(1, 1, v_q_size), requires_grad=True)
        self.cell = StackedCell(question_size, question_size, num_layers=num_layers,
                                dropout=dropout, rnn_cell=cell_type, residual=residual, **kwargs)

    def forward(self, question_pad, question_mask, passage_pad, passage_mask):
        hidden = self.question_pooling(question_pad, self.V_q,
                                       key_mask=question_mask, broadcast_key=True)  # 1 x batch x n

        inputs, ans_begin = self.passage_pooling(passage_pad, hidden,
                                                 key_mask=passage_mask, return_key_scores=True)

        hidden = hidden.expand([self.num_layers, hidden.size(1), hidden.size(2)])

        output, hidden = self.cell(inputs.squeeze(0), hidden)
        _, ans_end = self.passage_pooling(passage_pad, output.unsqueeze(0),
                                          key_mask=passage_mask, return_key_scores=True)

        return ans_begin, ans_end


def pack_residual(x_pack, y_pack):
    x_tensor, x_lengths = pad_packed_sequence(x_pack)
    y_tensor, y_lengths = pad_packed_sequence(y_pack)

    if x_lengths != y_lengths:
        raise ValueError("different lengths")

    return pack_padded_sequence(x_tensor + y_tensor, x_lengths)