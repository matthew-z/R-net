import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from modules.attention import AttentionPooling
from modules.recurrent import RNN, AttentionEncoder, AttentionEncoderCell, StackedCell
from utils.dataset import Documents


def pack_residual(x_pack, y_pack):
    x_tensor, x_lengths = pad_packed_sequence(x_pack)
    y_tensor, y_lengths = pad_packed_sequence(y_pack)

    if x_lengths != y_lengths:
        raise ValueError("different lengths")

    return pack_padded_sequence(x_tensor + y_tensor, x_lengths)

class PairEncoder(nn.Module):
    def __init__(self, question_embed_size, passage_embed_size, config,
                 cell_factory=AttentionEncoderCell):
        super().__init__()

        attn_args = [question_embed_size, passage_embed_size, config["hidden_size"]]
        attn_kwags = {"attn_size": 75, "batch_first": False}

        self.pair_encoder = AttentionEncoder(
            cell_factory, question_embed_size, passage_embed_size,
            config["hidden_size"], AttentionPooling, attn_args, attn_kwags,
            bidirectional=config["bidirectional"], mode=config["mode"],
            num_layers=config["num_layers"], dropout=config["dropout"],
            residual=config["residual"], gated=config["gated"],
            rnn_cell=config["rnn_cell"], attn_mode="pair_encoding"
        )

    def forward(self, questions, question_mark, passage):
        inputs = (passage, questions, question_mark)
        result = self.pair_encoder(inputs)
        return result


class SelfMatchingEncoder(nn.Module):
    def __init__(self, passage_embed_size, config,
                 cell_factory=AttentionEncoderCell):
        super().__init__()
        attn_args = [passage_embed_size, passage_embed_size]
        attn_kwags = {"attn_size": 75, "batch_first": False}

        self.pair_encoder = AttentionEncoder(
            cell_factory, passage_embed_size, passage_embed_size,
            config["hidden_size"], AttentionPooling, attn_args, attn_kwags,
            bidirectional=config["bidirectional"], mode=config["mode"],
            num_layers=config["num_layers"], dropout=config["dropout"],
            residual=config["residual"], gated=config["gated"],
            rnn_cell=config["rnn_cell"], attn_mode="self_matching"
        )

    def forward(self, questions, question_mark, passage):
        inputs = (passage, questions, question_mark)
        return self.pair_encoder(inputs)


class WordEmbedding(nn.Module):
    """
    Embed word with word-level embedding and char-level embedding
    """

    def __init__(self, char_embedding_config, word_embedding_config):
        super().__init__()

        word_embedding = word_embedding_config["embedding_weights"]
        word_vocab_size, word_embedding_dim = word_embedding.size()
        self.word_embedding_word_level = nn.Embedding(word_vocab_size, word_embedding_dim,
                                                      padding_idx=word_embedding_config["padding_idx"])
        self.word_embedding_word_level.weight = nn.Parameter(word_embedding,
                                                             requires_grad=word_embedding_config["update"])
        # self.embedding_size = word_embedding_dim + char_embedding_config["output_dim"]
        self.embedding_size = word_embedding_dim

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
    def __init__(self, input_size, config):
        super().__init__()

        self.question_encoder = RNN(input_size, config["hidden_size"],
                                    num_layers=config["num_layers"],
                                    bidirectional=config["bidirectional"],
                                    dropout=config["dropout"])

        self.passage_encoder = RNN(input_size, config["hidden_size"],
                                   num_layers=config["num_layers"],
                                   bidirectional=config["bidirectional"],
                                   dropout=config["dropout"])


    def forward(self, question_pack, context_pack):
        question_outputs, _ = self.question_encoder(question_pack)
        passage_outputs, _ = self.passage_encoder(context_pack)
        return question_outputs, passage_outputs


class PointerNetwork(nn.Module):
    def __init__(self, question_size, passage_size, hidden_size, attn_size=None,
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

        hidden = hidden.expand([self.num_layers, hidden.size(1), hidden.size(2)])

        inputs, ans_begin = self.passage_pooling(passage_pad, hidden,
                                                 key_mask=passage_mask, return_key_scores=True)
        _, hidden = self.cell(inputs.squeeze(0), hidden)
        _, ans_end = self.passage_pooling(passage_pad, hidden,
                                          key_mask=passage_mask, return_key_scores=True)

        return ans_begin, ans_end


class Model(nn.Module):
    def __init__(self, args, char_embedding_config, word_embedding_config, sentence_encoding_config,
                 pair_encoding_config, self_matching_config, pointer_config):
        super().__init__()
        self.embedding = WordEmbedding(char_embedding_config, word_embedding_config)

        self.r_net = RNet(args, self.embedding.embedding_size, char_embedding_config, word_embedding_config,
                          sentence_encoding_config,
                          pair_encoding_config, self_matching_config, pointer_config)

        self.device_id = args.device_id

    def forward(self, question: Documents, passage: Documents):
        embedded_question, embedded_passage = self.embedding(question.tensor, passage.tensor)

        if torch.cuda.is_available():
            embedded_question = embedded_question.cuda(self.device_id)
            embedded_passage = embedded_passage.cuda(self.device_id)
            question = question.cuda(self.device_id)
            passage = passage.cuda(self.device_id)

        return self.r_net(question, passage, embedded_question, embedded_passage)

    def cuda(self, *args, **kwargs):
        self.r_net.cuda(*args, **kwargs)
        return self


class RNet(nn.Module):
    def __init__(self, args, embedding_size, char_embedding_config, word_embedding_config, sentence_encoding_config,
                 pair_encoding_config, self_matching_config, pointer_config):
        super().__init__()
        self.current_score = 0
        self.sentence_encoding = SentenceEncoding(embedding_size, sentence_encoding_config)

        sentence_num_direction = (2 if sentence_encoding_config["bidirectional"] else 1)
        sentence_encoding_size = (sentence_encoding_config["hidden_size"]
                                  * sentence_num_direction)
        self.pair_encoder = PairEncoder(sentence_encoding_size,
                                        sentence_encoding_size,
                                        pair_encoding_config)

        self_match_num_direction = (2 if pair_encoding_config["bidirectional"] else 1)
        self.self_matching_encoder = SelfMatchingEncoder(pair_encoding_config["hidden_size"] * self_match_num_direction,
                                                         self_matching_config)

        question_size = sentence_num_direction * sentence_encoding_config["hidden_size"]
        passage_size = self_match_num_direction * pair_encoding_config["hidden_size"]
        self.pointer_output = PointerNetwork(question_size, passage_size, pointer_config["hidden_size"])
        self.residual = args.residual
        for name, weight in self.named_parameters():
            if weight.ndimension() >= 2:
                nn.init.orthogonal(weight)


    def forward(self, question: Documents, passage: Documents, embedded_question, embedded_passage):
        # embed words using char-level and word-level and concat them
        passage_pack, question_pack = self._sentence_encoding(embedded_passage, embedded_question,
                                                              passage, question)
        question_encoded_padded_sorted, _ = pad_packed_sequence(question_pack)  # (seq, batch, encode_size), lengths
        question_encoded_padded_original = question.restore_original_order(question_encoded_padded_sorted, batch_dim=1)
        # question and context has same ordering
        question_pad_in_passage_sorted_order = passage.to_sorted_order(question_encoded_padded_original,
                                                                       batch_dim=1)
        question_mask_in_passage_sorted_order = passage.to_sorted_order(question.mask_original, batch_dim=0).transpose(
            0, 1)

        paired_passage_pack = self._pair_encode(passage_pack, question_pad_in_passage_sorted_order,
                                                question_mask_in_passage_sorted_order)

        if self.residual:
            paired_passage_pack = pack_residual(paired_passage_pack, passage_pack)

        self_matched_passage_pack = self._self_match_encode(paired_passage_pack, passage)

        if self.residual:
            self_matched_passage_pack = pack_residual(paired_passage_pack, self_matched_passage_pack)

        begin, end = self.pointer_output(question_pad_in_passage_sorted_order,
                                         question_mask_in_passage_sorted_order,
                                         pad_packed_sequence(self_matched_passage_pack)[0],
                                         passage.to_sorted_order(passage.mask_original, batch_dim=0).transpose(0, 1))

        return (passage.restore_original_order(begin.transpose(0, 1), 0),
                passage.restore_original_order(end.transpose(0, 1), 0))

    def _sentence_encoding(self, embedded_passage, embedded_question, passage, question):
        question_pack = pack_padded_sequence(embedded_question, question.lengths, batch_first=True)
        passage_pack = pack_padded_sequence(embedded_passage, passage.lengths, batch_first=True)
        question_encoded_pack, passage_encoded_pack = self.sentence_encoding(question_pack, passage_pack)
        return passage_encoded_pack, question_encoded_pack

    def _self_match_encode(self, paired_passage_pack, passage):
        passage_mask_sorted_order = passage.to_sorted_order(passage.mask_original, batch_dim=0).transpose(0, 1)
        self_matched_passage, _ = self.self_matching_encoder(pad_packed_sequence(paired_passage_pack)[0],
                                                             passage_mask_sorted_order, paired_passage_pack)
        return self_matched_passage

    def _pair_encode(self, passage_encoded_pack, question_encoded_padded_in_passage_sorted_order,
                     question_mask_in_passage_order):
        paired_passage_pack, _ = self.pair_encoder(question_encoded_padded_in_passage_sorted_order,
                                                   question_mask_in_passage_order,
                                                   passage_encoded_pack)
        return paired_passage_pack
