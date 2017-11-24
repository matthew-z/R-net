import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.submodules import PairEncoder, SelfMatchingEncoder, CharLevelWordEmbeddingCnn, WordEmbedding, \
    SentenceEncoding, PointerNetwork, pack_residual
from modules.recurrent import AttentionEncoderCell
from utils.dataset import Documents, CharDocuments


class Model(nn.Module):
    def __init__(self,
                 args,
                 char_embedding_config,
                 word_embedding_config,
                 sentence_encoding_config,
                 pair_encoding_config,
                 self_matching_config,
                 pointer_config):
        super().__init__()
        self.word_embedding = WordEmbedding(
            word_embedding=word_embedding_config["embedding_weights"],
            padding_idx=word_embedding_config["padding_idx"],
            requires_grad=word_embedding_config["update"])

        self.char_embedding = CharLevelWordEmbeddingCnn(
            char_embedding_size=char_embedding_config["char_embedding_size"],
            char_num = char_embedding_config["char_num"],
            num_filters=char_embedding_config["num_filters"],
            ngram_filter_sizes=char_embedding_config["ngram_filter_sizes"],
            output_dim=char_embedding_config["output_dim"],
            activation=char_embedding_config["activation"],
            embedding_weights=char_embedding_config["embedding_weights"],
            padding_idx=char_embedding_config["padding_idx"],
            requires_grad=char_embedding_config["update"]
        )

        # we are going to concat the output of two embedding methods
        embedding_dim = self.word_embedding.output_dim + self.char_embedding.output_dim

        self.r_net = RNet(args,
                          embedding_dim,
                          sentence_encoding_config,
                          pair_encoding_config,
                          self_matching_config,
                          pointer_config)

        self.args = args

    def forward(self,
                question: Documents,
                question_char: CharDocuments,
                passage: Documents,
                passage_char: CharDocuments):

        # Embedding using Glove
        embedded_question, embedded_passage = self.word_embedding(question.tensor, passage.tensor)

        if torch.cuda.is_available():
            question_char = question_char.cuda(self.args.device_id)
            passage_char = passage_char.cuda(self.args.device_id)
            question = question.cuda(self.args.device_id)
            passage = passage.cuda(self.args.device_id)
            embedded_question = embedded_question.cuda(self.args.device_id)
            embedded_passage = embedded_passage.cuda(self.args.device_id)

        # char level embedding
        embedded_question_char = self.char_embedding(question_char.tensor)
        embedded_passage_char = self.char_embedding(passage_char.tensor)

        # concat word embedding and char level embedding
        embedded_passage_merged = torch.cat([embedded_passage, embedded_passage_char], dim=2)
        embedded_question_merged = torch.cat([embedded_question, embedded_question_char], dim=2)

        return self.r_net(question, passage, embedded_question_merged, embedded_passage_merged)

    def cuda(self, *args, **kwargs):
        self.r_net.cuda(*args, **kwargs)
        self.char_embedding.cuda(*args, **kwargs)
        return self


class RNet(nn.Module):
    def __init__(self, args, embedding_size,
                 sentence_encoding_config,
                 pair_encoding_config, self_matching_config, pointer_config):
        super().__init__()
        self.current_score = 0
        self.sentence_encoding = SentenceEncoding(embedding_size,
                                                  sentence_encoding_config["hidden_size"],
                                                  sentence_encoding_config["num_layers"],
                                                  sentence_encoding_config["bidirectional"],
                                                  sentence_encoding_config["dropout"])

        sentence_encoding_direction = (2 if sentence_encoding_config["bidirectional"] else 1)
        sentence_encoding_size = (sentence_encoding_config["hidden_size"] * sentence_encoding_direction)
        self.pair_encoder = PairEncoder(sentence_encoding_size,
                                        sentence_encoding_size,
                                        hidden_size=pair_encoding_config["hidden_size"],
                                        bidirectional=pair_encoding_config["bidirectional"],
                                        mode=pair_encoding_config["mode"],
                                        num_layers=pair_encoding_config["num_layers"],
                                        dropout=pair_encoding_config["dropout"],
                                        residual=pair_encoding_config["residual"],
                                        gated=pair_encoding_config["gated"],
                                        rnn_cell=pair_encoding_config["rnn_cell"],
                                        cell_factory=AttentionEncoderCell)

        pair_encoding_num_direction = (2 if pair_encoding_config["bidirectional"] else 1)
        self.self_matching_encoder = SelfMatchingEncoder(passage_embed_size=self_matching_config["hidden_size"] * pair_encoding_num_direction,
                                                         hidden_size=self_matching_config["hidden_size"],
                                                         bidirectional=self_matching_config["bidirectional"],
                                                         mode=self_matching_config["mode"],
                                                         num_layers=self_matching_config["num_layers"],
                                                         dropout=self_matching_config["dropout"],
                                                         residual=self_matching_config["residual"],
                                                         gated=self_matching_config["gated"],
                                                         rnn_cell=self_matching_config["rnn_cell"],
                                                         )

        question_size = sentence_encoding_direction * sentence_encoding_config["hidden_size"]
        passage_size = pair_encoding_num_direction * pair_encoding_config["hidden_size"]
        self.pointer_output = PointerNetwork(question_size,
                                             passage_size,
                                             num_layers=pointer_config["num_layers"],
                                             dropout=pointer_config["dropout"],
                                             cell_type=pointer_config["rnn_cell"])
        self.residual = args.residual
        for weight in self.parameters():
            if weight.ndimension() >= 2:
                nn.init.orthogonal(weight)

    def forward(self,
                question: Documents,
                passage: Documents,
                embedded_question,
                embedded_passage):
        # embed words using char-level and word-level and concat them
        passage_pack, question_pack = self._sentence_encoding(embedded_passage,
                                                              embedded_question,
                                                              passage, question)
        question_encoded_padded_sorted, _ = pad_packed_sequence(question_pack)  # (seq, batch, encode_size), lengths
        question_encoded_padded_original = question.restore_original_order(question_encoded_padded_sorted,
                                                                           batch_dim=1)
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

    def _sentence_encoding(self,
                           embedded_passage,
                           embedded_question,
                           passage,
                           question):
        question_pack = pack_padded_sequence(embedded_question, question.lengths, batch_first=True)
        passage_pack = pack_padded_sequence(embedded_passage, passage.lengths, batch_first=True)
        question_encoded_pack, passage_encoded_pack = self.sentence_encoding(question_pack, passage_pack)
        return passage_encoded_pack, question_encoded_pack

    def _self_match_encode(self,
                           paired_passage_pack,
                           passage):
        passage_mask_sorted_order = passage.to_sorted_order(passage.mask_original, batch_dim=0).transpose(0, 1)
        self_matched_passage, _ = self.self_matching_encoder(pad_packed_sequence(paired_passage_pack)[0],
                                                             passage_mask_sorted_order, paired_passage_pack)
        return self_matched_passage

    def _pair_encode(self,
                     passage_encoded_pack,
                     question_encoded_padded_in_passage_sorted_order,
                     question_mask_in_passage_order):
        paired_passage_pack, _ = self.pair_encoder(question_encoded_padded_in_passage_sorted_order,
                                                   question_mask_in_passage_order,
                                                   passage_encoded_pack)
        return paired_passage_pack
