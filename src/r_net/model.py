import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from attention import PairAttentionLayer, AttentionPooling
from dataset import Documents, Words
from embedding import CharLevelEmbedding
from recurrent import RNN, AttentionEncoder, AttentionEncoderCell


class PairEncoder(nn.Module):
    def __init__(self, question_embed_size, passage_embed_size, config,
                 cell_factory=AttentionEncoderCell):
        super().__init__()
        self.pair_encoder = AttentionEncoder(
            cell_factory, question_embed_size, passage_embed_size,
            config["hidden_size"], PairAttentionLayer,
            bidirectional=config["bidirectional"], mode=config["mode"], attn_size=config["attn_size"],
            num_layers=config["num_layers"], dropout=config["dropout"],
            residual=config["residual"], gated=config["gated"],
            rnn_cell=config["rnn_cell"]
        )

    def forward(self, questions, question_mark, passage):
        inputs = (passage, questions, question_mark)
        return self.pair_encoder(inputs)


class SelfMatchingEncoder(nn.Module):
    def __init__(self, question_embed_size, passage_embed_size, config,
                 cell_factory=AttentionEncoderCell):
        super().__init__()
        self.pair_encoder = AttentionEncoder(
            cell_factory, question_embed_size, passage_embed_size,
            config["hidden_size"], AttentionPooling,
            bidirectional=config["bidirectional"], mode=config["mode"], attn_size=config["attn_size"],
            num_layers=config["num_layers"], dropout=config["dropout"],
            residual=config["residual"], gated=config["gated"],
            rnn_cell=config["rnn_cell"]
        )

    def forward(self, questions, question_mark, passage):
        inputs = (passage, questions, question_mark)
        return self.pair_encoder(inputs)


class PointerNetwork(nn.Module):
    pass


class WordEmbedding(nn.Module):
    """
    Embed word with word-level embedding and char-level embedding
    """

    def __init__(self, char_embedding_config, word_embedding_config):
        super().__init__()
        # set char level word embedding
        char_embedding = char_embedding_config["embedding_weights"]

        char_vocab_size, char_embedding_dim = char_embedding.size()
        self.word_embedding_char_level = CharLevelEmbedding(char_vocab_size, char_embedding, char_embedding_dim,
                                                            char_embedding_config["output_dim"],
                                                            padding_idx=char_embedding_config["padding_idx"],
                                                            bidirectional=char_embedding_config["bidirectional"],
                                                            cell_type=char_embedding_config["cell_type"])

        # set word level word embedding

        word_embedding = word_embedding_config["embedding_weights"]
        word_vocab_size, word_embedding_dim = word_embedding.size()
        self.word_embedding_word_level = nn.Embedding(word_vocab_size, word_embedding_dim,
                                                      padding_idx=word_embedding_config["padding_idx"])
        self.word_embedding_word_level.weight = nn.Parameter(word_embedding,
                                                             requires_grad=word_embedding_config["update"])
        self.embedding_size = word_embedding_dim + char_embedding_config["output_dim"]

    def forward(self, words, *documents_lists):
        """

        :param words: all distinct words (char level) in seqs. Tuple of word tensors (batch first, with lengths) and word idx
        :param documents_lists: lists of documents like: (contexts_tensor, contexts_tensor_new), context_lengths)
        :return:   embedded batch (batch first)
        """

        word_embedded_in_char = self.word_embedding_char_level(words.words_tensor, words.words_lengths)

        result = []
        for doc in documents_lists:
            word_level_embed = self.word_embedding_word_level(doc.tensor)
            char_level_embed = (word_embedded_in_char.index_select(index=doc.tensor_new_dx.view(-1), dim=0)
                                .view(doc.tensor_new_dx.size(0), doc.tensor_new_dx.size(1), -1))
            new_embed = torch.cat((word_level_embed, char_level_embed), dim=2)
            result.append(new_embed)

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


class RNet(nn.Module):
    def __init__(self, char_embedding_config, word_embedding_config, sentence_encoding_config,
                 pair_encoding_config, self_matching_config, pointer_config):
        super().__init__()
        self.embedding = WordEmbedding(char_embedding_config, word_embedding_config)
        self.sentence_encoding = SentenceEncoding(self.embedding.embedding_size, sentence_encoding_config)

        num_direction = (2 if sentence_encoding_config["bidirectional"] else 1)
        sentence_encoding_size = (sentence_encoding_config["hidden_size"]
                                  * num_direction)
        self.pair_encoder = PairEncoder(sentence_encoding_size,
                                        sentence_encoding_size,
                                        pair_encoding_config)

        num_direction = (2 if pair_encoding_config["bidirectional"] else 1)
        self.self_matching_encoder = PairEncoder(pair_encoding_config["hidden_size"] * num_direction,
                                                 pair_encoding_config["hidden_size"] * num_direction,
                                                 self_matching_config)

    def forward(self, distinct_words: Words, question: Documents, passage: Documents):
        # embed words using char-level and word-level and concat them
        embedded_question, embedded_passage = self.embedding(distinct_words, question, passage)

        passage_pack, question_pack = self._sentence_encoding(embedded_passage, embedded_question,
                                                              passage, question)

        paired_passage_pack = self._pair_encode(passage, passage_pack, question, question_pack)

        self_matched_passage = self._self_match_encode(paired_passage_pack, passage)

        print(self_matched_passage)

    def _sentence_encoding(self, embedded_passage, embedded_question, passage, question):
        question_pack = pack_padded_sequence(embedded_question, question.lengths, batch_first=True)
        passage_pack = pack_padded_sequence(embedded_passage, passage.lengths, batch_first=True)
        question_encoded_pack, passage_encoded_pack = self.sentence_encoding(question_pack, passage_pack)
        return passage_encoded_pack, question_encoded_pack

    def _self_match_encode(self, paired_passage_pack, passage):
        passage_mask_sorted_order = passage.to_sorted_order(passage.mask_original, batch_dim=0).transpose(0, 1)
        self_matched_passage = self.self_matching_encoder(pad_packed_sequence(paired_passage_pack)[0],
                                                          passage_mask_sorted_order, paired_passage_pack)
        return self_matched_passage

    def _pair_encode(self, passage, passage_encoded_pack, question, question_encoded_pack):
        question_encoded_padded_sorted, _ = pad_packed_sequence(
            question_encoded_pack)  # (Seq, Batch, encode_size), lengths
        # passage_encoded_padded_sorted, _ = pad_packed_sequence(passage_encoded_pack)
        question_encoded_padded_original = question.restore_original_order(question_encoded_padded_sorted, batch_dim=1)
        # question and context has same ordering
        question_encoded_padded_in_passage_sorted_order = passage.to_sorted_order(question_encoded_padded_original,
                                                                                  batch_dim=1)
        question_mask_in_passage_order = passage.to_sorted_order(question.mask_original, batch_dim=0).transpose(0, 1)
        paired_passage_pack, _ = self.pair_encoder(question_encoded_padded_in_passage_sorted_order,
                                                   question_mask_in_passage_order,
                                                   passage_encoded_pack)
        return paired_passage_pack
