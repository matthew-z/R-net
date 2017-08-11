import torch
from torch import nn

from embedding import CharLevelEmbedding
from recurrent import RNN
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataset import Documents, Words

class PairedEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()


    def forward(self, question_padded, context_padded):
        pass


class SelfMatching(nn.Module):
    pass


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

        self.context_encoder = RNN(input_size, config["hidden_size"],
                                   num_layers=config["num_layers"],
                                   bidirectional=config["bidirectional"],
                                   dropout=config["dropout"])

    def forward(self, question_pack, context_pack):
        question_outputs, _ = self.question_encoder(question_pack)
        context_outputs, _ = self.context_encoder(context_pack)

        return question_outputs, context_outputs


class RNet(nn.Module):
    def __init__(self, char_embedding_config, word_embedding_config, sentence_encoding_config,
                 pair_encoding_config, self_matching_config, pointer_config):
        super().__init__()
        self.embedding = WordEmbedding(char_embedding_config, word_embedding_config)
        self.sentence_encoding = SentenceEncoding(self.embedding.embedding_size, sentence_encoding_config)

    def forward(self, distinct_words: Words, question: Documents , context: Documents):
        # embed words using char-level and word-level and concat them
        embedded_question, embedded_context = self.embedding(distinct_words, question, context)

        question_pack = pack_padded_sequence(embedded_question, question.lengths, batch_first=True)
        context_pack = pack_padded_sequence(embedded_context, context.lengths, batch_first=True)

        question_encoded_pack = self.sentence_encoding(question_pack, context_pack)
        context_encoded_pack = self.sentence_encoding(question_pack, context_pack)

        # (Seq, Batch, encode_size), lengths
        question_encoded_padded_sorted, _ = pad_packed_sequence(question_encoded_pack)
        context_encoded_padded_sorted, _ = pad_packed_sequence(context_encoded_pack)

        # restore unsorted order
        question_encoded_padded_original = question.restore_original_order(question_encoded_padded_sorted)
        context_encoded_padded_original = context.restore_original_order(context_encoded_padded_sorted)





