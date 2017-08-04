import torch
from torch import nn

from embedding import CharLevelEmbedding
from recurrent import RNN
from main import test_DoubleEmbedding


class PairedEncoding(nn.Module):
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

        word_embedded_in_char = self.word_embedding_char_level(words.words_tensor)
        word_embedded_in_char_lookup = torch.nn.Embedding(word_embedded_in_char.size(0), word_embedded_in_char.size(1))
        word_embedded_in_char_lookup.weight.data.copy_(word_embedded_in_char.data)

        result = []
        for doc in documents_lists:
            word_level_embed = self.word_embedding_word_level(doc.tensor)
            char_level_embed = word_embedded_in_char_lookup(doc.tensor_new_dx)

            new_embed = torch.cat((word_level_embed, char_level_embed), dim=2)
            result.append(new_embed)

        return result


class SentenceEncoding(nn.Module):
    def __init__(self, input_size, sentence_encoding_config):
        super().__init__()


        self.question_encoder = RNN(input_size, sentence_encoding_config["hidden_size"],
                                     num_layers=sentence_encoding_config["num_layers"],
                                     bidirectional=sentence_encoding_config["bidirectional"],
                                     dropout=sentence_encoding_config["dropout"])

        self.context_encoder = RNN(input_size, sentence_encoding_config["hidden_size"],
                                     num_layers=sentence_encoding_config["num_layers"],
                                     bidirectional=sentence_encoding_config["bidirectional"],
                                     dropout=sentence_encoding_config["dropout"])

    def forward(self, question, context):



        question_outputs, _ = self.question_encoder(packed_question)
        context_outputs, _ = self.context_encoder(packed_context)

        return question_outputs, context_outputs

class RNet(nn.Module):
    def __init__(self, char_embedding_config, word_embedding_config, pair_encoding_config):
        super().__init__()
        self.embedding = WordEmbedding(char_embedding_config, word_embedding_config)
        self.sentence_encoding = SentenceEncoding()

    def forward(self, distinct_words, question, context, answer):

        # embed words using char-level and word-level and concat them
        embedded_question, embedded_context = self.embedding(distinct_words, question, context)







if __name__ == "__main__":
    test_DoubleEmbedding()