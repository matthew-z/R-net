"""Embedding each word using word-level and char-level embedding"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from torch.utils.data import DataLoader
from torch.autograd import Variable

def collate(examples):
    embed_size = examples[0][0].size(1)
    PAD = 0.0
    examples.sort(key=lambda x:len(x[0]), reverse=True)

    seqs, words = zip(*examples)

    lengths = [len(s) for s in seqs]
    batch_length = max(lengths)

    seq_tensor = torch.FloatTensor(batch_length, len(seqs), embed_size).fill_(PAD)

    for i, s in enumerate(seqs):
        end_seq = lengths[i]
        seq_tensor[:end_seq, i, :].copy_(s[:end_seq])
    return pack_padded_sequence(Variable(seq_tensor), lengths), words

def embedding_char_level(word_dict, char_dict, char_vectors, char_embed_size, embed_size=300, num_layers=1, bidirectional=True, cache_path=None):
    """

    :param word_dict: word dict of data, does not contain special words
    :param char_dict: char dict of saved char embedding
    :param char_vectors: embedding vectors of char embedding
    :param char_embed_size: embedding size of char embedding
    :param embed_size: size of char-level word embedding
    :param num_layers:  num of layers of rnn which used in encoding words
    :param bidirectional: bidirectional property of rnn which used in encoding words
    :param cache_path: cache path of encoded words

    :return: word_embedding (encoded by char_level), same ordering as word_dict
    """
    dataloader = get_dataloader(char_dict, char_vectors, word_dict.keys())

    assert embed_size % 2 == 0, "embed_size must be an even number because it is bi-directional "

    gru = torch.nn.GRU(input_size=char_embed_size, hidden_size=embed_size//2,
                       num_layers=num_layers, bidirectional=bidirectional)

    char_level_embedding = torch.zeros(len(word_dict), embed_size)

    words_total = []
    current = 0

    for data in dataloader:
        pack, words = data
        outputs, hidden = gru(pack)
        hidden.view(len(words), embed_size), words
        batch_size = len(words)
        char_level_embedding[current:current+batch_size, :].copy_(hidden)
        words_total.extend(words)

    # TODO: char_level_embedding should have same ordering as word_dict

    return words_total, char_level_embedding



def get_dataloader(char_dict, char_vectors, words):
    sorted_words = sorted(words, key=len)
    embed_size = len(char_vectors[0])
    PAD = torch.zeros(embed_size)
    numerlized_words = []
    for word in sorted_words:
        numerlized_word = []
        for char in word:
            char_id = char_dict.get(char, -1)
            if char_id < 0:
                numerlized_word.append(PAD)
            else:
                numerlized_word.append(char_vectors[char_id])
        numerlized_words.append((torch.Tensor(numerlized_word), word))

    dataloader = DataLoader(numerlized_words, batch_size=3, num_workers=1,
                            collate_fn=collate)
    return dataloader


if __name__ == "__main__":
    words = ["hello", "bye",  "yoooooooooo", "hello", "bye",  "yoooooooooo"]

    char_dict = {"h":0, "e":1, "o":2, "b":3, "l":4, "y":5}
    char_vec = [[0.1, 0.2, 0.3],[0.3, 0.22, 0.33],[0.14, 0.21, 0.33],
                [0.15, 0.21, 0.33],[0.155, 0.223, 0.331],[0.1512, 0.123, 0.123],[0.1512, 0.123, 0.123] ]

    embedding_char_level(words, char_dict, char_vec, 3,3)
