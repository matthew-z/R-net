"""Embedding each word using word-level and char-level embedding"""

import torch
import torchtext
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader


class GRU(torch.nn.Module):
    """ used to train char-level embedding"""

    def __init__(self, input_size, vocab_size, hidden_size, num_layers=1, bidirectional=True, cell_type="lstm"):
        super().__init__()
        self.input_layer = torch.nn.Linear(input_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size * 2 if bidirectional else hidden_size, vocab_size)

        cell_type = cell_type.lower()
        if cell_type == "gru":
            network = torch.nn.GRU
        elif cell_type == "lstm":
            network = torch.nn.LSTM
        else:
            raise ValueError("invalid cell type %s" % cell_type)

        self.network = network(input_size=input_size, hidden_size=hidden_size,
                               num_layers=num_layers, bidirectional=bidirectional)

    def forward(self, pack):
        outputs, hidden = self.network(pack)
        outputs, lengths = pad_packed_sequence(outputs)
        outputs = self.output_layer(outputs)
        outputs = pack_padded_sequence(outputs, lengths)
        return outputs, hidden


def collate(examples):
    embed_size = examples[0][0].size(1)
    PAD = 0.0
    examples.sort(key=lambda x: len(x[0]), reverse=True)
    seqs, words = zip(*examples)

    lengths = [len(s) for s in seqs]
    batch_length = max(lengths)
    seq_tensor = torch.FloatTensor(batch_length, len(seqs), embed_size).fill_(PAD)

    for i, s in enumerate(seqs):
        end_seq = lengths[i]
        seq_tensor[:end_seq, i, :].copy_(s[:end_seq])
    return pack_padded_sequence(Variable(seq_tensor, requires_grad=False), lengths), words


def build_padded_target(target_words, char_dict):
    max_len = max(len(x) for x in target_words)
    target = torch.zeros(max_len, len(target_words))

    for i, word in enumerate(target_words):
        chars_without_init = torch.LongTensor([char_dict[c] for c in word[1:]] + [0])
        target[:len(chars_without_init), i].copy_(chars_without_init)

    return Variable(target.unsqueeze(2))


def build_target(target_words, char_dict):
    target = []
    for i, word in enumerate(target_words):
        # will ignore 999 in CrossEntropyLoss
        chars_without_init = [char_dict.get(c, 0) for c in word[1:]] + [999]
        target.extend(chars_without_init)
    return Variable(torch.LongTensor(target).view(-1))


def train_gru(gru, dataloader, epoch, char_dict, char_vectors, char_embed_size, hidden_size, num_layers=1,
              bidirectional=True):
    """
    How to train the weights in GRU to encode words using char embedding?
    """
    optimizer = torch.optim.Adam(gru.parameters())

    for _ in range(epoch):
        for data in dataloader:
            pack, words = data
            outputs, hidden = gru(pack)

            target = build_target(words, char_dict)

            cost_fn = torch.nn.CrossEntropyLoss(ignore_index=999)

            loss = cost_fn(outputs.data.view(-1, len(char_dict)), target)

            gru.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.data[0])


def embedding_char_level(word_dict, char_dict, char_vectors, char_embed_size, embed_size=300, num_layers=1,
                         bidirectional=True, cache_path=None):
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

    if bidirectional:
        assert embed_size % 2 == 0, "embed_size must be an even number because it is bi-directional "
        hidden_size = embed_size // 2
    else:
        hidden_size = embed_size

    gru = GRU(input_size=char_embed_size, vocab_size=len(char_dict), hidden_size=hidden_size,
              num_layers=num_layers, bidirectional=bidirectional)

    train_gru(gru, dataloader, 50, char_dict, char_vectors, char_embed_size,
              hidden_size, num_layers=num_layers, bidirectional=bidirectional)

    char_level_embedding = building_embedding_from_gru(gru, dataloader, embed_size, word_dict)

    return char_level_embedding


def building_embedding_from_gru(gru, dataloader, embed_size, word_dict):
    """Build word embedding from a trained GRU"""
    char_level_embedding = torch.zeros(len(word_dict), embed_size)
    for data in dataloader:
        pack, words = data
        outputs, (hidden, _) = gru(pack)
        hidden = hidden.data.permute(1, 0, 2).contiguous().view(len(words), -1)

        for i, word in enumerate(words):
            word_id = word_dict[word]
            char_level_embedding[word_id].copy_(hidden[i])
    return char_level_embedding


def get_dataloader(char_dict, char_vectors, words):
    """
    Load words
    """
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
        numerlized_words.append((torch.stack(numerlized_word), word))

    dataloader = DataLoader(numerlized_words, batch_size=128, num_workers=4,
                            collate_fn=collate, shuffle=True)
    return dataloader


if __name__ == "__main__":
    word_dict, _, _ = torchtext.vocab.load_word_vectors("./data/embedding/word", "glove.6B",
                                                        50)
    char_dict, char_vec, char_size = torchtext.vocab.load_word_vectors("./data/embedding/word", "glove_char.840B",
                                                                       300)
    embedding_char_level(word_dict, char_dict, char_vec, 300, 300)
