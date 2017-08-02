import os

import nltk
import spacy
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils import read_train_json, get_counter, read_dev_json, read_embedding


def padding(seqs, pad, batch_first=False):
    """

    :param seqs: tuple of seq_length x dim
    :return: seq_length x Batch x dim
    """
    lengths = [len(s) for s in seqs]
    batch_length = max(lengths)
    seq_tensor = torch.LongTensor(batch_length, len(seqs)).fill_(pad)
    for i, s in enumerate(seqs):
        end_seq = lengths[i]
        seq_tensor[:end_seq].copy_(s[:end_seq])
    if batch_first:
        seq_tensor = seq_tensor.t()

    return (seq_tensor, lengths)

class RawExample(object):
    pass


class SQuAD(Dataset):
    def __init__(self, path, word_embedding=None, char_embedding=None, embedding_cache_root=None, split="train",
                 tokenization="nltk", insert_start="<SOS>", insert_end="<EOS>",
                 debug_mode=True, debug_len=100, use_cuda=True):

        self.UNK = 0
        self.PAD = 1
        self.SOS = 2
        self.EOS = 3
        self.use_cuda = use_cuda

        if embedding_cache_root is None:
            embedding_cache_root = "./data/embedding"

        if word_embedding is None:
            word_embedding = (os.path.join(embedding_cache_root, "word"), "glove.840B", 300)

        # TODO: read trained char embedding in testing
        if char_embedding is None:
            char_embedding_root = os.path.join(embedding_cache_root, "char")
            char_embedding = (char_embedding_root, "glove_char.840B", 300)

        self.specials = ["<UNK>", "<PAD>"]
        if insert_start is not None:
            self.specials.append(insert_start)
        if insert_end is not None:
            self.specials.append(insert_end)

        self.insert_start = insert_start
        self.insert_end = insert_end
        self._set_tokenizer(tokenization)

        # read raw data from json
        if split == "train":
            self.examples_raw, context_with_counter = read_train_json(path, debug_mode, debug_len)
        else:
            self.examples_raw, context_with_counter = read_dev_json(path, debug_mode, debug_len)

        tokenized_context = [self.tokenizer(doc) for doc, count in context_with_counter]
        tokenized_question = [self.tokenizer(e.question) for e in self.examples_raw]

        # char/word counter
        word_counter, char_counter = get_counter(tokenized_context, tokenized_question)

        # build vocab
        self.itos, self.stoi, self.wv_vec = self._build_vocab(word_counter, word_embedding)
        self.itoc, self.ctoi, self.cv_vec = self._build_vocab(char_counter, char_embedding)

        # numeralize word level and char level
        self.numeralized_question = [self._numeralize_word_seq(question, self.stoi) for question in tokenized_question]
        self.numeralized_context = [self._numeralize_word_seq(context, self.stoi) for context in tokenized_context]

        self.numeralized_context_char = self._char_level_numeralize(tokenized_context)
        self.numeralized_question_char = self._char_level_numeralize(tokenized_question)

    def _char_level_numeralize(self, tokenized_docs):
        result = []
        for doc in tokenized_docs:
            words = []
            for word in doc:
                numeralized_word = self._numeralize_word_seq(word, self.ctoi)
                words.append(numeralized_word)
            result.append(words)
        return result

    def _numeralize_word_seq(self, seq, stoi, insert_sos=False, insert_eos=False):
        if self.insert_start is not None and insert_sos:
            result = [self.insert_start]
        else:
            result = []
        for word in seq:
            result.append(stoi.get(word, 0))
        if self.insert_end is not None and insert_eos:
            result.append(self.insert_end)
        return result

    def __getitem__(self, idx):
        question = self.numeralized_question[idx]
        question_char = self.numeralized_question_char[idx]

        context_id = self.examples_raw[idx].context_id
        context = self.numeralized_context[context_id]
        context_char = self.numeralized_context_char[context_id]

        answer_start = self.examples_raw[idx].answer_start
        answer_end = self.examples_raw[idx].answer_start

        return question, question_char, context, context_char, answer_start, answer_end

    def __len__(self):
        return len(self.examples_raw)

    def _build_vocab(self, counter, embedding_config):
        """
        :param counter: counter of words in dataset
        :param embedding_config: word_embedding config: (root, word_type, dim)
        :return: itos, stoi, vectors
        """

        wv_dict, wv_vectors, wv_size = read_embedding(embedding_config)

        # embedding size = glove vector size
        embed_size = wv_vectors.size(1)
        print("word embedding size: %d" % embed_size)

        # build itos and stoi
        # words_in_dataset = sorted(counter.keys(), key=lambda x: counter[x], reverse=True)
        words_in_dataset = counter.keys()

        itos = self.specials[:]
        stoi = {}

        itos.extend(words_in_dataset)
        for idx, word in enumerate(itos):
            stoi[word] = idx

        # build vectors
        vectors = torch.zeros([len(itos), embed_size])
        for word, idx in stoi.items():
            idx_in_pretrained_array = wv_dict.get(word, None)
            if idx_in_pretrained_array is not None:
                vectors[idx, :wv_size].copy_(wv_vectors[idx_in_pretrained_array])
        return itos, stoi, vectors

    def _set_tokenizer(self, tokenization):
        """
        Set tokenizer

        :param tokenization: tokenization method
        :return: None
        """
        if tokenization == "nltk":
            self.tokenizer = nltk.word_tokenize
        elif tokenization == "spacy":
            spacy_en = spacy.load("en")

            def spacy_tokenizer(seq):
                return [w.text for w in spacy_en(seq)]
            self.tokenizer = spacy_tokenizer
        else:
            raise ValueError("Invalid tokenizing method %s" % tokenization)

    def create_collate_fn(self, batch_first=False):


        def collate(examples):
            questions, char_questions, contexts, char_contexts, answer_starts, answer_ends = zip(*examples)



            #
            # embed_size = examples[0][0].size(1)
            # PAD = 0.0
            # examples.sort(key=lambda x: len(x[0]), reverse=True)
            # seqs, words = zip(*examples)
            #
            # lengths = [len(s) for s in seqs]
            # batch_length = max(lengths)
            # seq_tensor = torch.FloatTensor(batch_length, len(seqs), embed_size).fill_(PAD)
            #
            # for i, s in enumerate(seqs):
            #     end_seq = lengths[i]
            #     seq_tensor[:end_seq, i, :].copy_(s[:end_seq])
            # return pack_padded_sequence(Variable(seq_tensor, requires_grad=False), lengths), words

        return collate


    def get_dataloder(self, batch_size, train, num_workers=4):
        return DataLoader(self, batch_size=batch_size, shuffle=train,
                          collate_fn=self.create_collate_fn(), num_workers=num_workers)


if __name__ == "__main__":
    train_json = "./data/squad/train-v1.1.json"
    dataset = SQuAD(train_json)
    print(dataset[5])
