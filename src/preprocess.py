import functools
import json
import os
import random
from collections import Counter

from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchtext import vocab
import dill
import nltk
import progressbar
import spacy
import torch
import torchtext
from torchtext import data
import numpy as np
from collections import namedtuple

# only draw 100 examples for developing/debugging
DEBUG = True
DEBUG_LEN = 100

UNK = 0
PAD = 1
SOS = 2
EOS = 3


class RawExample(object):
    pass

def read_train_json(path):
    with open(path) as fin:
        data = json.load(fin)
    examples = []
    context_list = []

    for topic in data["data"]:
        title = topic["title"]
        for p in topic['paragraphs']:
            qas = p['qas']
            context = p['context']
            context_list.append((context, len(qas)))
            for qa in qas:
                question = qa["question"]
                answers = qa["answers"]
                question_id = qa["id"]
                for ans in answers:
                    answer_start = int(ans["answer_start"])
                    answer_text = ans["text"]
                    e = RawExample()
                    e.title = title
                    e.context_id = len(context_list) - 1
                    e.question = question
                    e.question_id = question_id
                    e.answer_start = answer_start
                    e.answer_text = answer_text
                    # lines.append([title, len(context_list) - 1, question, question_id, answer_start, answer_text])
                    examples.append(e)

                    if DEBUG and len(examples) >= DEBUG_LEN:
                        return examples, context_list

    return examples, context_list


def read_dev_json(path):
    with open(path) as fin:
        data = json.load(fin)
    examples = []
    context_list = []

    for topic in data["data"]:
        title = topic["title"]
        for p in topic['paragraphs']:
            qas = p['qas']
            context = p['context']
            context_list.append((context, len(qas)))

            for qa in qas:
                question = qa["question"]
                answers = qa["answers"]
                question_id = qa["id"]
                answer_start_list = [ans["answer_start"] for ans in answers]
                c = Counter(answer_start_list)
                most_common_answer, freq = c.most_common()[0]
                answer_text = None
                answer_start = None
                if freq > 1:
                    for i, ans_start in enumerate(answer_start_list):
                        if ans_start == most_common_answer:
                            answer_text = answers[i]["text"]
                            answer_start = answers[i]["answer_start"]
                            break
                else:
                    answer_text = answers[random.choice(range(len(answers)))]["text"]
                    answer_start = answers[random.choice(range(len(answers)))]["answer_start"]

                e = RawExample()
                e.title = title
                e.context_id = len(context_list) - 1
                e.question = question
                e.question_id = question_id
                e.answer_start = answer_start
                e.answer_text = answer_text
                # examples.append([title, len(context_list) - 1, question, question_id, answer_text])
                examples.append(e)

                if DEBUG and len(examples) >= DEBUG_LEN:
                    return examples, context_list

    return examples, context_list


def tokenized_by_answer(context, answer_text, answer_start, tokenizer):
    """
    Locate the answer token-level position after tokenizing as the original location is based on
    char-level

    snippet from: https://github.com/haichao592/squad-tf/blob/master/preprocess.py

    :param context:  passage
    :param answer_text:     context/passage
    :param answer_start:    answer start position (char level)
    :param tokenizer: tokenize function
    :return: tokenized passage, answer start index, answer end index (inclusive)
    """
    fore = context[:answer_start]
    mid = context[answer_start: answer_start + len(answer_text)]
    after = context[answer_start + len(answer_text):]

    tokenized_fore = tokenizer(fore)
    tokenized_mid = tokenizer(mid)
    tokenized_after = tokenizer(after)
    tokenized_text = tokenizer(answer_text)

    for i, j in zip(tokenized_text, tokenized_mid):
        if i != j:
            return None

    words = []
    words.extend(tokenized_fore)
    words.extend(tokenized_mid)
    words.extend(tokenized_after)
    answer_start_token, answer_end_token = len(tokenized_fore), len(tokenized_fore) + len(tokenized_mid) - 1
    return words, answer_start_token, answer_end_token


def create_padded_batch(max_length=5000, batch_first=False, sort=False, pack=False):
    def collate(seqs):
        if not torch.is_tensor(seqs[0]):
            return tuple([collate(s) for s in zip(*seqs)])
        if sort or pack:  # packing requires a sorted batch by length
            seqs.sort(key=len, reverse=True)
        lengths = [min(len(s), max_length) for s in seqs]
        batch_length = max(lengths)
        seq_tensor = torch.LongTensor(batch_length, len(seqs)).fill_(PAD)
        for i, s in enumerate(seqs):
            end_seq = lengths[i]
            seq_tensor[:end_seq, i].copy_(s[:end_seq])
        if batch_first:
            seq_tensor = seq_tensor.t()
        if pack:
            return pack_padded_sequence(seq_tensor, lengths, batch_first=batch_first)
        else:
            return (seq_tensor, lengths)

    return collate


def get_word_counter(*seqs):
    counter = {}
    for seq in seqs:
        for sentence in seq:
            for word in sentence:
                counter.setdefault(word, 0)
                counter[word] += 1
    return counter


def truncate_word_counter(word_counter, max_symbols):
    words = [(freq, word) for word, freq in word_counter.items()]
    words.sort()
    return {word: freq for freq, word in words[:max_symbols]}


class SQuAD(Dataset):
    def __init__(self, path, word_embedding=None, char_embedding=None, embedding_cache_root=None, split="train",
                 tokenization="nltk", insert_start="<SOS>", insert_end="<EOS>", char_level=True):
        self.insert_start = insert_start
        self.insert_end = insert_end
        self._set_tokenizer(tokenization)
        self.examples_raw, context_with_counter = read_train_json(path)
        self.embedding_cache_root = embedding_cache_root
        tokenized_context = [self.tokenizer(doc) for doc, count in context_with_counter]
        tokenized_question = [self.tokenizer(e.question) for e in self.examples_raw]

        word_counter = get_word_counter(tokenized_context, tokenized_question)

        if word_embedding is None:
            word_embedding = ("./data/embedding/word", "glove.840B", 300)

        self.itos, self.stoi, self.wv_vec = self._build_vocab(word_counter, word_embedding, char_embedding)

        self.numeralized_question = [self._numeralize_seq(question) for question in tokenized_question]
        self.numeralized_context = [self._numeralize_seq(context) for context in tokenized_context]

    def _numeralize_seq(self, seq):
        if self.insert_start is not None:
            result = [self.insert_start]
        else:
            result = []
        for word in seq:
            result.append(self.stoi.get(word, 0))
        if self.insert_end is not None:
            result.append(self.insert_end)
        return result

    def __getitem__(self, idx):
        question = self.numeralized_question[idx]
        context_id = self.examples_raw[idx].context_id
        context = self.numeralized_context[context_id]
        answer_start = self.examples_raw[idx].answer_start
        answer_end = self.examples_raw[idx].answer_start

        return question, context, answer_start, answer_end

    def __len__(self):
        return len(self.examples_raw)

    def read_embedding(self, word_embedding):
        root, word_type, dim = word_embedding
        wv_dict, wv_vectors, wv_size = vocab.load_word_vectors(root, word_type, dim)
        return wv_dict, wv_vectors, wv_size

    def _build_vocab(self, word_counter, word_embedding, char_embedding=None, specials=None):
        """
        :param word_counter: counter of words in dataset
        :param word_embedding: word_embedding config: (root, word_type, dim)
        :param char_embedding: char_embedding config: (root, word_type, dim)
        :param specials: special tokens in vocab, such as UNK, PAD, SOS, EOS
        :return: itos, stoi, vectors
        """

        wv_dict, wv_vectors, wv_size = self.read_embedding(word_embedding)

        # TODO: create word dict of data and feed into char_level embedding.
        #
        #
        # if char_embedding is not None:
        #     char_dict, char_vectors, char_vec_size = self.read_embedding(char_embedding)
        #     char_level_embedding = embedding_char_level(word_counter.keys(), char_dict, char_vectors,
        #                                                 cache_path=self.embedding_cache_root)

        # special tokens
        if specials is None:
            specials = ["<UNK>", "<PAD>", "<SOS>", "<EOS>"]

        # embedding size = glove vector size
        embed_size = wv_vectors.size(1)
        print("word embedding size: %d" % embed_size)

        # build itos and stoi
        words_in_dataset = sorted(word_counter.keys(), key=lambda x: word_counter[x], reverse=True)
        itos = specials[:]
        stoi = {}

        itos.extend(words_in_dataset)
        for idx, word in enumerate(words_in_dataset):
            stoi[word] = idx

        # build vectors
        vectors = torch.zeros([len(itos), embed_size])
        for word, idx in stoi.items():
            idx_in_glove = wv_dict.get(word, None)
            if idx_in_glove is not None:
                vectors[idx, :wv_size].copy_(wv_vectors[idx_in_glove])
        return itos, stoi, vectors

    def _set_tokenizer(self, tokenization):
        if tokenization == "nltk":
            self.tokenizer = nltk.word_tokenize

        elif tokenization == "spacy":
            spacy_en = spacy.load("en")

            def spacy_tokenizer(seq):
                return [w.text for w in spacy_en(seq)]

            self.tokenizer = spacy_tokenizer

        else:
            raise ValueError("Incorrect tokenizing method %s" % tokenization)

    def get_dataloder(self, batch_size, train, num_workers=4):
        # TODO: Need padding function
        return DataLoader(batch_size=batch_size, shuffle=train)


if __name__ == "__main__":
    train_json = "./data/squad/train-v1.1.json"
    dataset = SQuAD(train_json)




