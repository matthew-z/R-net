import functools
import json
import os
import random
from collections import Counter

from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset
from torchtext import vocab
import dill
import nltk
import progressbar
from r_net.embedding import embedding_char_level
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


class Example(object):
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
                    e = Example()
                    e.title = title
                    e.context_id = len(context_list) - 1
                    e.question = question
                    e.question_id = question_id
                    e.answer_start = answer_text
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
                if freq > 1:
                    for i, ans_start in enumerate(answer_start_list):
                        if ans_start == most_common_answer:
                            answer_text = answers[i]["text"]
                            break
                else:
                    answer_text = answers[random.choice(range(len(answers)))]["text"]

                e = Example()
                e.title = title
                e.context_id = len(context_list) - 1
                e.question = question
                e.question_id = question_id
                e.answer_start = answer_text
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
        for word in seq:
            counter.setdefault(word, 0)
            counter[word] += 1
    return counter


def truncate_word_counter(word_counter, max_symbols):
    words = [(freq, word) for word, freq in word_counter.items()]
    words.sort()
    return {word: freq for freq, word in words[:max_symbols]}


class SQuAD(Dataset):
    def __init__(self, path, word_embedding, char_embedding, embedding_cache_root=None, split="train",
                 tokenization="nltk", insert_start=SOS, insert_end=EOS, char_level=True):
        self._set_tokenizer(tokenization)
        examples_raw, context_with_counter = read_train_json(path)
        self.embedding_cache_root = embedding_cache_root
        self.tokenized_context = [self.tokenizer(doc) for doc, count in context_with_counter]
        self.tokenized_question = [self.tokenizer(e.question) for e in examples_raw]

        word_counter = get_word_counter(self.tokenized_context, self.tokenized_question)
        self._build_vocab(word_counter, word_embedding, char_embedding)

    def read_embedding(self, word_embedding):
        root, word_type, dim = word_embedding
        wv_dict, wv_vectors, wv_size = vocab.load_word_vectors(root, word_type, dim)
        return wv_dict, wv_vectors, wv_size

    def _build_vocab(self, word_counter, word_embedding, char_embedding=None):
        wv_dict, wv_vectors, wv_size = self.read_embedding(word_embedding)

        # TODO create word dict of data and feed into char_level embedding.


        if char_embedding is not None:
            char_dict, char_vectors, char_vec_size = self.read_embedding(char_embedding)
            char_level_embedding = embedding_char_level(word_counter.keys(), char_dict, char_vectors,
                                                        cache_path=self.embedding_cache_root)

        embed_size = wv_vectors.size(1)
        print("word embedding size: %d" % embed_size)

        wv_size = len(word_counter)
        self.itos = []
        self.stoi = {}
        self.vectors = torch.zeros([wv_size, embed_size])

    def _set_tokenizer(self, tokenization):
        if tokenization == "nltk":
            self.tokenizer = nltk.word_tokenize

        elif tokenization == "spacy":
            spacy_en = spacy.load("en")

            def spacy_tokenizer(seq):
                return [w.text for w in spacy_en(seq)]

            self.tokenizer = spacy_tokenizer

        raise ValueError("Incorrect tokenization method %s" % tokenization)
if __name__ == "__main__":
    train_json = "./data/squad/train-v1.1.json"
