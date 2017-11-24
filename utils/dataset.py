from functools import partial

import nltk
import spacy
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils.utils import read_train_json, read_dev_json, tokenized_by_answer, sort_idx


def padding(seqs, pad, output_batch_first=True):
    """
    :param seqs: sequence of seq_length x dim
    """
    lengths = [len(s) for s in seqs]


    seqs = [torch.Tensor(s) for s in seqs]
    batch_length = max(lengths)
    seq_tensor = torch.LongTensor(batch_length, len(seqs)).fill_(pad)
    mask = torch.LongTensor(seq_tensor.size()).fill_(0)

    for i, s in enumerate(seqs):
        end_seq = lengths[i]
        seq_tensor[:end_seq, i].copy_(s[:end_seq])
        mask[:end_seq, i].fill_(1)
    if output_batch_first:
        seq_tensor = seq_tensor.t()
        mask = mask.t()
    return (seq_tensor, mask,  lengths)


def padding_2d(seqs, pad):
    """
    :param seqs:  sequence of sentences, each word is a list of chars
    """
    sentence_length = 0
    word_length = 0
    lengths = []
    for seq in seqs:
        sentence_length = max(sentence_length, len(seq))
        for word in seq:
            word_length = max(word_length, len(word))

    tensor = torch.LongTensor(len(seqs), sentence_length, word_length).fill_(pad)
    mask = torch.LongTensor(tensor.size()).fill_(0)
    for i, s in enumerate(seqs):
        lengths.append([])
        for j, w in enumerate(s):
            tensor[i, j, :len(w)].copy_(torch.LongTensor(w))
            mask[i, j, :len(w)].fill_(1)
            lengths[-1].append(len(w))
    return tensor, mask, lengths


class Documents(object):
    """
        Helper class for organizing and sorting seqs

        should be batch_first for embedding
    """

    def __init__(self, questions, pad=0, batch_first=True):
        tensor, self.mask_original, lengths = padding(questions, pad, output_batch_first=batch_first)

        self.original_lengths = lengths
        sorted_lengths_tensor, self.sorted_idx = torch.sort(torch.LongTensor(lengths), dim=0, descending=True)
        self.tensor = tensor.index_select(dim=0, index=self.sorted_idx)
        self.lengths = list(sorted_lengths_tensor)
        self.original_idx = torch.LongTensor(sort_idx(self.sorted_idx))


    def variable(self, volatile=False):
        self.tensor = Variable(self.tensor, volatile=volatile)
        self.sorted_idx = Variable(self.sorted_idx, volatile=volatile)
        self.original_idx = Variable(self.original_idx, volatile=volatile)
        self.mask_original = Variable(self.mask_original, volatile=volatile)
        return self

    def cuda(self, *args, **kwargs):
        if torch.cuda.is_available():
            self.sorted_idx = self.sorted_idx.cuda(*args, **kwargs)
            self.original_idx = self.original_idx.cuda(*args, **kwargs)
            self.mask_original = self.mask_original.cuda(*args, **kwargs)

        return self

    def restore_original_order(self, sorted_tensor, batch_dim):
        return sorted_tensor.index_select(dim=batch_dim, index=self.original_idx)

    def to_sorted_order(self, original_tensor, batch_dim):
        return original_tensor.index_select(dim=batch_dim, index=self.sorted_idx)


class CharDocuments(object):
    def __init__(self, sentence_char, PAD=0):
        self.tensor, self.mask_original, self.lengths = padding_2d(sentence_char, PAD)

    def variable(self, volatile=False):
        self.tensor = Variable(self.tensor, volatile=volatile)
        self.mask_original = Variable(self.mask_original, volatile=volatile)
        return self

    def cuda(self,  *args, **kwargs):
        if torch.cuda.is_available():
            self.tensor = self.tensor.cuda(*args, **kwargs)
            self.mask_original = self.mask_original.cuda(*args, **kwargs)
        return self


class SQuAD(Dataset):
    def __init__(self, path, itos, stoi, itoc, ctoi,
                 tokenizer="nltk", split="train",
                 debug_mode=False, debug_len=50):
        """ Build a dataset to fetch batch"""

        self.insert_start = stoi.get("<SOS>", None)
        self.insert_end = stoi.get("<EOS>", None)
        self.UNK = stoi.get("<UNK>", None)
        self.PAD = stoi.get("<PAD>", None)
        self.stoi = stoi
        self.ctoi = ctoi
        self.itos = itos
        self.itoc = itoc
        self.split = split
        self._set_tokenizer(tokenizer)

        # Read and parsing raw data from json
        # Tokenizing with answer may result in different tokenized passage even the passage is the same one.
        # So we tokenize passage for each question in train split
        if self.split == "train":
            self.examples = read_train_json(path, debug_mode, debug_len)
            self._tokenize_passage_with_answer_for_train()
        else:
            self.examples = read_dev_json(path, debug_mode, debug_len)
            for e in self.examples:
                e.tokenized_passage = self.tokenizer(e.passage)

        for e in self.examples:
            e.tokenized_question = self.tokenizer(e.question)
            e.numeralized_question = self._numeralize_word_seq(e.tokenized_question, self.stoi)
            e.numeralized_passage = self._numeralize_word_seq(e.tokenized_passage, self.stoi)
            e.numeralized_question_char = self._char_level_numeralize(e.tokenized_question)
            e.numeralized_passage_char = self._char_level_numeralize(e.tokenized_passage)

    def _tokenize_passage_with_answer_for_train(self):
        for example in self.examples:
            example.tokenized_passage, answer_start, answer_end = tokenized_by_answer(example.passage, example.answer_text,
                                                                              example.answer_start, self.tokenizer)
            example.answer_position = (answer_start, answer_end)

    def _char_level_numeralize(self, tokenized_doc, insert_sos=False, insert_eos=False):
        result = [] if not insert_sos else [self.insert_start]
        for word in tokenized_doc:
            result.append(self._numeralize_word_seq(word, self.ctoi))
        if insert_eos:
            result.append(self.insert_end)
        return result

    def _numeralize_word_seq(self, seq, to_index, insert_sos=False, insert_eos=False):
        if self.insert_start is not None and insert_sos:
            result = [self.insert_start]
        else:
            result = []
        for word in seq:
            result.append(to_index.get(word, 0))
        if self.insert_end is not None and insert_eos:
            result.append(self.insert_end)
        return result

    def __getitem__(self, idx):
        item = self.examples[idx]
        if self.split == "train":
            return (item, item.question_id, item.numeralized_question, item.numeralized_question_char,
                    item.numeralized_passage, item.numeralized_passage_char,
                    item.answer_position, item.answer_text, item.tokenized_passage)
        else:
            return (item, item.question_id, item.numeralized_question, item.numeralized_question_char,
                    item.numeralized_passage, item.numeralized_passage_char,
                    item.tokenized_passage)

    def __len__(self):
        return len(self.examples)

    def _set_tokenizer(self, tokenizer):
        """
        Set tokenizer

        :param tokenizer: tokenization method
        :return: None
        """
        if tokenizer == "nltk":
            self.tokenizer = nltk.word_tokenize
        elif tokenizer == "spacy":
            spacy_en = spacy.load("en")

            def spacy_tokenizer(seq):
                return [w.text for w in spacy_en(seq)]

            self.tokenizer = spacy_tokenizer
        else:
            raise ValueError("Invalid tokenizing method %s" % tokenizer)

    def _create_collate_fn(self, batch_first=True):

        def collate(examples, this):
            if this.split == "train":
                items, question_ids, questions, questions_char, passages, passages_char, answers_positions, answer_texts, passage_tokenized = zip(
                    *examples)
            else:
                items, question_ids, questions, questions_char, passages, passages_char, passage_tokenized = zip(*examples)

            # sentences are not sorted by length
            question_document = Documents(questions,this.PAD)
            passages_document = Documents(passages, this.PAD)
            question_char_document = CharDocuments(questions_char, this.PAD)
            passage_char_document = CharDocuments(passages_char, this.PAD)

            if this.split == "train":
                return (question_ids,
                        question_document, question_char_document,
                        passages_document,passage_char_document,
                        torch.LongTensor(answers_positions), answer_texts)

            else:
                return (question_ids,
                        question_document, question_char_document,
                        passages_document, passage_char_document,
                        passage_tokenized)

        return partial(collate, this=self)

    def get_dataloader(self, batch_size, num_workers=4, shuffle=True, batch_first=True, pin_memory=False):
        """

        :param batch_first:  Currently, it must be True as nn.Embedding requires batch_first input
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory)
