from functools import partial

import nltk
import spacy
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils.utils import read_train_json, read_dev_json, tokenized_by_answer, sort_idx


def padding(seqs, pad, batch_first=False):
    """

    :param seqs: tuple of seq_length x dim
    :return: seq_length x Batch x dim
    """
    lengths = [len(s) for s in seqs]

    seqs = [torch.Tensor(s) for s in seqs]
    batch_length = max(lengths)
    seq_tensor = torch.LongTensor(batch_length, len(seqs)).fill_(pad)
    for i, s in enumerate(seqs):
        end_seq = lengths[i]
        seq_tensor[:end_seq, i].copy_(s[:end_seq])
    if batch_first:
        seq_tensor = seq_tensor.t()
    return (seq_tensor, lengths)


class Documents(object):
    """
        Helper class for organizing and sorting seqs

        should be batch_first for embedding
    """

    def __init__(self, tensor, lengths):
        self.original_lengths = lengths
        sorted_lengths_tensor, self.sorted_idx = torch.sort(torch.LongTensor(lengths), dim=0, descending=True)

        self.tensor = tensor.index_select(dim=0, index=self.sorted_idx)

        self.lengths = list(sorted_lengths_tensor)
        self.original_idx = torch.LongTensor(sort_idx(self.sorted_idx))

        self.mask_original = torch.zeros(*self.tensor.size())
        for i, length in enumerate(self.original_lengths):
            self.mask_original[i][:length].fill_(1)

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


class SQuAD(Dataset):
    def __init__(self, path, itos, stoi, itoc, ctoi, tokenizer="nltk", split="train",
                 debug_mode=False, debug_len=50):

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

    def _char_level_numeralize(self, tokenized_doc):
        result = []
        for word in tokenized_doc:
            result.append(self._numeralize_word_seq(word, self.ctoi))
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

            questions_tensor, question_lengths = padding(questions, this.PAD, batch_first=batch_first)
            passages_tensor, passage_lengths = padding(passages, this.PAD, batch_first=batch_first)

            # TODO: implement char level embedding

            question_document = Documents(questions_tensor, question_lengths)
            passages_document = Documents(passages_tensor, passage_lengths)

            if this.split == "train":
                return question_ids, question_document, passages_document, torch.LongTensor(answers_positions), answer_texts

            else:
                return question_ids, question_document, passages_document, passage_tokenized

        return partial(collate, this=self)

    def get_dataloader(self, batch_size, num_workers=4, shuffle=True, batch_first=True, pin_memory=False):
        """

        :param batch_first:  Currently, it must be True as nn.Embedding requires batch_first input
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory)
