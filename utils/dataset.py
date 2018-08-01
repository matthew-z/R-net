from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils.utils import read_train_json, read_dev_json, tokenized_by_answer, set_tokenizer
import progressbar

def padding(seqs, pad_value=0, output_batch_first=True):
    """
    :param seqs: sequence of seq_length x dim
    """
    lengths = [len(s) for s in seqs]

    seqs = [torch.Tensor(s) for s in seqs]
    batch_length = max(lengths)
    seq_tensor = torch.full([batch_length, len(seqs)], pad_value, dtype=torch.int64)
    mask = torch.zeros(seq_tensor.size(),dtype=torch.int64)

    for i, s in enumerate(seqs):
        end_seq = lengths[i]
        seq_tensor[:end_seq, i].copy_(s[:end_seq])
        mask[:end_seq, i].fill_(1)
    if output_batch_first:
        seq_tensor = seq_tensor.t()
        mask = mask.t()
    return (seq_tensor, mask, lengths)


def padding_2d(seqs,  pad_value=0):
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

    tensor = torch.full([len(seqs), sentence_length, word_length], pad_value,  dtype=torch.int64)
    mask = torch.zeros(tensor.size(),dtype=torch.int64)
    for i, s in enumerate(seqs):
        lengths.append([])
        for j, w in enumerate(s):
            tensor[i, j, :len(w)].copy_(torch.Tensor(w))
            mask[i, j, :len(w)].fill_(1)
            lengths[-1].append(len(w))
    return tensor, mask, lengths

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
        tokenizer = set_tokenizer(tokenizer)

        # Read and parsing raw data from json
        # Tokenizing with answer may result in different tokenized passage even the passage is the same one.
        # So we tokenize passage for each question in train split
        if self.split == "train":
            self.examples = read_train_json(path, debug_mode, debug_len)
            for example in self.examples:
                example.tokenized_passage, answer_start, answer_end = tokenized_by_answer(
                    example.passage,
                    example.answer_text,
                    example.answer_start,
                    tokenizer)
                example.answer_position = (answer_start, answer_end)
        else:
            self.examples = read_dev_json(path, debug_mode, debug_len)
            for example in self.examples:
                example.tokenized_passage = tokenizer(example.passage)

        for example in progressbar.progressbar(self.examples):
            example.tokenized_question = tokenizer(example.question)
            example.numeralized_question = self._numeralize_word_seq(example.tokenized_question, self.stoi)
            example.numeralized_passage = self._numeralize_word_seq(example.tokenized_passage, self.stoi)
            example.numeralized_question_char = self._char_level_numeralize(example.tokenized_question)
            example.numeralized_passage_char = self._char_level_numeralize(example.tokenized_passage)

        self.examples = [example for example in self.examples if len(example.tokenized_passage) <= 300]
        if self.split != "train":
            self.examples.sort(key=lambda example: len(example.numeralized_passage), reverse=True)

    def _char_level_numeralize(self, tokenized_doc, insert_sos=False, insert_eos=False):
        result = [] if not insert_sos else [self.insert_start]
        for word in tokenized_doc:
            result.append(self._numeralize_word_seq(word, self.ctoi))
        if insert_eos:
            result.append(self.insert_end)
        return result

    def _numeralize_word_seq(self, seq, to_index, insert_sos=False, insert_eos=False):
        result = [] if not insert_sos else [self.insert_start]
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
        return (item, item.question_id, item.numeralized_question, item.numeralized_question_char,
                item.numeralized_passage, item.numeralized_passage_char,
                item.tokenized_passage)

    def __len__(self):
        return len(self.examples)

    def _create_collate_fn(self, batch_first, device):
        if not batch_first:
            raise NotImplementedError

        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

        def collate(examples, this):
            if this.split == "train":
                items, question_ids, questions, questions_char, passages, \
                passages_char, answers_positions, answer_texts, passage_tokenized = zip(
                    *examples)
            else:
                items, question_ids, questions, questions_char, passages, \
                passages_char, passage_tokenized = zip(
                    *examples)

            # sentences are not sorted by length
            question, question_mask, _ = padding(questions, this.PAD)
            passage, passage_mask, _ = padding(passages, this.PAD)

            question_char, _, _ = padding_2d(questions_char,this.PAD)
            passage_char, _, _ = padding_2d(passages_char, this.PAD)

            if this.split == "train":
                answers_positions = torch.tensor(answers_positions, dtype=torch.int64)
                return (question_ids,
                        question,question_char, question_mask,
                        passage,passage_char, passage_mask,
                        answers_positions)

            return (question_ids,
                    question, question_char, question_mask,
                    passage, passage_char, passage_mask,
                    passage_tokenized)

        return partial(collate, this=self)

    def get_dataloader(self, batch_size, num_workers=4, shuffle=True, batch_first=True, pin_memory=False, device=None):
        """

        :param batch_first:  Currently, it must be True
        """
        if not batch_first:
            raise NotImplementedError

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=self._create_collate_fn(batch_first, device),
                          num_workers=num_workers, pin_memory=pin_memory)
