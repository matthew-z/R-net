import os

import nltk
import spacy
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import defaultdict
from utils import read_train_json, get_counter, read_dev_json, read_embedding, tokenized_by_answer

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

class SQuAD(Dataset):
    def __init__(self, path, word_embedding=None, char_embedding=None, embedding_cache_root=None, split="train",
                 tokenization="nltk", insert_start="<SOS>", insert_end="<EOS>",
                 debug_mode=False, debug_len=50):

        self.UNK = 0
        self.PAD = 1
        self.SOS = 2
        self.EOS = 3

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
        self.specials_set = set(self.specials)

        self.insert_start = insert_start
        self.insert_end = insert_end
        self._set_tokenizer(tokenization)
        self.split = split

        # read and parsing raw data from json
        if self.split == "train":
            self.examples_raw, context_with_counter = read_train_json(path, debug_mode, debug_len)
            tokenized_context, self.answers_positions = self.tokenize_context_with_answer(context_with_counter)
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

    def tokenize_context_with_answer(self, context_with_counter):
        tokenized_contexts = []
        ansewrs_postion = []
        for example in self.examples_raw:
            context, _ = context_with_counter[example.context_id]
            tokenized_context, answer_start, answer_end = tokenized_by_answer(context, example.answer_text,
                                                                              example.answer_start, self.tokenizer)
            tokenized_contexts.append(tokenized_context)
            ansewrs_postion.append((answer_start, answer_end))
        return tokenized_contexts, ansewrs_postion

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

        if self.split == "train":
            context = self.numeralized_context[idx]
        else:
            context = self.numeralized_context[self.examples_raw[idx].context_id]

        distinct_words = set()
        for word_idx in question + context:
            if word_idx in distinct_words:
                continue
            distinct_words.add(word_idx)
        answer_text = self.examples_raw[idx].answer_text
        question_id = self.examples_raw[idx].question_id

        if self.split == "train":
            answers_position = self.answers_positions[idx]
            return question_id, distinct_words, question, context, answers_position, answer_text
        else:
            return question_id, distinct_words, question, context

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

        def get_unk():
            return self.UNK

        stoi = defaultdict(get_unk)

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

        def word_to_chars(word_idx):
            if word_idx in self.specials_set:
                return [word_idx]
            return [self.ctoi[char] for char in self.itos[word_idx]]

        def collate(examples):
            if self.split == "train":
                question_ids, distinct_words_sets, questions, contexts, answers_positions, answer_texts = zip(*examples)
            else:
                question_ids, distinct_words_sets, questions, contexts = zip(*examples)

            # word idx and chars for char-level encoding
            words = set()
            for word_set in distinct_words_sets:
                words |= word_set
            words = list(words)
            words_in_chars = [word_to_chars(word_idx) for word_idx in words]
            words_in_chars_tensor = padding(words_in_chars, self.PAD)

            questions_tensor = padding(questions, self.PAD)
            contexts_tensor = padding(contexts, self.PAD)
            if self.split == "train":
                answers_positions = torch.LongTensor(answers_positions)
                return question_ids, words_in_chars_tensor, questions_tensor, contexts_tensor, answers_positions, answer_texts
            else:
                return question_ids, words_in_chars_tensor, questions_tensor, contexts_tensor

        return collate

    def get_dataloader(self, batch_size, num_workers=4):
        return DataLoader(self, batch_size=batch_size, shuffle=self.split == "train",
                          collate_fn=self.create_collate_fn(), num_workers=num_workers)


if __name__ == "__main__":
    def test_train_data():
        train_json = "./data/squad/train-v1.1.json"
        dataset = SQuAD(train_json)
        dataloader = dataset.get_dataloader(5, True)

        for batch in dataloader:
            words_in_chars, questions_tensor, contexts_tensor, answers, answer_text = batch

            for i, row in enumerate(answers):
                start, end = row
                print(start, end)
                print(" ".join([dataset.itos[idx] for idx in contexts_tensor[0][start:end + 1, i]]))
                print(answer_text[i])

                print()


    def test_dev_data():
        dev_json = "./data/squad/dev-v1.1.json"
        dataset = SQuAD(dev_json, split="dev")
        dataloader = dataset.get_dataloader(5, True)

        for batch in dataloader:
            question_ids, words_in_chars, questions, contexts = batch

            questions_tensor, lengths = questions
            for i, l in enumerate(lengths):
                print(question_ids[i])
                print(" ".join([dataset.itos[idx] for idx in questions_tensor[:l, i]]))


    test_dev_data()