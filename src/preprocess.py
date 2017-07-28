# -*- coidng: utf-8 -*-

"""
dataset(dict):
    data(list):
       paragraphs(list):
            context(str):
            qas(list):
                answers(list):  training set has 1 answer, where dev set has 3 answers
                    answer_start(int):
                    text(str):
                id(str):
                question(str):
       title(str):
    version(str):
"""

import numpy as np

import json
import tensorflow as tf
import io
import spacy

from collections import Counter

version = "1.1"
min_count = 5
VOCAB_SIZE = 50000
ENCODER_INPUT_LEN = 440
DECODER_INPUT_LEN = 413

spacy_en = spacy.load('en')

tokenizer = lambda s: [tok.text for tok in spacy_en.tokenizer(s)]


_PAD = "_PAD"
_UNK = "_UNK"
_SPL = "_SPL"
_START_VOCAB = [_PAD, _UNK, _SPL]

PAD_ID = 0
UNK_ID = 1
SPL_ID = 2

_B = "_B"
_I = "_I"
_O = "_O"
_GO = "_GO"
_EOS = "_EOS"
_DECODE_VOCAB = [_PAD, _B, _I, _O, _GO, _EOS]

B_ID = 1
I_ID = 2
O_ID = 3
GO_ID = 4
EOS_ID = 5


def load_bin_vec(fname):
    embedding = []
    word2idx = {}
    idx2word = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, emb_size = list(map(int, header.split()))
        binary_len = np.dtype('float32').itemsize * emb_size
        for line in range(vocab_size):
            word = []
            ch = f.read(1)
            while ch != ' ':
                if ch != '\n':
                    word.append(ch)
                ch = f.read(1)

            word = ''.join(word)
            embedding.append(np.fromstring(f.read(binary_len), dtype='float32'))
            word2idx[word] = line
            idx2word[line] = word

        embedding = np.array(embedding)

    return embedding, word2idx, idx2word


def parse_dataset(train_path, dev_path):
    with open(train_path, "r") as f_train, open(dev_path, "r") as f_dev:
        train_set = json.load(f_train)
        dev_set = json.load(f_dev)
        assert (train_set['version'] == version and dev_set['version'] == version)
        return [train_set, dev_set]


def parse2text(datasets, text_path):
    with io.open(text_path, "w", encoding="utf-8") as f_text:
        text_list = []
        for dataset in datasets:
            for article in dataset['data']:
                for paragraph in article['paragraphs']:
                    text_list.append(paragraph['context'])
                    for question in paragraph['qas']:
                        text_list.append(question['question'])
        f_text.write("\n".join(text_list))



def tokenized_by_answer(context, text, start):
    fore = context[:start]
    mid = context[start: start + len(text)]
    after = context[start + len(text):]

    tokenized_fore = tokenizer(fore)
    tokenized_mid = tokenizer(mid)
    tokenized_after = tokenizer(after)
    tokenized_text = tokenizer(text)

    for i, j in zip(tokenized_text, tokenized_mid):
        if i != j:
            return None

    words = []
    words.extend(tokenized_fore)
    words.extend(tokenized_mid)
    words.extend(tokenized_after)
    indies = list(range(len(tokenized_fore),len(tokenized_fore) + len(tokenized_mid)))
    return words, indies

def padding(sent, target_len):
    sent = ["</s>"] + sent + ["</b>"] * (target_len - len(sent)) + ["</e>"]
    return sent

def build_corpus(datasets):
    seq_len = 766 # + 2
    q_len = 60
    a_len = 46
    vocab = Counter()
    train_set = []
    dev_set = []
    c = Counter()

    for tokenized_set, dataset in zip([train_set, dev_set], datasets):
        for article in dataset['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    id = qa['id']
                    question = qa['question']
                    answer = qa['answers'][0]
                    text = answer['text']
                    start = int(answer['answer_start'])

                    return_val = tokenized_by_answer(context, text, start)
                    if return_val:
                        tokenized_context, indies = return_val
                        # if len(indies) > 40 or len(tokenized_context) > 413:
                        #     continue
                        tokenized_question = tokenizer(question)
                        c.update([len(tokenized_context) + len(tokenized_question)])
                    #     tokenized_context = padding(tokenized_context, seq_len)
                    #     tokenized_question = padding(tokenized_question, q_len)
                        vocab.update(tokenized_context)
                        vocab.update(tokenized_question)
                        tokenized_set.append((tokenized_context, tokenized_question, indies))
                    else:
                        print("tokenize error")
    print(c)
    return train_set, dev_set, vocab


def build_vocab(vocab):
    new_vocab = {}
    reversed_vocab = {}
    for index, word in enumerate(_START_VOCAB):
        new_vocab[word] = index
        reversed_vocab[index] = word
    for index, (word, count) in enumerate(vocab.most_common(VOCAB_SIZE - len(_START_VOCAB)), len(_START_VOCAB)):
        new_vocab[word] = index
        reversed_vocab[index] = word
    return new_vocab, reversed_vocab

def build_vector(seq, vocab):
    vec = [vocab[w] if w in vocab else vocab[_UNK] for w in seq]
    return vec

def build_qa_pairs(dataset, vocab):
    encoder_inputs = []
    decoder_inputs = []
    for context, question, indies in dataset:
        encoder_input = question + [_SPL] + context
        # encoder_pad = [_PAD]*(ENCODER_INPUT_LEN - len(encoder_input))
        encoder_inputs.append(build_vector(encoder_input, vocab))

        decoder_input = []
        for i in range(len(context)):
            if i in indies:
                decoder_input.append(I_ID) #I
            else:
                decoder_input.append(O_ID) #O
        decoder_input[indies[0]] = B_ID # B
        decoder_inputs.append(decoder_input)
    return encoder_inputs, decoder_inputs



def main():
    # embedding, word2idx, idx2word = load_bin_vec("./data/vectors")

    datasets = parse_dataset('./data/squad/train-v1.1.json', './data/squad/dev-v1.1.json')
    # parse2text(datasets, "./data/text.txt")

    train_set, dev_set, vocab = build_corpus(datasets)
    vocab, reversed_vocab = build_vocab(vocab)

    with io.open("./data/vocab%d.en" % VOCAB_SIZE, "w", encoding="utf-8") as f_vocab:
        vocab_list = sorted(vocab, key=vocab.get)
        for word in vocab_list:
            f_vocab.write(word + str("\n"))

    train_enc_inputs, train_dec_inputs = build_qa_pairs(train_set, vocab)
    dev_enc_inputs , dev_dec_inputs = build_qa_pairs(dev_set, vocab)
    with tf.gfile.GFile("./data/train.source", mode="w") as train_source:
        with tf.gfile.GFile("./data/train.target", mode="w") as train_target:
            for input in train_enc_inputs:
                train_source.write(" ".join(map(str,input)) + "\n")
            for input in train_dec_inputs:
                train_target.write(" ".join(map(str,input)) + "\n")

    with tf.gfile.GFile("./data/dev.source", mode="w") as dev_source:
        with tf.gfile.GFile("./data/dev.target", mode="w") as dev_target:
            for input in dev_enc_inputs:
                dev_source.write(" ".join(map(str,input)) + "\n")
            for input in dev_dec_inputs:
                dev_target.write(" ".join(map(str,input)) + "\n")

if __name__ == "__main__":
    main()