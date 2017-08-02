# Code From pytorch/examples
import json
import os
import random
import errno
from argparse import ArgumentParser
from collections import Counter

from torchtext import vocab

from preprocess import RawExample


def make_dirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext SNLI example')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_embed', type=int, default=300)
    parser.add_argument('--d_proj', type=int, default=300)
    parser.add_argument('--d_hidden', type=int, default=300)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--dev_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--dp_ratio', type=int, default=0.2)
    parser.add_argument('--no-bidirectional', action='store_false', dest='birnn')
    parser.add_argument('--preserve-case', action='store_false', dest='lower')
    parser.add_argument('--no-projection', action='store_false', dest='projection')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--data_cache', type=str, default=os.path.join(os.getcwd(), '.data_cache'))
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
    parser.add_argument('--word_vectors', type=str, default='glove.42B')
    parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args


def read_train_json(path, debug_mode, debug_len):
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
                    examples.append(e)

                    if debug_mode and len(examples) >= debug_len:
                        return examples, context_list

    return examples, context_list


def get_counter(*seqs):
    word_counter = {}
    char_counter = {}
    for seq in seqs:
        for doc in seq:
            for word in doc:
                word_counter.setdefault(word, 0)
                word_counter[word] += 1
                for char in word:
                    char_counter.setdefault(char, 0)
                    char_counter[char] += 1
    return word_counter, char_counter


def read_dev_json(path, debug_mode, debug_len):
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
                examples.append(e)

                if debug_mode and len(examples) >= debug_len:
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


def truncate_word_counter(word_counter, max_symbols):
    words = [(freq, word) for word, freq in word_counter.items()]
    words.sort()
    return {word: freq for freq, word in words[:max_symbols]}


def read_embedding(word_embedding):
    root, word_type, dim = word_embedding
    wv_dict, wv_vectors, wv_size = vocab.load_word_vectors(root, word_type, dim)
    return wv_dict, wv_vectors, wv_size
