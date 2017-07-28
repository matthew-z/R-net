import functools
import json
import os
import random
from collections import Counter

import dill
import nltk
import progressbar
import spacy
from torchtext import data

spacy_en = spacy.load("en")

DEBUG = True


def read_train_json(path):
    with open(path) as fin:
        data = json.load(fin)
    lines = []
    for topic in data["data"]:
        title = topic["title"]
        for p in topic['paragraphs']:
            qas = p['qas']
            context = p['context']
            for qa in qas:
                question = qa["question"]
                answers = qa["answers"]
                for ans in answers:
                    answer_start = int(ans["answer_start"])
                    answer_text = ans["text"]
                    lines.append([title, context, question, answer_start, answer_text])
    return lines[:100] if DEBUG else lines


def read_dev_json(path):
    with open(path) as fin:
        data = json.load(fin)
    lines = []
    for topic in data["data"]:
        title = topic["title"]
        for p in topic['paragraphs']:
            qas = p['qas']
            context = p['context']
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

                lines.append([title, context, question, question_id, answer_text])
    return lines[:100] if DEBUG else lines


def read_test_json(path):
    with open(path) as fin:
        data = json.load(fin)
    lines = []
    for topic in data["data"]:
        title = topic["title"]
        for p in topic['paragraphs']:
            qas = p['qas']
            context = p['context']
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

                lines.append([title, context, question, question_id, answer_text])
    return lines[:100] if DEBUG else lines


def tokenized_by_answer(context, text, start, tokenizer):
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
    answer_start_token, answer_end_token = len(tokenized_fore), len(tokenized_fore) + len(tokenized_mid) - 1
    return words, answer_start_token, answer_end_token


def extract_test_examples(fields, path):
    examples = []
    bar = progressbar.ProgressBar()
    lines = read_test_json(path)
    for _, context, question, question_id in bar(lines):
        ex = data.Example.fromlist([context, question, question_id],
                                   fields)
        examples.append(ex)
    return examples


def extract_train_examples(fields, path, tokenizer):
    bar = progressbar.ProgressBar()
    examples = []
    lines = read_train_json(path)
    for _, context, question, answer_start, answer_text in bar(lines):
        question_tokens = tokenizer(question)
        passage_tokens, answer_start_token, answer_end_token = tokenized_by_answer(context, answer_text,
                                                                                   answer_start, tokenizer)
        ex = data.Example.fromlist([passage_tokens, question_tokens, answer_start_token, answer_end_token], fields)
        examples.append(ex)
    return examples


def get_dataset(json_path, cache_root="./data/cache", mode="train", tokenizer=None):
    examples, fields = build_examples_from_json(json_path, mode, tokenizer)

    # examples_file_name="%s%s.examples" % (mode, "_DEBUG" if DEBUG else "")
    # example_file = os.path.join(cache_root, examples_file_name)
    #
    # if os.path.exists(example_file):
    #     print("loading examples from %s" % example_file)
    #     examples = pickle.load(open(example_file, "rb"))
    # else:
    #     print("building examples %s" % example_file)
    #     examples = extract_method(fields, json_path)
    #     pickle.dump(examples, open(example_file, "wb"))

    squad_dataset = data.Dataset(examples, fields)

    # build voc
    squad_dataset.fields["passage"].build_vocab(squad_dataset, [x.question for x in squad_dataset.examples],
                                                wv_type='glove.840B', wv_dir="./data/embedding/glove_word/",
                                                unk_init="zero")

    squad_dataset.fields["question"].vocab = squad_dataset.fields["passage"].vocab

    #
    # squad_dataset.fields["question"].build_vocab(squad_dataset, [x.passage for x in squad_dataset.examples],
    #                                              wv_type='glove.840B', wv_dir="./data/embedding/glove_word/",
    #                                              unk_init="zero")

    dataset_file_name = "%s%s.dataset" % (mode, "_DEBUG" if DEBUG else "")

    # squad_dataset.fields["answer_text"].build_vocab(squad_dataset,[x.passage for x in squad_dataset.examples], wv_type='glove.840B',
    #                                   wv_dir="./data/embedding/glove_word/", unk_init="zero")
    squad_dataset.sort_key = lambda x: (len(x.question), len(x.passage))
    dill.dump(squad_dataset, open(os.path.join(cache_root, dataset_file_name), "wb"))
    return squad_dataset


def build_examples_from_json(json_path, mode, tokenizer):
    if tokenizer == "spacy":
        tokenizer = lambda s: [tok.text for tok in spacy_en.tokenizer(s)]
    elif tokenizer is None:
        tokenizer = nltk.word_tokenize
    if mode == "train":
        extract_method = functools.partial(extract_train_examples, tokenizer=tokenizer)

        fields = [("passage", data.Field(tokenize=tuple)),
                  ("question", data.Field(tokenize=tokenizer)),
                  ("answer_start", data.Field(sequential=False, use_vocab=False, preprocessing=int)),
                  ("answer_end", data.Field(sequential=False, use_vocab=False, preprocessing=int)),
                  # ("answer_text", data.Field(tokenize=tokenize))
                  ]

    elif mode == "dev":
        extract_method = extract_test_examples
        fields = [("passage", data.Field(tokenize=tokenizer)),
                  ("question", data.Field(tokenize=tokenizer)),
                  ("question_id", data.Field(sequential=False, use_vocab=False)),
                  # ("answer_text", data.Field(tokenize=tokenize))
                  ]

    elif mode == 'test':
        extract_method = extract_test_examples
        fields = [("passage", data.Field(tokenize=tokenizer)),
                  ("question", data.Field(tokenize=tokenizer)),
                  ("question_id", data.Field(sequential=False, use_vocab=False))]

    else:
        raise ValueError("Incorrect mode %s for building dataset" % mode)
    examples = extract_method(fields, json_path)
    return examples, fields


def get_iter_for_training(batch_size=128, device=None, tokenizer=None, cache_root="./data/cache"):
    train_json = "./data/squad/train-v1.1.json"
    dev_json = "./data/squad/dev-v1.1.json"

    train_dataset = get_dataset(json_path=train_json, mode="train", tokenizer=tokenizer, cache_root=cache_root)
    dev_dataset = get_dataset(json_path=dev_json, mode="dev", tokenizer=tokenizer, cache_root=cache_root)

    train_iter, dev_iter = data.BucketIterator.splits(
        (train_dataset, dev_dataset), batch_size=batch_size, device=device)

    return train_iter, dev_iter


train_json = "./data/squad/train-v1.1.json"
get_dataset(json_path=train_json, mode="train")
