import torch
from torch.autograd import Variable

from model import WordEmbedding
from utils import sort_idx




def test_DoubleEmbedding():
    train_json = "./data/squad/train-v1.1.json"
    from dataset import SQuAD
    dataset = SQuAD(train_json, debug_mode=True)
    dataloader = dataset.get_dataloader(5, shuffle=False)

    char_embedding = {"embedding_weights":dataset.cv_vec, "padding_idx":dataset.PAD, "update":True,
                      "bidirectional":True, "cell_type":"gru", "output_dim":300}

    word_embedding = {"embedding_weights":dataset.wv_vec, "padding_idx":dataset.PAD, "update":False}

    embedding = WordEmbedding(char_embedding, word_embedding)

    for batch in dataloader:
        question_ids, words, questions, contexts, answers, answers_texts = batch

        words.to_variable()
        questions.to_variable()
        contexts.to_variable()

        result = embedding(words, questions, contexts)

        print(result[1])
        break